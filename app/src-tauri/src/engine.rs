//! Starts the Python engine (`clipsdex serve`) next to the app and stops it on exit.
//!
//! The app picks a random token and hands it over in `CLIPSDEX_API_TOKEN` (never on the
//! command line, where other programs could read it). The engine answers on 127.0.0.1
//! and prints one JSON line with its port when it is ready. It also watches its stdin:
//! the app holds that pipe open, so if the app crashes the pipe closes and the engine
//! stops by itself instead of running on in the background.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};

/// First start can take a while: `uv` may create the Python environment.
const START_TIMEOUT: Duration = Duration::from_secs(300);
/// How long the engine gets to finish after its stdin is closed, before it is killed.
const STOP_GRACE: Duration = Duration::from_secs(5);
const STDERR_LINES: usize = 40;
pub const STATUS_EVENT: &str = "engine-status";

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "lowercase")]
pub enum EngineStatus {
    Starting,
    Ready { url: String, token: String },
    Failed { message: String },
}

#[derive(Deserialize)]
struct ReadyLine {
    event: String,
    port: u16,
}

pub struct Engine {
    status: Mutex<EngineStatus>,
    child: Mutex<Option<Child>>,
    stdin: Mutex<Option<ChildStdin>>,
    /// Bumped by every start and stop: threads of an older run stay quiet.
    generation: AtomicU64,
}

impl Default for Engine {
    fn default() -> Self {
        Self {
            status: Mutex::new(EngineStatus::Starting),
            child: Mutex::new(None),
            stdin: Mutex::new(None),
            generation: AtomicU64::new(0),
        }
    }
}

impl Engine {
    pub fn status(&self) -> EngineStatus {
        self.status.lock().expect("engine status lock").clone()
    }
}

fn new_token() -> String {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).expect("the OS random number generator is unavailable");
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The ready line, or None for any other output.
pub fn parse_ready(line: &str) -> Option<u16> {
    let ready: ReadyLine = serde_json::from_str(line.trim()).ok()?;
    (ready.event == "ready" && ready.port != 0).then_some(ready.port)
}

fn engine_dir() -> PathBuf {
    // Development: the engine's uv project in this repository. Release builds ship it
    // with the app (step 8.1).
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("engine")
}

fn engine_command(token: &str) -> Command {
    let dir = engine_dir();
    let mut cmd = Command::new("uv");
    cmd.arg("run")
        .arg("--project")
        .arg(&dir)
        .args(["clipsdex", "serve", "--exit-with-stdin"])
        .current_dir(&dir)
        .env("CLIPSDEX_API_TOKEN", token)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if cfg!(debug_assertions) {
        // `tauri dev` serves the UI from Vite; let it call the API too.
        cmd.env("CLIPSDEX_API_EXTRA_ORIGINS", "http://localhost:1420");
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

fn set_status(app: &AppHandle, generation: u64, status: EngineStatus) {
    let engine = app.state::<Engine>();
    if engine.generation.load(Ordering::SeqCst) != generation {
        return; // a newer start or a stop happened since
    }
    *engine.status.lock().expect("engine status lock") = status.clone();
    let _ = app.emit(STATUS_EVENT, status);
}

fn failure(reason: &str, stderr: &Mutex<VecDeque<String>>) -> EngineStatus {
    let tail: Vec<String> = stderr
        .lock()
        .expect("stderr lock")
        .iter()
        .cloned()
        .collect();
    let mut message = reason.to_string();
    if !tail.is_empty() {
        message.push_str("\n\n");
        message.push_str(&tail.join("\n"));
    }
    EngineStatus::Failed { message }
}

/// Start the engine in the background; the UI follows `STATUS_EVENT` or asks `status()`.
pub fn start(app: &AppHandle) {
    let token = new_token();
    let generation = app
        .state::<Engine>()
        .generation
        .fetch_add(1, Ordering::SeqCst)
        + 1;
    set_status(app, generation, EngineStatus::Starting);
    let mut child = match engine_command(&token).spawn() {
        Ok(child) => child,
        Err(e) => {
            set_status(
                app,
                generation,
                EngineStatus::Failed {
                    message: format!(
                        "Could not start the ClipsDex engine ({e}). In development, `uv` must \
                         be installed and on the PATH."
                    ),
                },
            );
            return;
        }
    };
    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");
    let engine = app.state::<Engine>();
    *engine.stdin.lock().expect("stdin lock") = child.stdin.take();
    *engine.child.lock().expect("child lock") = Some(child);

    let stderr_tail = Arc::new(Mutex::new(VecDeque::with_capacity(STDERR_LINES)));
    {
        let tail = Arc::clone(&stderr_tail);
        thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                let mut tail = tail.lock().expect("stderr lock");
                if tail.len() == STDERR_LINES {
                    tail.pop_front();
                }
                tail.push_back(line);
            }
        });
    }

    let started = Instant::now();
    let ready = Arc::new(Mutex::new(false));
    {
        let app = app.clone();
        let ready = Arc::clone(&ready);
        let tail = Arc::clone(&stderr_tail);
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if let Some(port) = parse_ready(&line) {
                    *ready.lock().expect("ready lock") = true;
                    let url = format!("http://127.0.0.1:{port}");
                    set_status(
                        &app,
                        generation,
                        EngineStatus::Ready {
                            url,
                            token: token.clone(),
                        },
                    );
                }
            }
            // stdout closed: the engine has exited.
            if !*ready.lock().expect("ready lock") {
                thread::sleep(Duration::from_millis(200)); // let stderr catch up
                let status = failure("The ClipsDex engine stopped while starting.", &tail);
                set_status(&app, generation, status);
            } else {
                let status = failure("The ClipsDex engine stopped unexpectedly.", &tail);
                set_status(&app, generation, status);
            }
        });
    }
    {
        let app = app.clone();
        thread::spawn(move || {
            while started.elapsed() < START_TIMEOUT {
                if *ready.lock().expect("ready lock") {
                    return;
                }
                thread::sleep(Duration::from_millis(250));
            }
            if matches!(app.state::<Engine>().status(), EngineStatus::Starting) {
                let status = failure("The ClipsDex engine did not start in time.", &stderr_tail);
                set_status(&app, generation, status);
                stop(&app);
            }
        });
    }
}

/// Ask the engine to stop (close its stdin), then kill it if it takes too long.
pub fn stop(app: &AppHandle) {
    let engine = app.state::<Engine>();
    engine.generation.fetch_add(1, Ordering::SeqCst);
    drop(engine.stdin.lock().expect("stdin lock").take());
    let Some(mut child) = engine.child.lock().expect("child lock").take() else {
        return;
    };
    let deadline = Instant::now() + STOP_GRACE;
    while Instant::now() < deadline {
        if let Ok(Some(_)) = child.try_wait() {
            return;
        }
        thread::sleep(Duration::from_millis(100));
    }
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_only_the_ready_line() {
        assert_eq!(
            parse_ready(r#"{"event": "ready", "port": 53124}"#),
            Some(53124)
        );
        assert_eq!(
            parse_ready(r#"{"event":"ready","port":5,"token":"x"}"#),
            Some(5)
        );
        assert_eq!(parse_ready(r#"{"event": "ready", "port": 0}"#), None);
        assert_eq!(parse_ready(r#"{"event": "other", "port": 1}"#), None);
        assert_eq!(parse_ready("Installed 3 packages"), None);
    }

    #[test]
    fn tokens_are_long_and_random() {
        let (a, b) = (new_token(), new_token());
        assert_eq!(a.len(), 64);
        assert_ne!(a, b);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
