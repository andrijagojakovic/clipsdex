mod engine;

use engine::{Engine, EngineStatus};
use tauri::{AppHandle, RunEvent, State};

/// Where the engine is and the token to use, or why it is not running.
#[tauri::command]
fn engine_status(engine: State<'_, Engine>) -> EngineStatus {
    engine.status()
}

/// Stop the engine (if running) and start it again, e.g. after it failed.
#[tauri::command]
fn restart_engine(app: AppHandle) {
    std::thread::spawn(move || {
        engine::stop(&app);
        engine::start(&app);
    });
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(Engine::default())
        .setup(|app| {
            engine::start(app.handle());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![engine_status, restart_engine])
        .build(tauri::generate_context!())
        .expect("error while building the ClipsDex app")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                engine::stop(app);
            }
        });
}
