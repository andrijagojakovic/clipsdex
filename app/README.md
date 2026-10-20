# ClipsDex desktop app

Tauri v2 shell with a React + TypeScript UI. The app starts the Python engine
(`clipsdex serve`) in the background, talks to it over `127.0.0.1` with a random
token per start, and stops it when the app closes. If the app crashes, the engine
notices that its stdin closed and exits too.

## Development

Needs Node.js, pnpm, Rust (MSVC toolchain) and the engine set up with `uv` (see the
main README). From `app/`:

```powershell
pnpm install
pnpm tauri dev        # opens the app; the engine runs from ../engine with uv
pnpm typecheck        # TypeScript
pnpm test             # UI unit tests (vitest)
```

Rust checks, from `app/src-tauri/`:

```powershell
cargo clippy --all-targets -- -D warnings
cargo test
```

The engine uses your normal data folder (`%LOCALAPPDATA%\ClipsDex`). To try the app
with another library, set `CLIPSDEX_DATA_DIR` before `pnpm tauri dev`.
