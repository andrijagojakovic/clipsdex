# ClipsDex roadmap

What is done and what comes next, up to the first Windows release (v0.1) and beyond.
Step numbers (e.g. 3.2) are referenced in commit messages.

`[x]` done · `[ ]` planned

## Phase 0 – Repository and tooling

- [x] **0.2** Repository layout, MIT license, `.gitignore`, README
- [x] **0.3** Python engine project (`uv`, ruff, pyright, pytest)
- [x] **0.4** CI on a Windows runner: lint, type check, tests
- [x] **0.5** Locating ffmpeg / ffprobe / fpcalc (override → bundled → `PATH`) and `clipsdex doctor`

## Phase 1 – Engine foundation

- [x] **1.1** Data folder (`%LOCALAPPDATA%\ClipsDex`) and Fast / Balanced / Detailed model presets
- [x] **1.2** SQLite database with migrations (SQLModel + Alembic)
- [x] **1.3** Import video files or folders; duplicates detected by content hash
- [ ] **1.4** Import saved posts and collections from the official Instagram data export
- [x] **1.5** CLI: `import`, `list`, `show`, `doctor`

## Phase 2 – Quick profile (automatic, cheap)

- [x] **2.1** Video metadata (ffprobe) and audio extraction
- [x] **2.2** Speech / music / silence ratios and vocals detection (PANNs, CPU)
- [x] **2.3** Scene cut count (PySceneDetect)
- [x] **2.4** On-screen text check (RapidOCR)
- [x] **2.5** LLM provider interface with a local Ollama implementation
- [x] **2.6** First category guess by a small local model, with confidence and reason
- [ ] **2.7** Evaluation set and `clipsdex eval` (accuracy and timing per stage)

## Phase 3 – Shared artifacts and background jobs

- [x] **3.1** Job queue in SQLite: retries, cancel, resume after a crash
- [x] **3.2** Background worker that groups work by model, so only one heavy model is loaded at a time
- [x] **3.3** Transcripts with timestamps and language detection (faster-whisper, CPU)
- [x] **3.4** Key frame selection: scene changes, near-duplicate removal, per-preset frame budget
- [x] **3.5** Full on-screen text extraction over the selected frames
- [x] **3.6** Short factual frame descriptions with a local vision model
- [x] **3.7** Time estimates before a run (`clipsdex plan`)

## Phase 4 – Analyzers (on demand)

- [x] **4.1** Analyzer framework: registry and "does this apply to this video" rules
- [x] **4.2** Full categorization that learns from the user's corrections
- [x] **4.3** Briefs: notes per video, merged into one Markdown document
- [ ] **4.4** Movie identification from text, dialogue and scene descriptions (no face recognition), verified against TMDb
- [ ] **4.5** Song identification: caption / on-screen text, then audio fingerprint (AcoustID)
- [ ] **4.6** Song identification for sped-up and slowed edits (tempo / pitch variants)
- [ ] **4.7** Song identification from sung lyrics (vocal separation → transcript → LRCLIB)
- [x] **4.8** `clipsdex run` for an analyzer on a category, a selection or one video

## Phase 5 – Search, corrections, export

- [ ] **5.1** Semantic search over transcripts, summaries and tags
- [ ] **5.2** Category management: create, rename, merge, move videos
- [ ] **5.3** Export to Markdown (Obsidian-friendly), JSON and CSV

## Phase 6 – Local API

- [ ] **6.1** HTTP API for the desktop app (localhost only, random port, token)
- [ ] **6.2** Live job progress (Server-Sent Events)
- [ ] **6.3** API tests

## Phase 7 – Desktop app (Tauri + React)

- [ ] **7.1** App shell that starts and stops the engine
- [ ] **7.2** Library view with filters, categories and "how it was found" for every result
- [ ] **7.3** Video playback, or a link to the original when the file was removed
- [ ] **7.4** Actions per category / selection / video, with time estimate and progress
- [ ] **7.5** Category editing
- [ ] **7.6** Settings: presets, frame budget, storage, optional API keys
- [ ] **7.7** Search and export screens

## Phase 8 – Installer and first run

- [ ] **8.1** Python environment set up automatically on first run
- [ ] **8.2** Use an existing Ollama, or a bundled model runtime
- [ ] **8.3** Hardware check and recommended preset
- [ ] **8.4** First-run wizard: one download button, disk space check, smoke test
- [ ] **8.5** Optional heavy models downloaded only when first needed

## Phase 9 – Optional downloader (off by default)

- [ ] **9.1** Fetch videos for exported links, isolated from the core
- [ ] **9.2** Explicit consent screen and setting
- [ ] **9.3** Rate limiting, session limits, resume, failure handling
- [ ] **9.4** Delete videos after analysis and show storage use

## Phase 10 – Release v0.1 (Windows)

- [ ] **10.1** Bundled tools and third-party licenses
- [ ] **10.2** Release workflow producing a Windows installer
- [ ] **10.3** User documentation
- [ ] **10.4** Test on a clean machine
- [ ] **10.5** Tag v0.1.0

## Later

- [ ] Optional cloud models with your own API key (data leaves your computer only if you choose this)
- [ ] Optional paid song identification services with your own key
- [ ] MCP server: ask Claude about your own library
- [ ] TikTok export
- [ ] macOS and Linux builds
- [ ] Code signing
- [ ] Optional GPU acceleration
