# ClipsDex

**Local, private, open source organizer for your saved short videos.**

ClipsDex analyzes the reels you saved on Instagram (TikTok later) and helps you make sense of them:

- sorts them into categories you can change,
- finds the song used in an edit, even when it is not credited,
- turns business and advice videos into a single brief,
- identifies the movies your saved clips come from.

Everything runs on your own computer with local AI models. Nothing leaves your machine unless you choose a cloud model with your own API key.

> **Status:** early development. Nothing is usable yet. Follow progress in [CLAUDE.md](CLAUDE.md).

## Development

Windows 11 is the primary platform. To check and install the developer tools:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup-dev.ps1          # check
powershell -ExecutionPolicy Bypass -File scripts\setup-dev.ps1 -Install # install missing tools
```

The Python engine lives in [`engine/`](engine/) and is managed with [uv](https://docs.astral.sh/uv/).

### External tools (ffmpeg, ffprobe, fpcalc)

For development, install them with winget (the setup script above does this too):

```powershell
winget install Gyan.FFmpeg          # ffmpeg + ffprobe
winget install AcoustID.Chromaprint # fpcalc
```

Then check that the engine finds them (from `engine/`):

```powershell
uv run clipsdex doctor
```

The engine looks for each tool in this order:

1. An override env var: `CLIPSDEX_FFMPEG`, `CLIPSDEX_FFPROBE`, `CLIPSDEX_FPCALC` (full path to the `.exe`).
2. The bundled tools folder in `CLIPSDEX_BIN_DIR` (set by the desktop app in releases).
3. Your `PATH`.

## License

[MIT](LICENSE)
