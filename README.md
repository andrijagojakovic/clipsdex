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

## License

[MIT](LICENSE)
