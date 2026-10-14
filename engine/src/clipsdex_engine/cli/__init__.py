"""ClipsDex command line interface."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from sqlmodel import Session

from clipsdex_engine import __version__, analyzers, binaries, config, db, library, llm, net
from clipsdex_engine import profile as quick_profile
from clipsdex_engine.db.models import JobStatus, SourceType
from clipsdex_engine.jobs import builtin, estimates, queue
from clipsdex_engine.jobs import plan as planning
from clipsdex_engine.jobs.handlers import WorkerEnv
from clipsdex_engine.jobs.worker import Worker, WorkerBusyError
from clipsdex_engine.sources import manual

app = typer.Typer(
    name="clipsdex",
    help="ClipsDex: local, private analysis of your saved short videos.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"clipsdex {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """ClipsDex: local, private analysis of your saved short videos."""


@app.command("import")
def import_(
    paths: Annotated[list[Path], typer.Argument(help="Video files and/or folders to import.")],
    recursive: Annotated[bool, typer.Option(help="Also look inside subfolders.")] = True,
) -> None:
    """Import video files from your computer. Files already in the library are skipped."""
    engine = db.open_database()
    report = manual.import_paths(engine, paths, recursive=recursive)

    for path, reason in report.failed:
        typer.secho(f"[FAILED]  {path}: {reason}", fg=typer.colors.RED)
    typer.echo(
        f"Imported {len(report.added)} new video(s), "
        f"skipped {len(report.duplicates)} already in the library."
    )
    if not report.added and not report.duplicates and not report.failed:
        typer.echo(f"No video files found. Supported: {', '.join(sorted(manual.VIDEO_EXTENSIONS))}")
    if report.failed:
        raise typer.Exit(code=1)


def _fmt_time(value: datetime | None) -> str:
    return value.astimezone().strftime("%Y-%m-%d %H:%M") if value else "-"


def _or_dash(value: object) -> str:
    return "-" if value is None else str(value)


def _download_printer() -> net.Progress:
    """Prints "Downloading a model…" once, then every 10%."""
    last_shown = -1

    def show(done: int, total: int | None) -> None:
        nonlocal last_shown
        if last_shown < 0:
            size = f" ({total / 1e6:,.0f} MB)" if total else ""
            typer.echo(f"Downloading a model, one time only{size}...")
            last_shown = 0
        if total and (percent := done * 100 // total) >= last_shown + 10:
            last_shown = percent
            typer.echo(f"  {percent}%")

    return show


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


@app.command("list")
def list_(
    source: Annotated[SourceType | None, typer.Option(help="Only videos from this source.")] = None,
    author: Annotated[str | None, typer.Option(help="Only videos by this author.")] = None,
    limit: Annotated[int | None, typer.Option(min=1, help="Show at most this many.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON instead of a table.")] = False,
) -> None:
    """List the videos in your library."""
    engine = db.open_database()
    with Session(engine) as session:
        videos = library.list_videos(
            session, library.VideoFilter(source_type=source, author=author, limit=limit)
        )
        if as_json:
            typer.echo(json.dumps([v.model_dump(mode="json") for v in videos], indent=2))
            return
        if not videos:
            typer.echo("No videos yet. Add some with: clipsdex import <folder>")
            return

        table = Table(box=None, header_style="bold")
        for column in ("ID", "Video", "Author", "Collection", "Imported"):
            table.add_column(column)
        for v in videos:
            table.add_row(
                str(v.id),
                library.display_name(v),
                v.author or "-",
                v.collection_name or "-",
                _fmt_time(v.imported_at),
            )
        Console().print(table)
        typer.echo(f"\n{len(videos)} video(s).")


@app.command()
def show(video_id: Annotated[int, typer.Argument(help="Video ID from `clipsdex list`.")]) -> None:
    """Show everything ClipsDex knows about one video."""
    engine = db.open_database()
    with Session(engine) as session:
        details = library.video_details(session, video_id)
        if details is None:
            typer.secho(f"No video with ID {video_id}. See: clipsdex list", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        v = details.video
        rows: list[tuple[str, str]] = [
            ("ID", str(v.id)),
            ("Video", library.display_name(v)),
            ("Source", details.source.display_name),
            ("File", v.file_path or ("deleted after analysis" if v.file_deleted else "-")),
            ("Link", v.original_url or "-"),
            ("Author", v.author or "-"),
            ("Collection", v.collection_name or "-"),
            ("Caption", v.caption or "-"),
            ("Saved", _fmt_time(v.saved_at)),
            ("Imported", _fmt_time(v.imported_at)),
        ]
        if details.category is not None:
            how = str(details.category_assigned_by)
            if details.category_confidence is not None:
                how += f", {details.category_confidence:.0%} sure"
            rows.append(("Category", f"{details.category.name} ({how})"))
        else:
            rows.append(("Category", "not categorized yet"))
        for label, value in rows:
            typer.echo(f"{label + ':':<12}{value}")

        p = details.profile
        typer.echo("\nProfile:")
        if p is None:
            typer.echo("  not profiled yet")
        else:
            size = f"{p.width}x{p.height}" if p.width and p.height else "-"
            audio = {True: "yes", False: "no", None: "-"}[p.has_audio]
            typer.echo(f"  duration {_or_dash(p.duration_s)} s, size {size}, audio {audio}")
            typer.echo(f"  cuts {_or_dash(p.cut_count)}")
            typer.echo(
                f"  speech {_pct(p.speech_ratio)}, music {_pct(p.music_ratio)}, "
                f"silence {_pct(p.silence_ratio)}"
            )
            if p.has_vocals is not None:
                typer.echo(f"  vocals in music: {'yes' if p.has_vocals else 'no'} (best guess)")
            if p.has_text is not None:
                typer.echo(f"  text on screen: {'yes' if p.has_text else 'no'}")
            for line in (p.ocr_text or "").splitlines():
                typer.echo(f"    | {line}")

        typer.echo("\nResults:")
        if not details.results:
            typer.echo("  none yet")
        for r in details.results:
            conf = f"{r.confidence:.0%}" if r.confidence is not None else "-"
            typer.echo(f"  {r.analyzer}: {json.dumps(r.value)} (confidence {conf})")
            typer.echo(f"    found by {r.method}" + (f": {r.evidence}" if r.evidence else ""))

        for a in details.artifacts:
            if a.type == "transcript" and isinstance(a.data, dict):
                language = a.data.get("language") or "no speech"
                text = a.data.get("text") or ""
                typer.echo(f"\nTranscript ({a.params.get('model')}, language: {language}):")
                typer.echo(f"  {text[:300] + ('…' if len(text) > 300 else '') or '(empty)'}")
            if a.type == "frames" and isinstance(a.data, dict):
                shots = a.data.get("frames", [])
                times = ", ".join(f"{f['t']:.1f}s" for f in shots)
                typer.echo(
                    f"\nKey frames: {len(shots)} of {a.data.get('candidates')} candidates, "
                    f"{a.data.get('scenes')} scene(s), budget {a.data.get('budget')}: {times}"
                )
                if a.path:
                    typer.echo(f"  {config.data_dir() / a.path}")
            if a.type == "frame_descriptions" and isinstance(a.data, dict):
                typer.echo(f"\nFrame descriptions ({a.data.get('model')}):")
                for f in a.data.get("frames", []):
                    if "error" in f:
                        typer.echo(f"  {f['t']:.1f}s  could not describe: {f['error']}")
                    else:
                        typer.echo(f"  {f['t']:.1f}s  [{f['shot_type']}] {f['description']}")
                for f in a.data.get("skipped", []):
                    typer.echo(f"  {f['t']:.1f}s  skipped: {f['reason']}")
            if a.type == "ocr" and isinstance(a.data, dict):
                lines = a.data.get("lines", [])
                typer.echo(f"\nOn-screen text ({len(lines)} unique line(s)):")
                for line in lines[:15]:
                    typer.echo(f"  | {line['text']}  (from {line['first_t']:.1f}s)")
                if len(lines) > 15:
                    typer.echo(f"  … and {len(lines) - 15} more")

        if details.artifacts:
            names = ", ".join(sorted({a.type for a in details.artifacts}))
            typer.echo(f"\nCached artifacts: {names}")


@app.command()
def profile(
    video: Annotated[
        list[int] | None, typer.Option("--video", help="Only this video ID (repeatable).")
    ] = None,
    redo: Annotated[bool, typer.Option(help="Rebuild profiles that already exist.")] = False,
    guess: Annotated[
        bool, typer.Option(help="Also guess a category with the local text model.")
    ] = True,
    in_background: Annotated[
        bool,
        typer.Option("--queue", help="Only add jobs to the queue; `clipsdex worker` runs them."),
    ] = False,
) -> None:
    """Build the quick profile of new videos (duration, audio, cuts, text) and guess categories."""
    paths = config.data_paths()
    engine = db.open_database(paths)
    if in_background:
        env = WorkerEnv(engine, paths)
        ids = builtin.enqueue_profiling(env, video_ids=video, redo=redo, guess=guess)
        typer.echo(f"Queued {len(ids)} job(s). Run them with: clipsdex worker")
        return
    models = quick_profile.default_models(paths, progress=_download_printer())
    try:
        report = quick_profile.run_quick_profile(
            engine, paths, video_ids=video, redo=redo, models=models, guess=guess
        )
    except (binaries.BinaryNotFoundError, net.DownloadError) as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(code=1) from e

    failed = False
    for video_id, reason in report.failed:
        typer.secho(f"[FAILED]  video {video_id}: {reason}", fg=typer.colors.RED)
        failed = True
    if report.profiled or report.failed:
        typer.echo(f"Profiled {len(report.profiled)} video(s), {len(report.failed)} failed.")

    g = report.guesses
    if g is not None:
        for video_id, reason in g.failed:
            typer.secho(f"[FAILED]  category for video {video_id}: {reason}", fg=typer.colors.RED)
            failed = True
        if g.skipped_reason:
            typer.secho(f"Category guesses skipped: {g.skipped_reason}", fg=typer.colors.YELLOW)
        elif g.guessed:
            typer.echo(f"Guessed categories for {len(g.guessed)} video(s).")

    nothing_done = not (report.profiled or report.failed or (g and (g.guessed or g.failed)))
    if nothing_done and not (g and g.skipped_reason):
        typer.echo("Nothing to do. Use --redo to rebuild existing profiles.")
    if failed:
        raise typer.Exit(code=1)


@app.command()
def transcribe(
    video: Annotated[
        list[int] | None, typer.Option("--video", help="Only this video ID (repeatable).")
    ] = None,
    redo: Annotated[
        bool, typer.Option(help="Transcribe again even if a transcript exists.")
    ] = False,
) -> None:
    """Queue speech-to-text for videos without a transcript (`clipsdex worker` runs it)."""
    paths = config.data_paths()
    env = WorkerEnv(db.open_database(paths), paths)
    ids = builtin.enqueue_transcripts(env, video_ids=video, redo=redo)
    typer.echo(f"Queued {len(ids)} transcript job(s). Run them with: clipsdex worker")


@app.command("frames")
def frames_cmd(
    video: Annotated[
        list[int] | None, typer.Option("--video", help="Only this video ID (repeatable).")
    ] = None,
    redo: Annotated[bool, typer.Option(help="Select frames again if they exist.")] = False,
) -> None:
    """Queue key-frame selection for videos (`clipsdex worker` runs it)."""
    paths = config.data_paths()
    env = WorkerEnv(db.open_database(paths), paths)
    ids = builtin.enqueue_frames(env, video_ids=video, redo=redo)
    typer.echo(f"Queued {len(ids)} frame job(s). Run them with: clipsdex worker")


@app.command("ocr")
def ocr_cmd(
    video: Annotated[
        list[int] | None, typer.Option("--video", help="Only this video ID (repeatable).")
    ] = None,
    redo: Annotated[bool, typer.Option(help="Read the text again if it exists.")] = False,
) -> None:
    """Queue reading on-screen text from the key frames (`clipsdex worker` runs it)."""
    paths = config.data_paths()
    env = WorkerEnv(db.open_database(paths), paths)
    ids = builtin.enqueue_ocr(env, video_ids=video, redo=redo)
    typer.echo(f"Queued {len(ids)} job(s). Run them with: clipsdex worker")


@app.command()
def describe(
    video: Annotated[
        list[int] | None, typer.Option("--video", help="Only this video ID (repeatable).")
    ] = None,
    redo: Annotated[bool, typer.Option(help="Describe the frames again.")] = False,
) -> None:
    """Queue short descriptions of the key frames by the local vision model."""
    paths = config.data_paths()
    env = WorkerEnv(db.open_database(paths), paths)
    ids = builtin.enqueue_frame_descriptions(env, video_ids=video, redo=redo)
    typer.echo(f"Queued {len(ids)} job(s). Run them with: clipsdex worker")


@app.command()
def plan(
    analyzer: Annotated[
        list[str] | None,
        typer.Option(help="What you want to run, e.g. brief, categorize, movie_id (repeatable)."),
    ] = None,
    artifact: Annotated[
        list[str] | None,
        typer.Option(help="Or name artifacts directly: transcript, frames, ocr, …"),
    ] = None,
    category: Annotated[str | None, typer.Option(help="Only videos in this category.")] = None,
    video: Annotated[
        list[int] | None,
        typer.Option(
            "--video",
            help="Only this video ID (repeatable); runs even where an analyzer would skip it.",
        ),
    ] = None,
    queue_it: Annotated[bool, typer.Option("--queue", help="Also queue the missing work.")] = False,
) -> None:
    """Show what a request would compute and how long it would take on this computer."""
    paths = config.data_paths()
    engine = db.open_database(paths)
    if not (analyzer or artifact):
        typer.secho(
            "Say what to plan: --analyzer brief, or --artifact transcript.", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)
    try:
        planning.check_names(analyzer or [], artifact or [])
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(code=1) from e

    p = planning.build_plan(engine, analyzer or [], artifact or [], category, video)
    if video is not None:
        scope = "the selected videos"
    elif category:
        scope = f'category "{category}"'
    else:
        usual = sorted({c for a in analyzer or [] for c in analyzers.get(a).default_categories})
        every = artifact or any(not analyzers.get(a).default_categories for a in analyzer or [])
        scope = "all videos" if every or not usual else f"the usual categories ({', '.join(usual)})"
    what = ", ".join([*(analyzer or []), *(artifact or [])])
    typer.echo(f"Plan: {what} for {scope}: {len(p.videos)} video(s)")
    if not p.videos:
        typer.echo("No videos match.")
        return
    for v in p.videos:
        if v.blocked:
            status = f"can't run: {v.blocked}"
        elif v.missing:
            status = "needs " + ", ".join(v.missing)
        else:
            status = "ready" if v.analyzers or not v.skipped else "nothing to do"
        typer.echo(f"  #{v.video_id:<4} {v.name[:40]:<40} {status}")
        for name, reason in v.skipped.items():
            typer.echo(f"        skips {name}: {reason}")
    for name in p.not_available:
        typer.secho(
            f"\n{analyzers.get(name).title} ({name}) is not available in this version yet; "
            "only the inputs it reads are planned.",
            fg=typer.colors.YELLOW,
        )

    if not p.jobs:
        typer.echo("\nEverything needed is already computed.")
        return
    typer.echo("")
    for j in p.jobs:
        basis = (
            f"based on {j.measured_jobs} job(s) on this computer"
            if j.measured_jobs
            else "default estimate"
        )
        typer.echo(
            f"  {j.name:<19} {j.videos:>3} video(s)  {estimates.humanize(j.seconds):>9}  ({basis})"
        )
    if p.load_seconds:
        typer.echo(f"  {'loading models':<19} {'':>12}  {estimates.humanize(p.load_seconds):>9}")
    typer.echo(f"\nTotal: {estimates.humanize(p.total_seconds)}")

    if queue_it:
        ids = planning.queue_plan(WorkerEnv(engine, paths), p)
        typer.echo(f"Queued {len(ids)} job(s). Run them with: clipsdex worker")


@app.command()
def worker(
    watch: Annotated[
        bool, typer.Option(help="Keep running and wait for new jobs (stop with Ctrl+C).")
    ] = False,
) -> None:
    """Run queued jobs, grouped so only one heavy model is loaded at a time."""
    paths = config.data_paths()
    env = WorkerEnv(db.open_database(paths), paths, download_progress=_download_printer())
    w = Worker(env, builtin.HANDLERS, on_event=lambda e: typer.echo(f"  {e}"))
    try:
        report = w.run(until_idle=not watch)
    except WorkerBusyError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(code=1) from e
    except KeyboardInterrupt:
        typer.echo("\nStopped. Unfinished jobs stay queued and continue next time.")
        raise typer.Exit(code=130) from None

    for job_id, reason in report.failed:
        typer.secho(f"[FAILED]  job {job_id}: {reason}", fg=typer.colors.RED)
    typer.echo(
        f"Done {len(report.done)}, failed {len(report.failed)}, "
        f"cancelled {len(report.cancelled)}, will retry {len(report.retried)}."
    )
    if report.failed:
        raise typer.Exit(code=1)


jobs_app = typer.Typer(help="See and cancel background jobs.", no_args_is_help=False)
app.add_typer(jobs_app, name="jobs")


@jobs_app.callback(invoke_without_command=True)
def jobs_overview(
    ctx: typer.Context,
    status: Annotated[JobStatus | None, typer.Option(help="Only jobs with this status.")] = None,
    limit: Annotated[int, typer.Option(min=1, help="How many recent jobs to show.")] = 20,
) -> None:
    """Show how many jobs are queued, running, done, failed or cancelled."""
    if ctx.invoked_subcommand is not None:
        return
    engine = db.open_database()
    counts = queue.counts(engine)
    typer.echo("  ".join(f"{s.value}: {n}" for s, n in counts.items()))
    jobs = queue.recent(engine, limit=limit, status=status)
    if not jobs:
        typer.echo("No jobs.")
        return
    table = Table(box=None, header_style="bold")
    for column in ("ID", "Job", "Video", "Status", "Progress", "Tries", "Error"):
        table.add_column(column)
    for j in jobs:
        table.add_row(
            str(j.id),
            f"{j.kind.value}:{j.name}",
            _or_dash(j.video_id),
            j.status.value,
            f"{j.progress:.0%}",
            str(j.attempts),
            (j.error or "")[:60],
        )
    Console().print(table)


@jobs_app.command("cancel")
def jobs_cancel(
    job_ids: Annotated[list[int] | None, typer.Argument(help="Job IDs to cancel.")] = None,
    all_jobs: Annotated[
        bool, typer.Option("--all", help="Cancel every queued or running job.")
    ] = False,
) -> None:
    """Cancel jobs. Running jobs stop at their next checkpoint."""
    if not job_ids and not all_jobs:
        typer.secho("Give job IDs, or --all to cancel everything.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    engine = db.open_database()
    n = queue.cancel(engine, None if all_jobs else job_ids)
    typer.echo(f"Cancelled {n} job(s).")


@app.command()
def doctor() -> None:
    """Check that the tools and local AI models ClipsDex needs are installed."""
    typer.echo(f"Data folder: {config.data_dir()}\n")
    missing = 0
    for name in binaries.BINARIES:
        resolved = binaries.resolve(name)
        if resolved.path is None:
            missing += 1
            typer.secho(f"[MISSING] {name}: {resolved.problem}", fg=typer.colors.YELLOW)
            typer.echo(f"          {binaries.INSTALL_HINTS[name]}")
            continue
        typer.secho(f"[OK]      {name} ({resolved.source}): {resolved.path}", fg=typer.colors.GREEN)
        ver = binaries.version(resolved.path)
        typer.echo(f"          {ver or 'found, but it did not report a version'}")

    missing += _doctor_models()

    if missing:
        typer.secho(f"\n{missing} item(s) missing.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    typer.secho("\nEverything needed is installed.", fg=typer.colors.GREEN)


def _doctor_models() -> int:
    """Check the local model runtime and the models the current preset needs."""
    paths = config.data_paths()
    settings = config.Settings()
    if paths.db.is_file():  # don't create a library just to read settings
        with Session(db.open_database(paths)) as session:
            settings = db.load_settings(session)

    typer.echo("")
    provider = llm.default_provider()
    health = provider.health()
    if not health.ok:
        typer.secho(f"[MISSING] Ollama: {health.message}", fg=typer.colors.YELLOW)
        typer.echo("          Install it from https://ollama.com, or start the Ollama app.")
        return 1
    typer.secho(f"[OK]      {health.message}", fg=typer.colors.GREEN)

    missing = 0
    models = settings.models()
    for model in dict.fromkeys([models.text, models.vision, models.embedding]):
        if provider.has_model(model):
            typer.secho(f"[OK]      model {model}", fg=typer.colors.GREEN)
        else:
            missing += 1
            typer.secho(
                f"[MISSING] model {model} ({settings.preset} preset)", fg=typer.colors.YELLOW
            )
            typer.echo(f"          Download it with: ollama pull {model}")
    return missing
