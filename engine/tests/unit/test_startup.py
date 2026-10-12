import subprocess
import sys


def test_cli_import_does_not_load_heavy_ml_libraries() -> None:
    """torch costs ~6 s to import on Windows; only load it when a model is actually used."""
    code = (
        "import sys, clipsdex_engine.cli; "
        "heavy = ('torch', 'onnxruntime', 'rapidocr', 'faster_whisper', 'ctranslate2'); "
        "print([m for m in heavy if m in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
