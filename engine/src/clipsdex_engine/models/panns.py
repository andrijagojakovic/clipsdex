"""PANNs Cnn14 (16 kHz) audio tagger: 527 AudioSet classes per audio segment.

Model architecture adapted from https://github.com/qiuqiangkong/audioset_tagging_cnn
(MIT License, Copyright (c) 2020 Qiuqiang Kong). We vendor the small model definition
instead of depending on ``panns-inference``, which downloads files with ``wget`` into the
user's home folder at import time (no ``wget`` on Windows; data must stay in our data dir).

Weights: ``Cnn14_16k_mAP=0.438.pth`` from https://zenodo.org/records/3987831 (CC BY 4.0),
downloaded on first use into ``<data dir>/models/panns/``. We never ship the weights.
"""

from __future__ import annotations

import codecs
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy._core.multiarray import _reconstruct as _np_reconstruct  # pyright: ignore[reportPrivateUsage]
from torch import nn

from clipsdex_engine import net
from clipsdex_engine.models import audioset

SAMPLE_RATE = audioset.SAMPLE_RATE
WINDOW = 512
HOP = 160
MEL_BINS = 64
NUM_CLASSES = audioset.NUM_CLASSES

CHECKPOINT_NAME = "Cnn14_16k_mAP=0.438.pth"
CHECKPOINT_URL = "https://zenodo.org/records/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1"
CHECKPOINT_MD5 = "362fc5ff18f1d6ad2f6d464b45893f2c"
CHECKPOINT_BYTES = 358_668_570


# Globals the (legacy-format) checkpoint references besides tensors; found by scanning its
# pickle opcodes. These only rebuild plain numpy arrays.
_NUMPY_ARRAY_GLOBALS: list[Any] = [
    (_np_reconstruct, "numpy.core.multiarray._reconstruct"),
    np.ndarray,
    np.dtype,
    codecs.encode,
    # Numeric dtype classes only (never the object dtype, which could carry arbitrary objects).
    *(type(np.dtype(code)) for code in "?bBhHiIlLqQefdg"),
]


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, pool_size: tuple[int, int]) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool_size)


class Cnn14(nn.Module):
    """Inference-only Cnn14. The log-mel front end matches torchlibrosa's (used in training)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("window", torch.hann_window(WINDOW, periodic=True), persistent=False)
        self.register_buffer("mel_w", torch.zeros(WINDOW // 2 + 1, MEL_BINS))  # from checkpoint
        self.bn0 = nn.BatchNorm2d(MEL_BINS)
        self.conv_block1 = _ConvBlock(1, 64)
        self.conv_block2 = _ConvBlock(64, 128)
        self.conv_block3 = _ConvBlock(128, 256)
        self.conv_block4 = _ConvBlock(256, 512)
        self.conv_block5 = _ConvBlock(512, 1024)
        self.conv_block6 = _ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, NUM_CLASSES)

    def logmel(self, waveform: torch.Tensor) -> torch.Tensor:
        """(batch, samples) -> (batch, 1, frames, mel_bins)."""
        spec = torch.stft(
            waveform,
            n_fft=WINDOW,
            hop_length=HOP,
            win_length=WINDOW,
            window=self.window,  # pyright: ignore[reportArgumentType]
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = spec.abs().pow(2).transpose(1, 2)  # (batch, frames, freq)
        mel = power @ self.mel_w  # pyright: ignore[reportOperatorIssue]
        return (10.0 * torch.log10(mel.clamp(min=1e-10))).unsqueeze(1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.logmel(waveform)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for block in (
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ):
            x = block(x, pool_size=(2, 2))
        x = self.conv_block6(x, pool_size=(1, 1))
        x = torch.mean(x, dim=3)
        x = torch.max(x, dim=2).values + torch.mean(x, dim=2)
        x = F.relu_(self.fc1(x))
        return torch.sigmoid(self.fc_audioset(x))


def checkpoint_path(models_dir: Path) -> Path:
    return models_dir / "panns" / CHECKPOINT_NAME


def ensure_checkpoint(models_dir: Path, progress: net.Progress | None = None) -> Path:
    path = checkpoint_path(models_dir)
    if path.is_file() and path.stat().st_size == CHECKPOINT_BYTES:
        return path
    return net.download(CHECKPOINT_URL, path, md5=CHECKPOINT_MD5, progress=progress)


def _convert_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map the original checkpoint keys onto our module (front end and augmentation differ)."""
    converted: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key == "logmel_extractor.melW":
            converted["mel_w"] = value
        elif key.startswith(("spectrogram_extractor.", "logmel_extractor.", "spec_augmenter.")):
            continue  # STFT is done with torch.stft; augmentation is training-only
        else:
            converted[key] = value
    return converted


class PannsTagger:
    """Loads Cnn14 once; ``predict`` takes 1-D float32 segments at 16 kHz."""

    def __init__(self, checkpoint: Path, batch_size: int = 16) -> None:
        # Stay in safe (weights_only) mode; only allow what the checkpoint's plain numpy
        # arrays need. Never load downloaded checkpoints with weights_only=False.
        with torch.serialization.safe_globals(_NUMPY_ARRAY_GLOBALS):
            raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = raw.get("model", raw)
        self.model = Cnn14()
        self.model.load_state_dict(_convert_state_dict(state), strict=True)
        self.model.eval()
        self.batch_size = batch_size

    def predict(self, segments: np.ndarray) -> np.ndarray:
        """(n, samples) float32 -> (n, 527) class probabilities."""
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(segments), self.batch_size):
                batch = torch.from_numpy(segments[start : start + self.batch_size])
                outputs.append(self.model(batch).numpy())
        if not outputs:
            return np.zeros((0, NUM_CLASSES), dtype=np.float32)
        return np.concatenate(outputs)
