"""Prepare Speech Commands V2-35 splits and cached 40-by-98 MFCCs."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import wave

import numpy as np
import torch
import torchaudio


SAMPLE_RATE = 16_000
COEFFICIENTS = 40
FRAMES = 98
SPLIT_COUNTS = (84_843, 9_981, 11_005)
CACHE_VERSION = 1


def official_split(root):
    labels = sorted(
        p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))
    )
    if len(labels) != 35:
        raise ValueError("Expected the 35 command directories of Speech Commands V2")
    files = {p.relative_to(root).as_posix() for label in labels for p in (root / label).glob("*.wav")}
    lists = [root / f"{split}_list.txt" for split in ("validation", "testing")]
    validation, test = [set(path.read_text().splitlines()) for path in lists]
    if validation & test or not (validation | test) <= files:
        raise ValueError("Invalid or incomplete Speech Commands validation and test lists")
    splits = (sorted(files - validation - test), sorted(validation), sorted(test))
    if tuple(map(len, splits)) != SPLIT_COUNTS:
        raise ValueError(f"Unexpected Speech Commands V2 split counts {tuple(map(len, splits))}")
    return labels, splits, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in lists}


def read_wave(path):
    with wave.open(str(path), "rb") as clip:
        if (clip.getframerate(), clip.getnchannels(), clip.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"Expected a 16 kHz mono PCM16 recording at {path}")
        values = np.frombuffer(clip.readframes(SAMPLE_RATE), dtype="<i2")
    waveform = np.zeros(SAMPLE_RATE, dtype=np.float32)
    waveform[: len(values)] = values * (1.0 / 32768.0)
    return waveform


class Features:
    def __init__(self):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=480,
            hop_length=160,
            n_mels=COEFFICIENTS,
            center=False,
            norm="slaney",
            mel_scale="slaney",
        )
        self.dct = torchaudio.functional.create_dct(COEFFICIENTS, COEFFICIENTS, norm="ortho")

    def __call__(self, waveform):
        mel = self.mel(waveform)
        # librosa.power_to_db uses a per-clip 80 dB floor and reference power 1.
        db = 10 * mel.clamp_min(1e-10).log10()
        db = torch.maximum(db, db.amax(dim=(-2, -1), keepdim=True) - 80)
        return (self.dct.T @ db).contiguous()


def prepare(root, cache, workers):
    labels, splits, split_hashes = official_split(root)
    label_id = {label: idx for idx, label in enumerate(labels)}
    paths = [path for split in splits for path in split]
    transform = Features()
    if cache.exists():
        raise FileExistsError(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=cache.name + ".", dir=cache.parent) as temporary:
        building = Path(temporary)
        features = np.lib.format.open_memmap(
            building / "features.npy", mode="w+", dtype=np.float32,
            shape=(len(paths), COEFFICIENTS, FRAMES),
        )
        with ThreadPoolExecutor(max_workers=workers) as reader, torch.inference_mode():
            for start in range(0, len(paths), 256):
                group = paths[start : start + 256]
                waveform = np.stack(list(reader.map(read_wave, (root / path for path in group))))
                values = transform(torch.from_numpy(waveform)).numpy()
                features[start : start + len(group)] = values
                if start % 8192 == 0:
                    print(f"Prepared {start + len(group):,} / {len(paths):,} clips", flush=True)
        features.flush()
        np.save(
            building / "labels.npy",
            np.array([label_id[path.split("/", 1)[0]] for path in paths], dtype=np.int64),
        )
        manifest = {
            "version": CACHE_VERSION,
            "dataset": "Speech Commands V2-35",
            "classes": labels,
            "split_counts": list(SPLIT_COUNTS),
            "split_sha256": split_hashes,
            "torch": str(torch.__version__),
            "torchaudio": torchaudio.__version__,
        }
        (building / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        building.rename(cache)
    print(f"Prepared {len(paths):,} clips in {cache}", flush=True)


class CachedSpeech:
    def __init__(self, cache):
        self.manifest = json.loads((cache / "manifest.json").read_text())
        self.features = np.load(cache / "features.npy", mmap_mode="r", allow_pickle=False)
        self.labels = np.load(cache / "labels.npy", allow_pickle=False)
        count = sum(SPLIT_COUNTS)
        if (
            self.manifest["version"] != CACHE_VERSION
            or self.manifest["split_counts"] != list(SPLIT_COUNTS)
            or len(self.manifest["classes"]) != 35
            or self.features.shape != (count, COEFFICIENTS, FRAMES)
            or self.features.dtype != np.float32
            or self.labels.shape != (count,)
            or self.labels.dtype != np.int64
        ):
            raise ValueError(f"Incompatible Speech Commands cache at {cache}")
        offsets = np.cumsum((0, *SPLIT_COUNTS))
        self.train, self.validation, self.test = [
            np.arange(start, stop) for start, stop in zip(offsets[:-1], offsets[1:])
        ]

    def batch(self, indices, device):
        x = torch.from_numpy(self.features[indices]).unsqueeze(1).to(device)
        labels = torch.from_numpy(self.labels[indices]).to(device)
        return x, labels


def spec_augment(features, generator):
    # Match the official widths and use fresh masks on each batch, without editing the cache.
    b, _, freq, frames = features.shape
    for size, limit, shape in ((frames, 25, (b, 1, 1, frames)), (freq, 7, (b, 1, freq, 1))):
        random = torch.rand(2, b, 2, generator=generator, device=features.device)
        widths = (random[0] * limit).long()
        starts = (random[1] * (size - widths)).long()
        positions = torch.arange(size, device=features.device)
        masked = ((positions >= starts[..., None]) & (positions < (starts + widths)[..., None])).any(dim=-2)
        features = features.masked_fill(masked.reshape(shape), 0)
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Extracted speech_commands_v0.02 directory")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    torch.set_num_threads(min(args.workers, 4))
    prepare(args.data, args.cache, args.workers)


if __name__ == "__main__":
    main()
