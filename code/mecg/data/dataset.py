"""
ECG dataset and datamodule.

Substantive design points
-------------------------
1. NORMALISATION. Signals are stored raw (mV) and normalised at load time.
   `norm_mode="dataset"` standardises with statistics computed once on the
   training split, preserving relative amplitude; `norm_mode="per_record"`
   reproduces the original submission's per-record, per-lead z-scoring, which
   divides each lead by its own SD and therefore destroys absolute voltage --
   the diagnostic criterion for ventricular hypertrophy.

2. EVAL LOADERS. `get_eval_loaders()` returns loaders that are unshuffled,
   unaugmented and keep every sample. Any analysis that pairs embeddings with
   an external per-record array (probing against physiological descriptors,
   patient-level bootstrap) MUST use these; the training loader shuffles and
   drops the last partial batch, so row i of its embedding matrix is not
   record i of X_train.

3. WEARABLE PROXIES. `lead_subset` restricts input leads (e.g. lead I alone,
   smartwatch-like) and `corruption` injects artefacts at a controlled SNR,
   deterministically in the record index so evaluation is reproducible.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

LEAD_SUBSETS: Dict[str, List[int]] = {
    "12lead": list(range(12)),
    "8lead": [0, 1, 6, 7, 8, 9, 10, 11],   # independent leads (I, II, V1-V6)
    "6lead": [0, 1, 2, 3, 4, 5],           # limb leads
    "3lead": [0, 1, 5],                    # I, II, aVF
    "2lead": [0, 1],                       # I, II
    "lead_I": [0],                         # single lead, smartwatch-like
    "lead_II": [1],                        # single lead, chest-patch-like
}

_NORM_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}


# ---------------------------------------------------------------------------
def compute_norm_stats(x_train_path: str, max_records: int = 8000,
                       seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-lead mean/SD over a random sample of the training split.

    Cached by path: the robustness sweep instantiates ~25 datamodules and the
    lead ablation another 7, so recomputing would dominate their runtime.
    Sampling is random rather than the first N records, because PTB-XL is
    ordered by ecg_id and the head of the file is not representative.
    """
    key = os.path.abspath(x_train_path)
    if key in _NORM_CACHE:
        return _NORM_CACHE[key]
    X = np.load(x_train_path, mmap_mode="r")
    n = min(max_records, X.shape[0])
    idx = np.sort(np.random.default_rng(seed).choice(X.shape[0], n, replace=False))
    sub = np.asarray(X[idx], dtype=np.float64)
    mean = sub.mean(axis=(0, 2)).astype(np.float32)
    std = np.maximum(sub.std(axis=(0, 2)), 1e-6).astype(np.float32)
    _NORM_CACHE[key] = (mean, std)
    return mean, std


class ECGDataset(Dataset):
    """X: (N, 12, T) float32 raw mV; y: (N, K) float32 multi-hot."""

    def __init__(self, x_path, y_path, augment: bool = False,
                 config: Optional[dict] = None,
                 lead_subset: Optional[str] = None,
                 corruption: Optional[dict] = None,
                 norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 norm_mode: str = "dataset"):
        self.X = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path).astype(np.float32)
        self.augment = augment
        self.config = config or {}
        self.norm_mode = norm_mode

        aug = self.config.get("augmentation", {})
        self.noise_std = float(aug.get("gaussian_noise_std", 0.01))
        self.scale_range = tuple(aug.get("random_scale", [0.9, 1.1]))
        self.shift_max = int(aug.get("random_shift", 50))
        self.p_noise = float(aug.get("p_noise", 0.5))
        self.p_scale = float(aug.get("p_scale", 0.5))
        self.p_shift = float(aug.get("p_shift", 0.5))
        self.p_drop = float(aug.get("p_lead_dropout", 0.1))
        self.p_wander = float(aug.get("p_baseline_wander", 0.0))

        self.lead_idx = (LEAD_SUBSETS[lead_subset]
                         if lead_subset not in (None, "12lead") else None)
        self.corruption = corruption
        self.norm_stats = norm_stats

    def __len__(self):
        return self.X.shape[0]

    @property
    def num_leads(self):
        return len(self.lead_idx) if self.lead_idx is not None else self.X.shape[1]

    def _normalise(self, x):
        if self.norm_mode == "per_record":
            mu = x.mean(axis=1, keepdims=True)
            sd = x.std(axis=1, keepdims=True)
            return np.where(sd < 1e-8, x - mu, (x - mu) / (sd + 1e-8)).astype(np.float32)
        if self.norm_stats is not None:
            mean, std = self.norm_stats
            return ((x - mean[:, None]) / std[:, None]).astype(np.float32)
        return x

    def __getitem__(self, idx):
        x = np.array(self.X[idx], dtype=np.float32)
        y = self.y[idx]
        x = self._normalise(x)
        if self.corruption is not None:
            x = apply_corruption(x, idx=idx, **self.corruption)
        if self.augment:
            x = self._augment(x)
        if self.lead_idx is not None:
            x = x[self.lead_idx]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y)

    def _augment(self, x):
        if np.random.random() < self.p_noise:
            x = x + np.random.normal(0, self.noise_std, x.shape).astype(np.float32)
        if np.random.random() < self.p_scale:
            x = x * np.float32(np.random.uniform(*self.scale_range))
        if np.random.random() < self.p_shift:
            x = np.roll(x, np.random.randint(-self.shift_max, self.shift_max + 1),
                        axis=-1)
        if self.p_wander > 0 and np.random.random() < self.p_wander:
            t = np.arange(x.shape[1], dtype=np.float32)
            wander = (np.random.uniform(0.05, 0.2) *
                      np.sin(2 * np.pi * np.random.uniform(0.05, 0.5) * t / 100.0
                             + np.random.uniform(0, 2 * np.pi)))
            x = x + wander.astype(np.float32)[None, :]
        if np.random.random() < self.p_drop:
            x = x.copy()
            k = np.random.randint(1, 3)
            x[np.random.choice(x.shape[0], k, replace=False)] = 0.0
        return x.astype(np.float32)


# ---------------------------------------------------------------------------
# Wearable-like corruption
# ---------------------------------------------------------------------------
def _bandlimited_noise(n, fs, lo, hi, rng):
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spec = rng.normal(size=len(freqs)) + 1j * rng.normal(size=len(freqs))
    spec = spec * ((freqs >= lo) & (freqs <= hi))
    sig = np.fft.irfft(spec, n=n)
    return sig / (sig.std() + 1e-8)


def apply_corruption(x: np.ndarray, kind: str = "baseline_wander",
                     snr_db: float = 10.0, fs: int = 100,
                     idx: int = 0, seed: int = 1234) -> np.ndarray:
    """
    Add a physiologically plausible artefact at a target per-lead SNR.

    kinds: baseline_wander (respiration/drift, 0.05-0.8 Hz), muscle (EMG,
    15-45 Hz), electrode_motion (0.5-8 Hz with bursts), powerline (50 Hz),
    mixed (all three). Deterministic in `idx`.
    """
    rng = np.random.default_rng(seed + int(idx))
    C, T = x.shape
    sig_power = np.mean(x ** 2, axis=1, keepdims=True) + 1e-12
    target = sig_power / (10 ** (snr_db / 10.0))

    def _make(k):
        if k == "baseline_wander":
            return np.stack([_bandlimited_noise(T, fs, 0.05, 0.8, rng)
                             for _ in range(C)])
        if k == "muscle":
            hi = min(45.0, fs / 2 - 1)
            return np.stack([_bandlimited_noise(T, fs, 15.0, hi, rng)
                             for _ in range(C)])
        if k == "electrode_motion":
            base = np.stack([_bandlimited_noise(T, fs, 0.5, 8.0, rng)
                             for _ in range(C)])
            burst = np.zeros(T)
            for _ in range(int(rng.integers(1, 4))):
                s = int(rng.integers(0, max(1, T - fs)))
                burst[s:s + fs] = 1.0
            burst = np.convolve(burst, np.ones(5) / 5, mode="same")
            return base * (1.0 + 3.0 * burst[None, :])
        if k == "powerline":
            t = np.arange(T) / fs
            ph = rng.uniform(0, 2 * np.pi, size=(C, 1))
            return np.sin(2 * np.pi * 50.0 * t[None, :] + ph)
        raise ValueError(f"unknown corruption '{k}'")

    noise = (sum(_make(k) for k in ("baseline_wander", "muscle",
                                    "electrode_motion")) / 3.0
             if kind == "mixed" else _make(kind))
    noise = noise / (noise.std(axis=1, keepdims=True) + 1e-8)
    return (x + noise * np.sqrt(target)).astype(np.float32)


# ---------------------------------------------------------------------------
class ECGDataModule:
    def __init__(self, data_dir: str = "data/processed",
                 config: Optional[dict] = None,
                 lead_subset: Optional[str] = None,
                 test_corruption: Optional[dict] = None):
        self.data_dir = data_dir
        self.config = config or {}
        tcfg = self.config.get("training", {})
        self.batch_size = int(tcfg.get("batch_size", 128))
        self.num_workers = int(tcfg.get("num_workers", 8))
        self.pin_memory = bool(tcfg.get("pin_memory", True))
        self.lead_subset = lead_subset
        self.test_corruption = test_corruption

        self.norm_mode = self.config.get("data", {}).get("norm_mode", "dataset")
        self._norm_stats = (compute_norm_stats(os.path.join(data_dir, "X_train.npy"))
                            if self.norm_mode == "dataset" else None)

    def _ds(self, split, augment, corruption=None):
        return ECGDataset(
            os.path.join(self.data_dir, f"X_{split}.npy"),
            os.path.join(self.data_dir, f"y_{split}.npy"),
            augment=augment, config=self.config,
            lead_subset=self.lead_subset, corruption=corruption,
            norm_stats=self._norm_stats, norm_mode=self.norm_mode,
        )

    def _loader(self, ds, shuffle, bs=None, seed=42, drop_last=False):
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(
            ds, batch_size=bs or self.batch_size, shuffle=shuffle,
            drop_last=drop_last, generator=g if shuffle else None,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def get_dataloaders(self, seed: int = 42):
        """Training loaders: train is shuffled, augmented, drops last batch."""
        aug = bool(self.config.get("augmentation", {}).get("enabled", True))
        return (self._loader(self._ds("train", aug), True, seed=seed, drop_last=True),
                self._loader(self._ds("val", False), False, bs=self.batch_size * 4),
                self._loader(self._ds("test", False, self.test_corruption), False,
                             bs=self.batch_size * 4))

    def get_eval_loaders(self):
        """
        Deterministic loaders for analysis: no shuffle, no augmentation, no
        dropped samples. Row i of the resulting embedding matrix is record i
        of X_<split>.npy, which prefix probing and patient-level bootstrap
        both depend on.
        """
        bs = self.batch_size * 4
        return (self._loader(self._ds("train", False), False, bs=bs),
                self._loader(self._ds("val", False), False, bs=bs),
                self._loader(self._ds("test", False, self.test_corruption),
                             False, bs=bs))

    def get_external_loader(self, external_dir: str):
        """External corpus as a held-out test set, normalised with train stats."""
        ds = ECGDataset(
            os.path.join(external_dir, "X_test.npy"),
            os.path.join(external_dir, "y_test.npy"),
            augment=False, config=self.config, lead_subset=self.lead_subset,
            norm_stats=self._norm_stats, norm_mode=self.norm_mode,
        )
        return self._loader(ds, False, bs=self.batch_size * 4)
