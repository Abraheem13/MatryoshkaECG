"""
Interpretable ECG descriptors used as probing targets.

Reviewer 2: "the proposed clinical hierarchy of embedding dimensions is
speculative and not validated by probing or interpretability analysis."

To test Table II we need ground-truth values for the physiological quantities
it names (rhythm/RR, QRS duration, ST deviation, T-wave, voltage/P-wave). We
derive them directly from the waveform with NeuroKit2 so that the hierarchy
claim becomes a falsifiable measurement rather than an assertion.

Each descriptor maps to one row of the claimed hierarchy:

    rr_mean, rr_sd, hr        -> d=16  "rhythm, gross RR interval"
    qrs_duration              -> d=32  "QRS duration, bundle branch"
    st_level                  -> d=64  "ST-segment deviation"
    t_amplitude, qrs_axis     -> d=128 "T-wave inversion, axis deviation"
    r_amplitude, p_amplitude  -> d=256 "voltage, P-wave morphology"

If probe performance for, say, `st_level` does not saturate near d=64, the
hierarchy in Table II is not supported and the paper must say so.
"""

from __future__ import annotations

import warnings
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")

FEATURE_NAMES: List[str] = [
    "hr", "rr_mean", "rr_sd", "rr_rmssd",
    "qrs_duration", "qt_interval",
    "st_level", "t_amplitude",
    "r_amplitude", "p_amplitude",
    "qrs_axis",
]

# Which nesting dimension Table II predicts the feature should be resolved by.
HIERARCHY_CLAIM: Dict[str, int] = {
    "hr": 16, "rr_mean": 16, "rr_sd": 16, "rr_rmssd": 16,
    "qrs_duration": 32, "qt_interval": 32,
    "st_level": 64,
    "t_amplitude": 128, "qrs_axis": 128,
    "r_amplitude": 256, "p_amplitude": 256,
}

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


def _safe(val, default=np.nan):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def extract_features_single(sig_12xT: np.ndarray, fs: int = 100) -> Dict[str, float]:
    """
    Extract descriptors from one 12-lead recording, shape (12, T).

    Lead II drives delineation (best P-wave visibility); amplitudes are taken
    from the leads where they are clinically read.
    """
    import neurokit2 as nk

    out = {k: np.nan for k in FEATURE_NAMES}
    lead_ii = np.asarray(sig_12xT[1], dtype=float)
    lead_i = np.asarray(sig_12xT[0], dtype=float)
    lead_avf = np.asarray(sig_12xT[5], dtype=float)
    lead_v5 = np.asarray(sig_12xT[10], dtype=float)

    try:
        cleaned = nk.ecg_clean(lead_ii, sampling_rate=fs)
        rpeaks_info = nk.ecg_peaks(cleaned, sampling_rate=fs)[1]
        rpeaks = np.asarray(rpeaks_info["ECG_R_Peaks"], dtype=int)
    except Exception:
        return out

    if len(rpeaks) < 3:
        return out

    rr = np.diff(rpeaks) / fs * 1000.0  # ms
    out["rr_mean"] = _safe(np.mean(rr))
    out["rr_sd"] = _safe(np.std(rr))
    out["rr_rmssd"] = _safe(np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 1 else np.nan)
    out["hr"] = _safe(60000.0 / np.mean(rr)) if np.mean(rr) > 0 else np.nan

    try:
        _, waves = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs,
                                    method="dwt")
    except Exception:
        waves = {}

    def _idx(key):
        arr = waves.get(key, [])
        arr = np.asarray([a for a in arr if a is not None and np.isfinite(a)],
                         dtype=float)
        return arr.astype(int) if arr.size else np.array([], dtype=int)

    q_on, s_off = _idx("ECG_R_Onsets"), _idx("ECG_R_Offsets")
    t_off = _idx("ECG_T_Offsets")
    t_pk = _idx("ECG_T_Peaks")
    p_pk = _idx("ECG_P_Peaks")

    k = min(len(q_on), len(s_off))
    if k > 0:
        dur = (s_off[:k] - q_on[:k]) / fs * 1000.0
        dur = dur[(dur > 40) & (dur < 250)]
        out["qrs_duration"] = _safe(np.median(dur)) if dur.size else np.nan

    k = min(len(q_on), len(t_off))
    if k > 0:
        qt = (t_off[:k] - q_on[:k]) / fs * 1000.0
        qt = qt[(qt > 200) & (qt < 700)]
        out["qt_interval"] = _safe(np.median(qt)) if qt.size else np.nan

    # ST level: amplitude 60 ms after QRS offset, lead V5, relative to PQ
    if len(s_off) > 0:
        off = int(round(0.06 * fs))
        pts = [lead_v5[j + off] for j in s_off if 0 <= j + off < len(lead_v5)]
        base = [lead_v5[j] for j in q_on if 0 <= j < len(lead_v5)]
        if pts:
            out["st_level"] = _safe(np.median(pts) -
                                    (np.median(base) if base else 0.0))

    if t_pk.size:
        vals = [lead_v5[j] for j in t_pk if 0 <= j < len(lead_v5)]
        out["t_amplitude"] = _safe(np.median(vals)) if vals else np.nan

    if p_pk.size:
        vals = [lead_ii[j] for j in p_pk if 0 <= j < len(lead_ii)]
        out["p_amplitude"] = _safe(np.median(vals)) if vals else np.nan

    vals = [lead_v5[j] for j in rpeaks if 0 <= j < len(lead_v5)]
    out["r_amplitude"] = _safe(np.median(vals)) if vals else np.nan

    # Frontal QRS axis from net deflection in leads I and aVF
    net_i = np.sum([lead_i[j] for j in rpeaks if 0 <= j < len(lead_i)])
    net_f = np.sum([lead_avf[j] for j in rpeaks if 0 <= j < len(lead_avf)])
    out["qrs_axis"] = _safe(np.degrees(np.arctan2(net_f, net_i)))

    return out


def extract_features_batch(X: np.ndarray, fs: int = 100,
                           n_jobs: int = 8, verbose: bool = True) -> np.ndarray:
    """
    X: (N, 12, T) -> (N, len(FEATURE_NAMES)) with NaN for failed delineation.
    """
    from joblib import Parallel, delayed
    from tqdm import tqdm

    def _one(i):
        return extract_features_single(X[i], fs=fs)

    rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one)(i) for i in tqdm(range(X.shape[0]),
                                       desc="  ECG descriptors",
                                       disable=not verbose)
    )
    return np.array([[r[k] for k in FEATURE_NAMES] for r in rows], dtype=np.float32)
