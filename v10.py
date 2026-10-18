# =============================================================================
# ADAPTIVE COGNITIVE PRESSURE FIELD (A-CPF) + KOOPMAN + DFCG
# Kaggle Dual-T4 Research Pipeline - Version 10.0
# =============================================================================
# HOW TO RUN ON KAGGLE (Accelerator: GPU T4 x2, Internet ON if the dataset is
# not attached):
#     Option A (recommended, uses BOTH T4 GPUs in parallel):
#         %%writefile acpf_pipeline_v10.py      <- paste this file in a cell
#         !python acpf_pipeline_v10.py
#     Option B: paste the whole file into one cell and run it (single GPU).
#
# WHAT WAS FIXED vs v9.1 (all verified by a smoke test on synthetic data):
#   [BUG]   pandas>=2.2/3.0: fillna(method="ffill") removed -> crash.
#   [BUG]   matplotlib>=3.9: boxplot(labels=...) deprecated/removed -> crash.
#   [BUG]   Scenario matching with startswith("1") also matched "10_".."13_".
#   [BUG]   Missing questionnaire labels were silently filled with 0.0 and a
#           subject-mean fallback assigned unrelated labels. Now: missing ->
#           segment dropped, never invented.
#   [BUG]   Label scale was re-inferred per data split (train/val could be
#           normalised differently). Now inferred ONCE from the label file.
#   [BUG]   Features with ~0 train variance were divided by 1e-6 on unseen
#           subjects -> 1e6-sized inputs -> NaN. Now std floor = 1 + clipping.
#   [BUG]   Global sort_index() interleaved scenarios whose time stamps overlap.
#   [BUG]   Final holdout was usable_sids[-4:] = 4 NOVICES only. Now stratified.
#   [LEAK]  Validation fold was used for early stopping AND for reporting.
#           Now every fold has an inner validation split; the test fold is used
#           once, for reporting only. Same protocol for every baseline.
#   [LEAK]  Feature schema/normalisation now come from training subjects only.
#   [STAT]  Expert-vs-novice test now on subject means of OUT-OF-FOLD
#           predictions (n=10 vs 10), not on 100k overlapping windows.
#   [T4]    Whole feature tensor lives on the GPU; batches are built by index
#           arithmetic (no Python DataLoader bottleneck, no pin-memory issues).
#   [T4]    Koopman rollout in fp32 (matrix power), SVD clip outside autocast,
#           GRU.flatten_parameters(), DFCG edge MLP without materialised pairs.
#   [T4]    Fold-level parallelism: one job per GPU via CUDA_VISIBLE_DEVICES
#           (DataParallel is slower than this for a model this small).
#   [T4]    Per-epoch window cap -> training time is bounded and predictable.
#   [OPS]   Resumable jobs, time-budget guard, no pip installs, no tqdm spam.
#
# WHAT THIS CODE DOES *NOT* CLAIM: it estimates questionnaire-derived driver
# workload/stress (scenario-level labels), not Emergency-Vehicle detection and
# not a physical "pressure field". Koopman/DFCG are auxiliary branches trained
# jointly; they do not feed the pressure head (see report).
# =============================================================================
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import mannwhitneyu
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

VERSION = "10.0"
T_START = time.time()

# =============================================================================
# 0. ENVIRONMENT, SWITCHES, SEEDS
# =============================================================================
IN_KAGGLE = os.path.exists("/kaggle/working")
SMOKE = os.environ.get("ACPF_SMOKE", "0") == "1"                      # tiny CPU test run
ALLOW_SYNTHETIC_FALLBACK = os.environ.get("ACPF_ALLOW_SYNTHETIC", "1") == "1"
NO_PARALLEL = os.environ.get("ACPF_NO_PARALLEL", "0") == "1"
FORCE_WORKERS = int(os.environ.get("ACPF_FORCE_WORKERS", "0"))        # test hook
TIME_BUDGET_HOURS = float(os.environ.get("ACPF_TIME_BUDGET_H", "10.5"))  # Kaggle: 12h hard limit
RUN_FINAL_MODEL = os.environ.get("ACPF_RUN_FINAL", "1") == "1"
GLOBAL_SEED = 42

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")
_AC_DEV = "cuda" if USE_CUDA else "cpu"
if USE_CUDA:
    torch.backends.cudnn.benchmark = True


def log(msg: str):
    print(msg, flush=True)


def set_seed(seed: int = GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed_all(seed)


def autocast_ctx(enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(enabled and USE_CUDA))


def no_autocast():
    return torch.autocast(device_type=_AC_DEV, enabled=False)


def make_scaler(enabled: bool):
    if not (enabled and USE_CUDA):
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except Exception:
        return torch.cuda.amp.GradScaler()


set_seed()

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
HORIZONS = [5, 10, 15]
MODALITY_PREFIXES = ["Traffic_", "CAN_", "EDA_", "BVP_", "EyeTrack_"]
LABEL_COLS = ["NASA_TLX", "Stress"]
META_COLS = ["subject_id", "scenario"]
BASELINE_SCENARIO = "0_Baseline"
ALL_SCENARIOS = [
    "1_EnterAuxiliaryStreet", "2_EnterMainStreet", "3_U-Turn", "4_StraightDrive",
    "5_RightTurn-P1", "6_RightTurn-P2", "7_LeftTurn-1", "8_LeftTurn-2",
    "9_RightTurn-1", "10_RightTurn-2", "11_RightTurn-3", "12_RightTurn-4", "13_LeftTurn-3",
]
GITHUB_REPO = "https://github.com/AIR-DISCOVER/ExpertDrivingDataset.git"

CONFIG: Dict[str, Any] = {
    "data": {
        "subjects": [f"E{i:02d}" for i in range(1, 11)] + [f"N{i:02d}" for i in range(1, 11)],
        "scenarios": ALL_SCENARIOS,
        "include_baseline": True,
        "feature_clip": 8.0,          # |z| clip after standardisation
    },
    "models": {
        "acpf_hidden": 256, "acpf_dropout": 0.15,
        "koopman_latent": 24, "koopman_enc": [128, 64], "koopman_dec": [64, 128],
        "gru_baseline_hidden": 128,
    },
    "training": {
        "batch_size": 128,
        "eval_batch_size": 512,
        "epochs": 60,                       # final model
        "cv_epochs": 40,                    # per fold
        "patience": 10,
        "cv_patience": 8,
        "lr": 3e-4,
        "baseline_lr": 1e-3,
        "weight_decay": 1e-4,
        "warmup_epochs": 3,
        "min_lr_factor": 0.02,
        "noise_std": 0.02,
        "grad_clip": 1.0,
        "mixed_precision": True,
        "sequence_length": 30,
        "sequence_stride": 2,
        "max_train_windows_per_epoch": 120_000,   # bounds time/epoch; windows overlap ~93%
        "max_val_windows": 60_000,
        "n_folds": 5,
        "inner_val_frac": 0.2,
        "cv_seed": 42,
        "nan_batch_abort_frac": 0.10,
    },
    "loss": {"lambda_koopman": 0.05, "lambda_graph": 0.02, "lambda_koop_reg": 0.005},
}

if SMOKE:  # tiny settings so the whole pipeline can be verified on a CPU in minutes
    CONFIG["models"].update({"acpf_hidden": 64, "gru_baseline_hidden": 32})
    CONFIG["training"].update({
        "batch_size": 64, "eval_batch_size": 128, "epochs": 2, "cv_epochs": 2,
        "patience": 2, "cv_patience": 2, "warmup_epochs": 1,
        "max_train_windows_per_epoch": 1280, "max_val_windows": 1500, "sequence_stride": 6,
    })
    CONFIG["models"]["koopman_enc"] = [32, 16]
    CONFIG["models"]["koopman_dec"] = [16, 32]


def config_hash() -> str:
    blob = json.dumps({k: CONFIG[k] for k in ("data", "models", "training", "loss")},
                      sort_keys=True, default=str) + VERSION
    return hashlib.md5(blob.encode()).hexdigest()[:10]


def make_paths(out_root: str) -> Dict[str, Path]:
    root = Path(out_root)
    p = {k: root / k for k in ("synchronized", "jobs", "figures", "results", "checkpoints", "logs")}
    p["root"] = root
    for v in p.values():
        v.mkdir(parents=True, exist_ok=True)
    return p


# =============================================================================
# 2. SMALL UTILITIES
# =============================================================================
_TIME_LIKE = re.compile(r"(^|_)(time|timestamp|ts|frame|index|idx|date)(_|$)", re.I)


def canon_sid(s: Any) -> str:
    s = str(s).strip()
    m = re.match(r"^([EeNn])[A-Za-z_\-\s]*?0*(\d+)$", s)
    return f"{m.group(1).upper()}{int(m.group(2))}" if m else s


def scen_key(s: Any) -> Optional[int]:
    m = re.match(r"^\D{0,12}?(\d+)", str(s).strip())
    return int(m.group(1)) if m else None


def save_df(df: pd.DataFrame, path: Path):
    try:
        df.to_parquet(path)
    except Exception:
        df.to_csv(path.with_suffix(".csv"), index=True)


def read_df(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    csv = path.with_suffix(".csv")
    if csv.exists():
        return pd.read_csv(csv, index_col=0)
    raise FileNotFoundError(f"Neither {path} nor {csv} exists")


_SUBJECT_CACHE: Dict[str, pd.DataFrame] = {}


def read_subject_cached(sync_dir: Path, sid: str) -> pd.DataFrame:
    key = f"{sync_dir}/{sid}"
    if key not in _SUBJECT_CACHE:
        _SUBJECT_CACHE[key] = read_df(Path(sync_dir) / f"{sid}.parquet")
    return _SUBJECT_CACHE[key]


def regression_metrics(yt: np.ndarray, yp: np.ndarray) -> Dict[str, float]:
    yt, yp = np.asarray(yt, dtype=np.float64), np.asarray(yp, dtype=np.float64)
    if len(yt) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    err = yt - yp
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 1e-12 else float("nan")
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))), "r2": r2}


# =============================================================================
# 3. DATASET DISCOVERY (+ CLEARLY-LABELLED SYNTHETIC FALLBACK FOR DEBUGGING)
# =============================================================================
def is_valid_dataset(p: str) -> bool:
    if not os.path.isdir(p):
        return False
    subs = [d.lower() for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
    return all(any(k in s for s in subs) for k in ("traffic", "can", "driver"))


def _bounded_walk(root: str, max_depth: int = 5):
    root = os.path.abspath(root)
    base = root.count(os.sep)
    for cur, dirs, _ in os.walk(root):
        yield cur
        if cur.count(os.sep) - base >= max_depth:
            dirs[:] = []


def generate_synthetic_dataset(target_dir: str):
    log(f"\n[SYNTHETIC] Generating 20-subject DEBUG dataset at {target_dir} (NOT real data!)")
    rng = np.random.RandomState(0)
    base = Path(target_dir)
    scenarios = [BASELINE_SCENARIO] + ALL_SCENARIOS
    subjects = [f"E{i:02d}" for i in range(1, 11)] + [f"N{i:02d}" for i in range(1, 11)]
    sd = base / "3-Driver" / "5-Scale"
    sd.mkdir(parents=True, exist_ok=True)

    def mult(scen):
        if scen == BASELINE_SCENARIO:
            return 0.5
        return 1.4 if any(k in scen for k in ("Turn", "Enter")) else 0.85

    tlx_rows, st_rows = [], []
    for sid in subjects:
        exp = sid.startswith("E")
        bw, bs = (32.0 if exp else 68.0), (1.9 if exp else 3.6)
        for scen in scenarios:
            tlx_rows.append({"subject_id": sid, "scenario": scen,
                             "value": float(np.clip(bw * mult(scen) + rng.normal(0, 2.5), 10, 95))})
            st_rows.append({"subject_id": sid, "scenario": scen,
                            "value": float(np.clip(bs * mult(scen) + rng.normal(0, 0.15), 1, 5))})
    pd.DataFrame(tlx_rows).to_csv(sd / "NASA-TLX.csv", index=False)
    pd.DataFrame(st_rows).to_csv(sd / "Stress.csv", index=False)

    for sid in subjects:
        dtype = "ExpertDriver" if sid.startswith("E") else "NoviceDriver"
        ts_dir = base / "3-Driver" / "3-EDA" / "Timestamp"
        ts_dir.mkdir(parents=True, exist_ok=True)
        rows, cur = [], 0.0
        for scen in scenarios:
            dur = 25.0 if scen == BASELINE_SCENARIO else 35.0
            rows.append({"scenario": scen, "start": cur, "end": cur + dur})
            cur += dur + 5.0
        pd.DataFrame(rows).to_csv(ts_dir / f"{sid}-timedivision.csv", index=False)
        n = int(cur * 10)
        t = np.linspace(0, cur, n)
        level = np.zeros(n)
        for r in rows:
            level[(t >= r["start"]) & (t <= r["end"])] = mult(r["scenario"]) * (0.4 if sid.startswith("E") else 0.9)
        for mod, col, sig in (("3-EDA", "EDA", 1.5 + level + 0.3 * np.sin(t * 0.04)),
                              ("4-BVP", "BVP", np.sin(t * 1.2 * np.pi) * (1 + level))):
            d = base / "3-Driver" / mod / dtype
            d.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"time": t, col: sig + rng.normal(0, 0.04, n)}).to_csv(d / f"{sid}.csv", index=False)
        for r in rows:
            m = mult(r["scenario"])
            npt = int((r["end"] - r["start"]) * 10)
            st = np.linspace(r["start"], r["end"], npt)
            sp = 45.0 + 8.0 * np.sin(st * 0.15) + rng.normal(0, 0.8, npt)
            acc = np.gradient(sp, st)
            cd = base / "2-CANBus" / r["scenario"] / dtype
            cd.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"time": st, "speed": sp, "acceleration": acc,
                          "brake": np.maximum(0, -acc * 4),
                          "steering_angle": np.sin(st * 0.4) * 12.0 * m,
                          "lane_deviation": np.abs(np.sin(st * 0.08)) * 0.35 * m}).to_csv(cd / f"{sid}.csv", index=False)
            td = base / "1-TrafficRecorder" / r["scenario"] / dtype
            td.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"time": st, "density": np.clip(0.4 + 0.3 * np.sin(st * 0.08), 0.1, 0.9),
                          "vehicle_count": rng.randint(5, 25, npt),
                          "road_complexity": 0.6 * m}).to_csv(td / f"{sid}.csv", index=False)


def find_dataset_path() -> Tuple[str, bool]:
    """Returns (path, is_synthetic)."""
    env = os.environ.get("EXPERT_DRIVING_DATA_PATH")
    if env and is_valid_dataset(env):
        log(f"[DATASET] EXPERT_DRIVING_DATA_PATH -> {env}")
        return env, False
    if os.path.isdir("/kaggle/input"):
        for root in _bounded_walk("/kaggle/input"):
            if is_valid_dataset(root):
                log(f"[DATASET] Found attached Kaggle dataset: {root}")
                return root, False
            sub = os.path.join(root, "Dataset")
            if is_valid_dataset(sub):
                log(f"[DATASET] Found attached Kaggle dataset: {sub}")
                return sub, False
    for lp in ("/kaggle/working/input_data/Dataset", "./data/raw/ExpertDrivingDataset/Dataset",
               "./Dataset", "./data/raw/Dataset"):
        if is_valid_dataset(lp):
            log(f"[DATASET] Found local dataset: {lp}")
            return lp, False

    tmp = "/kaggle/working/temp_repo" if IN_KAGGLE else "./temp_repo"
    target = "/kaggle/working/input_data/Dataset" if IN_KAGGLE else "./data/raw/ExpertDrivingDataset/Dataset"
    try:
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.check_call(["git", "clone", "--depth", "1", GITHUB_REPO, tmp], timeout=900)
        src = os.path.join(tmp, "Dataset")
        if is_valid_dataset(src):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copytree(src, target, dirs_exist_ok=True)
            shutil.rmtree(tmp, ignore_errors=True)
            log(f"[DATASET] Cloned from GitHub -> {target}")
            return target, False
    except Exception as e:
        log(f"[WARN] git clone failed ({type(e).__name__}): attach the dataset or enable Internet.")

    if not ALLOW_SYNTHETIC_FALLBACK:
        raise FileNotFoundError(
            "ExpertDrivingDataset not found. Attach it as a Kaggle dataset, set EXPERT_DRIVING_DATA_PATH, "
            "or enable Internet so it can be cloned.")
    dbg = "/kaggle/working/debug_dataset/Dataset" if IN_KAGGLE else "./debug_dataset/Dataset"
    if not is_valid_dataset(dbg):
        generate_synthetic_dataset(dbg)
    log("=" * 78)
    log("  WARNING: REAL DATASET NOT FOUND -> RUNNING ON SYNTHETIC DEBUG DATA.")
    log("  Results are meaningless for research; outputs go to a *_SYNTHETIC_DEBUG folder.")
    log("=" * 78)
    return dbg, True


# =============================================================================
# 4. QUESTIONNAIRE LABELS (loaded once, scale inferred once)
# =============================================================================
def _find_col(cols, keys, exclude=()):
    for k in keys:
        for c in cols:
            if c not in exclude and k in str(c).lower():
                return c
    return None


def _normalise_label(name: str, v: pd.Series) -> Tuple[pd.Series, str]:
    lo, hi = float(v.min()), float(v.max())
    if name == "NASA_TLX":
        if lo >= 0 and hi <= 1.0 + 1e-9:
            out, how = v, "already [0,1]"
        elif hi <= 10:
            out, how = v / 10.0, "0-10 -> /10"
        elif hi <= 21:
            out, how = (v - 1.0) / 20.0, "1-21 -> (x-1)/20"
        else:
            out, how = v / 100.0, "0-100 -> /100"
    else:  # Stress
        if lo >= 0 and hi <= 1.0 + 1e-9:
            out, how = v, "already [0,1]"
        elif hi <= 5:
            out, how = (v - 1.0) / 4.0, "1-5 -> (x-1)/4"
        elif hi <= 7:
            out, how = (v - 1.0) / 6.0, "1-7 -> (x-1)/6"
        elif hi <= 10:
            out, how = v / 10.0, "0-10 -> /10"
        else:
            out, how = v / 100.0, "0-100 -> /100"
    return out.clip(0.0, 1.0), f"raw range [{lo:.2f}, {hi:.2f}], {how}"


def load_scale_labels(raw_path: str) -> Dict[str, pd.DataFrame]:
    scale_dir = Path(raw_path) / "3-Driver" / "5-Scale"
    files = sorted(scale_dir.glob("*.csv")) if scale_dir.exists() else []
    spec = {"NASA_TLX": (["nasatlx", "tlx"], r"(total|overall|weighted|global|tlx|value|score)"),
            "Stress": (["stress"], r"(stress|value|score|total|level)")}
    out: Dict[str, pd.DataFrame] = {}
    for name, (keys, pref) in spec.items():
        path = next((f for f in files if any(k in re.sub(r"[^a-z0-9]", "", f.stem.lower()) for k in keys)), None)
        if path is None:
            continue
        try:
            df = pd.read_csv(path)
            df.columns = [str(c).strip() for c in df.columns]
            sid_col = _find_col(df.columns, ["subject", "participant", "driver", "id"]) or df.columns[0]
            scen_col = _find_col(df.columns, ["scenario", "condition", "task", "scene"], exclude=(sid_col,))
            skip = re.compile(r"(trial|order|index|idx|time|session|age|gender)", re.I)
            cand = [c for c in df.columns if c not in (sid_col, scen_col) and not skip.search(c)]
            for c in cand:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            cand = [c for c in cand if df[c].notna().any()]
            if scen_col:
                chosen = [c for c in cand if re.search(pref, c, re.I)][:1] or cand
                val = df[chosen].mean(axis=1)
                tab = pd.DataFrame({"subject_id": df[sid_col].astype(str).str.strip(),
                                    "scenario": df[scen_col].astype(str).str.strip(), "value": val})
            else:
                tab = df.melt(id_vars=[sid_col], value_vars=cand, var_name="scenario", value_name="value")
                tab = tab.rename(columns={sid_col: "subject_id"})
                tab["subject_id"] = tab["subject_id"].astype(str).str.strip()
                tab["scenario"] = tab["scenario"].astype(str).str.strip()
            tab["value"] = pd.to_numeric(tab["value"], errors="coerce")
            tab = tab.dropna(subset=["value"]).copy()
            if tab.empty:
                continue
            tab["value"], how = _normalise_label(name, tab["value"])
            tab["sid_c"] = tab["subject_id"].map(canon_sid)
            tab["scen_k"] = tab["scenario"].map(scen_key)
            out[name] = tab.reset_index(drop=True)
            log(f"  [LABELS] {name}: {len(tab)} ratings, {tab['sid_c'].nunique()} subjects  ({how})  <- {path.name}")
        except Exception as e:
            log(f"  [WARN] could not parse {path.name}: {type(e).__name__}: {e}")
    if not out:
        raise FileNotFoundError(
            f"No usable NASA-TLX / Stress file in {scale_dir}. Files seen: {[f.name for f in files]}. "
            "Synthetic labels are intentionally not generated.")
    return out


# =============================================================================
# 5. MULTIMODAL SYNCHRONISATION
# =============================================================================
class DataSynchronizer:
    def __init__(self, raw_path: str, sync_dir: Path, labels: Dict[str, pd.DataFrame], cfg: Dict):
        self.raw = Path(raw_path)
        self.sync_dir = Path(sync_dir)
        self.scenarios = cfg["data"]["scenarios"]
        self.include_bl = cfg["data"]["include_baseline"]
        self.labels = labels
        self.sync_dir.mkdir(parents=True, exist_ok=True)

    # ---- loading helpers ----
    def _load_csv(self, fp: Path, start: float = 0.0, end: Optional[float] = None) -> pd.DataFrame:
        if not fp.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(fp)
            if df.shape[1] < 2:
                return pd.DataFrame()
            df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
            tcol = next((c for c in df.columns if "time" in c.lower()), df.columns[0])
            df[tcol] = pd.to_numeric(df[tcol], errors="coerce")
            df = df.dropna(subset=[tcol])
            df = df[df[tcol] >= start]
            if end is not None:
                df = df[df[tcol] <= end]
            if df.empty:
                return pd.DataFrame()
            df = df.set_index(tcol)
            df.index.name = None
            df = df[~df.index.duplicated(keep="first")].sort_index()
            df = df[[c for c in df.columns if not _TIME_LIKE.search(str(c))]]
            df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
            if df.shape[1] == 0:
                return pd.DataFrame()
            df = df.interpolate(method="index", limit_direction="both")
            return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        except Exception:
            return pd.DataFrame()

    def _load_timediv(self, sid: str) -> Optional[pd.DataFrame]:
        for mod in ("3-EDA", "4-BVP"):
            for name in (f"{sid}-timedivision.csv", f"{sid}_timedivision.csv", f"{sid}.csv"):
                p = self.raw / "3-Driver" / mod / "Timestamp" / name
                if p.exists():
                    try:
                        return pd.read_csv(p)
                    except Exception:
                        pass
        return None

    @staticmethod
    def _time_window(td: Optional[pd.DataFrame], scenario: str) -> Tuple[float, Optional[float], bool]:
        if td is None or td.empty:
            return 0.0, None, True          # no time division at all: whole recording
        cm = {re.sub(r"[^a-z]", "", str(c).lower()): c for c in td.columns}
        sc = cm.get("scenario") or cm.get("scene") or cm.get("task")
        st = cm.get("start") or cm.get("starttime")
        en = cm.get("end") or cm.get("endtime")
        if not (sc and st and en):
            return 0.0, None, True
        names = td[sc].astype(str).str.strip()
        rows = td[names == scenario]
        if rows.empty:
            k = scen_key(scenario)
            if k is not None:
                rows = td[names.map(scen_key) == k]
        if rows.empty:
            return 0.0, None, False          # scenario not in the time division: skip it
        return float(rows[st].iloc[0]), float(rows[en].iloc[0]), True

    @staticmethod
    def _slice(df: pd.DataFrame, start: float, end: Optional[float]) -> pd.DataFrame:
        if df.empty:
            return df
        r = df[df.index >= start]
        return r[r.index <= end] if end is not None else r

    @staticmethod
    def _resample(df: pd.DataFrame, ref: pd.Index) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(index=ref)
        df = df.copy()
        df.index = pd.to_numeric(df.index, errors="coerce")
        df = df[~df.index.isna()]
        df = df[~df.index.duplicated(keep="first")].sort_index()
        tol = 0.5
        if len(ref) > 1:
            tol = float(np.median(np.diff(ref.values.astype(float)))) * 0.8
            tol = max(tol, 1e-9)
        out = df.reindex(ref, method="nearest", tolerance=tol)
        return out.interpolate(method="index", limit_direction="both").fillna(0.0)

    def _label(self, name: str, sid: str, scenario: str) -> float:
        tab = self.labels.get(name)
        if tab is None:
            return float("nan")
        rows = tab[tab["sid_c"] == canon_sid(sid)]
        if rows.empty:
            return float("nan")
        ex = rows[rows["scenario"] == scenario]
        if ex.empty:
            k = scen_key(scenario)
            ex = rows[rows["scen_k"] == k] if k is not None else rows.iloc[0:0]
        return float(ex["value"].mean()) if len(ex) else float("nan")

    # ---- main ----
    def synchronize_subject(self, sid: str) -> int:
        dtype = "ExpertDriver" if sid.upper().startswith("E") else "NoviceDriver"
        scen_list = ([BASELINE_SCENARIO] if self.include_bl else []) + self.scenarios
        td = self._load_timediv(sid)
        eda = self._load_csv(self.raw / "3-Driver" / "3-EDA" / dtype / f"{sid}.csv")
        bvp = self._load_csv(self.raw / "3-Driver" / "4-BVP" / dtype / f"{sid}.csv")
        segs, skipped = [], []
        for scen in scen_list:
            start, end, found = self._time_window(td, scen)
            if not found:
                skipped.append(scen)
                continue
            fr: Dict[str, pd.DataFrame] = {}
            for nm, fp in (("Traffic", self.raw / "1-TrafficRecorder" / scen / dtype / f"{sid}.csv"),
                           ("CAN", self.raw / "2-CANBus" / scen / dtype / f"{sid}.csv"),
                           ("EyeTrack", self.raw / "3-Driver" / "2-EyeTracking" / scen / dtype / f"{sid}.csv")):
                d = self._load_csv(fp, start, end)
                if not d.empty:
                    fr[nm] = d
            if not eda.empty:
                fr["EDA"] = self._slice(eda, start, end)
            if not bvp.empty:
                fr["BVP"] = self._slice(bvp, start, end)
            ref = None
            for key in ("CAN", "Traffic", "EDA", "BVP"):
                if key in fr and not fr[key].empty:
                    ref = fr[key].index
                    break
            if ref is None:
                continue
            parts = [self._resample(d, ref).add_prefix(f"{nm}_") for nm, d in fr.items() if not d.empty]
            al = pd.concat(parts, axis=1)
            al.columns = [re.sub(r"\s+", "_", str(c)) for c in al.columns]
            al = al.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            al["subject_id"] = sid
            al["scenario"] = scen
            for ln in LABEL_COLS:
                al[ln] = self._label(ln, sid, scen)
            segs.append(al)
        if skipped:
            log(f"  [SYNC] {sid}: scenarios not in time-division and skipped: {skipped}")
        if not segs:
            log(f"  [WARN] {sid}: no usable segments")
            return 0
        comb = pd.concat(segs, axis=0, sort=False)      # NO global sort: keeps scenario blocks intact
        feat = [c for c in comb.columns if c not in META_COLS and c not in LABEL_COLS]
        comb[feat] = comb[feat].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        for ln in LABEL_COLS:
            comb[ln] = comb[ln].astype(np.float32)
        save_df(comb, self.sync_dir / f"{sid}.parquet")
        cov = float(comb[LABEL_COLS].notna().any(axis=1).mean())
        log(f"  [SYNC] {sid}: {len(comb):>8d} rows, {len(feat):>3d} features, {len(segs)} segments, label coverage {cov:.0%}")
        return len(comb)


def build_feature_schema(sync_dir: Path, subjects: List[str]) -> List[str]:
    cols = set()
    for sid in subjects:
        df = read_subject_cached(sync_dir, sid)
        for c in df.columns:
            if c in META_COLS or c in LABEL_COLS or _TIME_LIKE.search(str(c)):
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.add(str(c))
    return sorted(cols)


def make_groups(cols: List[str]) -> Tuple[List[str], List[List[int]]]:
    names, groups = [], []
    for p in MODALITY_PREFIXES:
        idx = [i for i, c in enumerate(cols) if c.startswith(p)]
        if idx:
            names.append(p.rstrip("_"))
            groups.append(idx)
    used = {i for g in groups for i in g}
    other = [i for i in range(len(cols)) if i not in used]
    if other:
        names.append("Other")
        groups.append(other)
    return names, groups


# =============================================================================
# 6. LEAK-FREE WINDOWS (segment-bounded) + GPU-RESIDENT BATCHING
# =============================================================================
class WindowData:
    """All windows (and their future targets) lie inside ONE (subject, scenario) segment."""

    def __init__(self, sync_dir: Path, subjects: List[str], cfg: Dict, feat_cols: List[str],
                 mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None, name: str = ""):
        tr = cfg["training"]
        self.L = tr["sequence_length"]
        stride = tr["sequence_stride"]
        H = max(HORIZONS)
        clip = cfg["data"]["feature_clip"]
        self.feat_cols = list(feat_cols)
        D = len(self.feat_cols)

        raw_x, raw_y, meta = [], [], []
        n_nolabel = n_short = 0
        for sid in subjects:
            df = read_subject_cached(sync_dir, sid)
            scen_col = df["scenario"].astype(str).to_numpy() if "scenario" in df.columns else np.full(len(df), "default")
            X_all = df.reindex(columns=self.feat_cols).to_numpy(dtype=np.float32, na_value=np.nan)
            labs = [df[c].to_numpy(dtype=np.float64) for c in LABEL_COLS if c in df.columns]
            order, seen = [], set()
            for s in scen_col:
                if s not in seen:
                    seen.add(s)
                    order.append(s)
            for scen in order:
                idx = np.flatnonzero(scen_col == scen)
                if len(idx) < self.L + H:
                    n_short += 1
                    continue
                if not labs:
                    n_nolabel += 1
                    continue
                with np.errstate(all="ignore"):
                    Y = np.nanmean(np.vstack([l[idx] for l in labs]), axis=0)
                if not np.isfinite(Y).all():
                    n_nolabel += 1
                    continue
                raw_x.append(np.nan_to_num(X_all[idx], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
                raw_y.append(Y.astype(np.float32))
                meta.append((sid, scen))
        if not raw_x:
            raise ValueError(f"[{name}] no usable segments for {subjects} (no label: {n_nolabel}, too short: {n_short}).")

        if mean is None or std is None:
            n = sum(len(x) for x in raw_x)
            s1 = np.zeros(D, dtype=np.float64)
            s2 = np.zeros(D, dtype=np.float64)
            for x in raw_x:
                s1 += x.sum(0, dtype=np.float64)
                s2 += (x.astype(np.float64) ** 2).sum(0)
            mean = (s1 / n).astype(np.float32)
            std = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0.0)).astype(np.float32)
            std = np.where(std < 1e-6, 1.0, std).astype(np.float32)     # near-constant -> no blow-up
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)

        feats, press, starts, wseg = [], [], [], []
        self.seg_subject, self.seg_scenario, offsets, off = [], [], [], 0
        for si, (x, y, (sid, scen)) in enumerate(zip(raw_x, raw_y, meta)):
            z = np.clip((x - self.mean) / self.std, -clip, clip).astype(np.float32)
            feats.append(z)
            press.append(y)
            rel = np.arange(0, len(z) - self.L - H + 1, stride, dtype=np.int64)
            starts.append(rel + off)
            wseg.append(np.full(len(rel), si, dtype=np.int32))
            self.seg_subject.append(sid)
            self.seg_scenario.append(scen)
            offsets.append(off)
            off += len(z)
        self.feats = np.ascontiguousarray(np.concatenate(feats, 0))
        self.press = np.concatenate(press, 0)
        self.starts = np.concatenate(starts, 0)
        self.win_seg = np.concatenate(wseg, 0)
        self.seg_offset = np.asarray(offsets, dtype=np.int64)
        self.win_pos = self.starts - self.seg_offset[self.win_seg]
        if len(self.starts) == 0:
            raise ValueError(f"[{name}] zero windows were produced.")
        log(f"  [DATA] {name:<10s} subjects={len(subjects):>2d} segments={len(raw_x):>3d} "
            f"windows={len(self.starts):>8d} D={D}  (dropped: no-label={n_nolabel}, too-short={n_short})")

    def y_last(self, widx: np.ndarray) -> np.ndarray:
        return self.press[self.starts[widx] + self.L - 1]


class WindowLoader:
    def __init__(self, wd: WindowData, device: torch.device, batch_size: int, shuffle: bool = False,
                 drop_last: bool = False, select: Optional[np.ndarray] = None,
                 max_per_epoch: Optional[int] = None, seed: int = 0):
        self.device, self.L = device, wd.L
        self.bs, self.shuffle, self.drop_last, self.cap = batch_size, shuffle, drop_last, max_per_epoch
        self.feats = torch.from_numpy(wd.feats).to(device)
        self.press = torch.from_numpy(wd.press).to(device)
        widx = np.arange(len(wd.starts)) if select is None else np.asarray(select, dtype=np.int64)
        self.starts = torch.from_numpy(wd.starts[widx]).to(device)
        self.widx = torch.from_numpy(widx).to(device)
        self.ar = torch.arange(self.L, device=device)
        self.hz = torch.tensor(HORIZONS, device=device, dtype=torch.long)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    def n_eff(self) -> int:
        n = len(self.starts)
        return min(n, self.cap) if (self.shuffle and self.cap) else n

    def __len__(self) -> int:
        n = self.n_eff()
        return n // self.bs if self.drop_last else (n + self.bs - 1) // self.bs

    def __iter__(self):
        n = len(self.starts)
        if self.shuffle:
            order = torch.randperm(n, generator=self.gen, device=self.device)
            if self.cap and n > self.cap:
                order = order[: self.cap]
        else:
            order = torch.arange(n, device=self.device)
        for i in range(0, len(order), self.bs):
            sel = order[i: i + self.bs]
            if self.drop_last and len(sel) < self.bs:
                break
            s = self.starts[sel]
            rows = s[:, None] + self.ar[None, :]
            yield {"features": self.feats[rows],
                   "pressure": self.press[rows].unsqueeze(-1),
                   "future": self.feats[s[:, None] + self.L + self.hz[None, :] - 1],
                   "widx": self.widx[sel]}


def even_subsample(n: int, cap: Optional[int]) -> Optional[np.ndarray]:
    if cap is None or n <= cap:
        return None
    return np.unique(np.linspace(0, n - 1, cap).astype(np.int64))


# =============================================================================
# 7. MODELS
# =============================================================================
class BaselineVanillaGRU(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.gru.flatten_parameters()
        h, _ = self.gru(torch.nan_to_num(x))
        return self.head(h)


class ACPF(nn.Module):
    """Gated multimodal fusion: one gate per REAL modality group (Traffic/CAN/EDA/BVP/EyeTrack)."""

    def __init__(self, input_dim: int, groups: List[List[int]], hidden: int = 256, dropout: float = 0.15):
        super().__init__()
        self.n_groups = len(groups)
        for i, g in enumerate(groups):
            self.register_buffer(f"gidx{i}", torch.as_tensor(g, dtype=torch.long), persistent=False)
        self.gate_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(len(g), max(16, min(64, len(g)))), nn.GELU(),
                          nn.Linear(max(16, min(64, len(g))), 1)) for g in groups])
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden // 2, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.nan_to_num(x)
        chunks = [x.index_select(-1, getattr(self, f"gidx{i}")) for i in range(self.n_groups)]
        scores = torch.cat([net(c) for net, c in zip(self.gate_nets, chunks)], dim=-1)
        gates = F.softmax(scores.float(), dim=-1)
        weighted = torch.cat([c * (gates[..., i:i + 1] * self.n_groups).to(c.dtype)
                              for i, c in enumerate(chunks)], dim=-1)
        h = self.proj(weighted)
        self.gru.flatten_parameters()
        g, _ = self.gru(h)
        h = self.ln(h + g)
        return self.head(h), gates


class KoopmanModel(nn.Module):
    def __init__(self, input_dim: int, latent: int, enc: List[int], dec: List[int], steps: List[int]):
        super().__init__()
        self.latent, self.steps = latent, list(steps)
        layers, prev = [], input_dim
        for h in enc:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, latent))
        self.encoder = nn.Sequential(*layers)
        layers, prev = [], latent
        for h in dec:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*layers)
        self.K = nn.Parameter(torch.randn(latent, latent) / math.sqrt(latent))

    def stable_K(self) -> torch.Tensor:
        """Spectral norm clipped to 1; straight-through gradient to K. Always fp32."""
        with torch.no_grad(), no_autocast():
            Kf = self.K.float()
            try:
                U, S, Vh = torch.linalg.svd(Kf)
                Ks = (U * S.clamp(max=1.0)) @ Vh
            except Exception:
                Ks = Kf / torch.clamp(torch.linalg.norm(Kf), min=1.0)
        return self.K + (Ks - self.K).detach()

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        x = torch.nan_to_num(x)
        z_flat = self.encoder(x.reshape(B * T, D))
        z = z_flat.view(B, T, self.latent)
        x_rec = self.decoder(z_flat).view(B, T, D)
        with no_autocast():
            Kt = self.stable_K().float().t()
            z_last = z[:, -1, :].float()
            z_future = torch.stack([z_last @ torch.linalg.matrix_power(Kt, k) for k in self.steps], dim=1)  # (B,nh,latent)
        fc = self.decoder(z_future.reshape(-1, self.latent)).view(B, len(self.steps), D)
        return x_rec, fc, z


class DFCG(nn.Module):
    """Dynamic feature-coupling graph over latent nodes. Edge MLP(2->h->1) evaluated without
    materialising the N*N pair tensor: first layer is a_i + b_j (identical maths)."""

    def __init__(self, n_nodes: int, hidden: int = 32):
        super().__init__()
        self.l1 = nn.Linear(2, hidden)
        self.l2 = nn.Linear(hidden, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, T, N = z.shape
        s = z.reshape(B * T, N)
        W = self.l1.weight
        a = (s.unsqueeze(-1) * W[:, 0]).to(s.dtype)       # (BT,N,h)
        b = (s.unsqueeze(-1) * W[:, 1]).to(s.dtype)
        h = F.relu(a.unsqueeze(2) + b.unsqueeze(1) + self.l1.bias.to(s.dtype))   # (BT,N,N,h)
        adj = torch.sigmoid(self.l2(h).squeeze(-1))        # (BT,N,N)
        coupled = torch.bmm(adj, s.unsqueeze(-1)).squeeze(-1)
        return torch.sigmoid(s + coupled).view(B, T, N)


class UnifiedModel(nn.Module):
    def __init__(self, acpf: ACPF, koop: KoopmanModel, dfcg: DFCG):
        super().__init__()
        self.acpf, self.koopman, self.dfcg = acpf, koop, dfcg

    def forward(self, x):
        p, gates = self.acpf(x)
        x_rec, fc, z = self.koopman(x)
        zg = self.dfcg(z)
        return p, gates, x_rec, fc, zg


class CombinedLoss(nn.Module):
    def __init__(self, lk: float, lg: float, lreg: float):
        super().__init__()
        self.lk, self.lg, self.lreg = lk, lg, lreg

    def forward(self, p, y, x_rec, x_in, fc, fut, zg, K):
        l_cpf = F.mse_loss(p.float(), y.float())
        l_rec = F.mse_loss(x_rec.float(), x_in.float())
        l_koop = F.mse_loss(fc.float(), fut.float())
        zg = zg.float()
        l_graph = ((zg[:, 1:] - zg[:, :-1]) ** 2).mean()
        reg = torch.linalg.norm(K.float())
        total = l_cpf + self.lk * (l_rec + l_koop) + self.lg * l_graph + self.lreg * reg
        return total, l_cpf.detach()


# =============================================================================
# 8. TRAINER (used for the full model AND the GRU baseline, identical protocol)
# =============================================================================
class Trainer:
    def __init__(self, cfg: Dict, kind: str, feat_dim: int, groups: List[List[int]], tag: str,
                 ckpt_path: Optional[Path] = None):
        assert kind in ("full", "gru")
        tr, m = cfg["training"], cfg["models"]
        self.cfg, self.kind, self.tag, self.ckpt_path = cfg, kind, tag, ckpt_path
        self.device = DEVICE
        if kind == "full":
            acpf = ACPF(feat_dim, groups, m["acpf_hidden"], m["acpf_dropout"])
            koop = KoopmanModel(feat_dim, m["koopman_latent"], m["koopman_enc"], m["koopman_dec"], HORIZONS)
            self.model = UnifiedModel(acpf, koop, DFCG(m["koopman_latent"])).to(self.device)
            self.base_lr = tr["lr"]
            self.crit = CombinedLoss(cfg["loss"]["lambda_koopman"], cfg["loss"]["lambda_graph"],
                                     cfg["loss"]["lambda_koop_reg"])
        else:
            self.model = BaselineVanillaGRU(feat_dim, m["gru_baseline_hidden"]).to(self.device)
            self.base_lr = tr["baseline_lr"]
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr, weight_decay=tr["weight_decay"])
        self.use_amp = bool(tr["mixed_precision"] and USE_CUDA)
        self.scaler = make_scaler(self.use_amp)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.history: List[Dict] = []
        self.epochs_run = 0

    def _lr_factor(self, ep: int, total: int) -> float:
        tr = self.cfg["training"]
        w = max(1, tr["warmup_epochs"])
        if ep < w:
            return (ep + 1) / w
        prog = (ep - w) / max(1, total - w)
        mf = tr["min_lr_factor"]
        return mf + (1 - mf) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    def _cpf(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.acpf(x)[0] if self.kind == "full" else self.model(x)

    def _train_epoch(self, loader: WindowLoader) -> Tuple[float, int, int]:
        tr = self.cfg["training"]
        self.model.train()
        tot = torch.zeros((), device=self.device)
        n_ok = n_bad = 0
        for b in loader:
            x, y = b["features"], b["pressure"]
            if tr["noise_std"] > 0:
                x = x + torch.randn_like(x) * tr["noise_std"]
            with autocast_ctx(self.use_amp):
                if self.kind == "full":
                    p, _g, xr, fc, zg = self.model(x)
                    loss, _ = self.crit(p, y, xr, x, fc, b["future"], zg, self.model.koopman.K)
                else:
                    loss = F.mse_loss(self.model(x).float(), y.float())
            if not bool(torch.isfinite(loss)):
                n_bad += 1
                continue
            self.opt.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), tr["grad_clip"])
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), tr["grad_clip"])
                self.opt.step()
            tot += loss.detach()
            n_ok += 1
        return float(tot) / max(n_ok, 1), n_ok, n_bad

    @torch.no_grad()
    def predict(self, loader: WindowLoader, want_gates: bool = False):
        """Last-time-step prediction per window (the step that has seen the full 30-sample context)."""
        self.model.eval()
        W, P, G = [], [], []
        for b in loader:
            with autocast_ctx(self.use_amp):
                if self.kind == "full":
                    out, gates = self.model.acpf(b["features"])
                    if want_gates:
                        G.append(gates[:, -1, :].float().cpu())
                else:
                    out = self.model(b["features"])
            P.append(out[:, -1, 0].float().cpu())
            W.append(b["widx"].cpu())
        res = (torch.cat(W).numpy(), torch.cat(P).numpy())
        return res + ((torch.cat(G).numpy(),) if want_gates else ())

    def _val_mse(self, loader: WindowLoader, wd: WindowData) -> float:
        widx, pred = self.predict(loader)
        return float(np.mean((pred - wd.y_last(widx)) ** 2))

    def fit(self, train_loader: WindowLoader, val_loader: WindowLoader, val_wd: WindowData,
            epochs: int, patience: int) -> List[Dict]:
        tr = self.cfg["training"]
        best, best_state, bad, t_fit = float("inf"), None, 0, time.time()
        for ep in range(epochs):
            f = self._lr_factor(ep, epochs)
            for g in self.opt.param_groups:
                g["lr"] = self.base_lr * f
            t0 = time.time()
            tl, n_ok, n_bad = self._train_epoch(train_loader)
            vl = self._val_mse(val_loader, val_wd)
            dt = time.time() - t0
            self.epochs_run = ep + 1
            self.history.append({"epoch": ep + 1, "train_loss": tl, "val_mse_last": vl,
                                 "lr": self.base_lr * f, "sec": dt, "skipped_batches": n_bad})
            improved = np.isfinite(vl) and vl < best - 1e-7
            if improved:
                best, bad = vl, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                if self.ckpt_path is not None:
                    torch.save(best_state, self.ckpt_path)
            else:
                bad += 1
            eta = dt * (epochs - ep - 1) / 60
            log(f"  [{self.tag}] ep {ep + 1:>2d}/{epochs} train={tl:.5f} val_mse={vl:.5f} "
                f"best={best:.5f} lr={self.base_lr * f:.1e} {dt:5.1f}s  worst-case ETA {eta:5.1f} min"
                f"{'  *' if improved else ''}{f'  skipped={n_bad}' if n_bad else ''}")
            if n_ok == 0 or n_bad > tr["nan_batch_abort_frac"] * max(1, n_ok + n_bad):
                log(f"  [{self.tag}] WARNING: {n_bad} non-finite batches this epoch -> stopping, restoring best.")
                break
            if bad >= patience:
                log(f"  [{self.tag}] early stopping at epoch {ep + 1} (best val_mse={best:.5f})")
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.fit_seconds = time.time() - t_fit
        self.best_val = best
        return self.history

    def free(self):
        del self.model, self.opt
        self.scaler = None
        gc.collect()
        if USE_CUDA:
            torch.cuda.empty_cache()


# =============================================================================
# 9. STATIC BASELINES + EVALUATION
# =============================================================================
def window_mean_matrix(loader: WindowLoader, wd: WindowData) -> Tuple[np.ndarray, np.ndarray]:
    sh, cap = loader.shuffle, loader.cap
    loader.shuffle, loader.cap = False, None
    X, W = [], []
    for b in loader:
        X.append(b["features"].mean(1).float().cpu())
        W.append(b["widx"].cpu())
    loader.shuffle, loader.cap = sh, cap
    widx = torch.cat(W).numpy()
    return torch.cat(X).numpy(), wd.y_last(widx)


def segment_r2(wd: WindowData, widx: np.ndarray, yt: np.ndarray, yp: np.ndarray) -> float:
    d = pd.DataFrame({"seg": wd.win_seg[widx], "t": yt, "p": yp}).groupby("seg").mean()
    return regression_metrics(d["t"].values, d["p"].values)["r2"]


def stratified_split(subjects: List[str], frac: float, seed: int) -> Tuple[List[str], List[str]]:
    """Returns (rest, picked) with `frac` of each driver group picked (>=1 per group when group>=2)."""
    rng = np.random.RandomState(seed)
    picked = []
    for pre in ("E", "N"):
        grp = sorted(s for s in subjects if s.upper().startswith(pre))
        if len(grp) >= 2:
            k = max(1, int(round(frac * len(grp))))
            picked += list(rng.permutation(grp)[:k])
    if not picked:
        picked = [sorted(subjects)[-1]]
    return [s for s in subjects if s not in picked], sorted(picked)


def make_subject_folds(subjects: List[str], n_folds: int, seed: int) -> List[Tuple[List[str], List[str]]]:
    rng = np.random.RandomState(seed)
    buckets: List[List[str]] = [[] for _ in range(n_folds)]
    c = 0
    for pre in ("E", "N"):
        grp = list(rng.permutation(sorted(s for s in subjects if s.upper().startswith(pre))))
        for s in grp:
            buckets[c % n_folds].append(s)
            c += 1
    return [([s for j, bk in enumerate(buckets) if j != i for s in bk], sorted(buckets[i])) for i in range(n_folds)]


def _job_dir(paths: Dict[str, Path], name: str) -> Path:
    d = Path(paths["jobs"]) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# =============================================================================
# 10. JOBS: one CV fold  /  the final model
# =============================================================================
def run_fold_job(f_idx: int, meta: Dict, cfg: Dict, paths: Dict[str, Path]):
    t_job = time.time()
    tr = cfg["training"]
    sync = Path(meta["sync_dir"])
    folds = make_subject_folds(meta["subjects"], meta["n_folds"], tr["cv_seed"])
    train_all, test_subs = folds[f_idx - 1]
    train_subs, val_subs = stratified_split(train_all, tr["inner_val_frac"], tr["cv_seed"] + f_idx)
    log(f"\n>>> FOLD {f_idx}/{meta['n_folds']}  train={train_subs}  inner-val={val_subs}  TEST={test_subs}")

    schema = build_feature_schema(sync, train_all)
    names, groups = make_groups(schema)
    log(f"  [SCHEMA] {len(schema)} features; modality groups: " + ", ".join(f"{n}={len(g)}" for n, g in zip(names, groups)))
    wtr = WindowData(sync, train_subs, cfg, schema, name="train")
    wiv = WindowData(sync, val_subs, cfg, schema, wtr.mean, wtr.std, name="inner-val")
    wte = WindowData(sync, test_subs, cfg, schema, wtr.mean, wtr.std, name="TEST")

    ltr = WindowLoader(wtr, DEVICE, tr["batch_size"], shuffle=True, drop_last=len(wtr.starts) >= 2 * tr["batch_size"],
                       max_per_epoch=tr["max_train_windows_per_epoch"], seed=tr["cv_seed"] + f_idx)
    sel = even_subsample(len(wiv.starts), tr["max_val_windows"])
    liv = WindowLoader(wiv, DEVICE, tr["eval_batch_size"], select=sel)
    lte = WindowLoader(wte, DEVICE, tr["eval_batch_size"])

    rows: List[Dict] = []

    def add(model_name, widx, pred, secs, epochs_run):
        yt = wte.y_last(widx)
        m = regression_metrics(yt, pred)
        m["r2_segment"] = segment_r2(wte, widx, yt, pred)
        rows.append({"fold": f_idx, "model": model_name, **m, "n_test_windows": len(widx),
                     "train_sec": secs, "epochs_run": epochs_run})
        log(f"  [RESULT] fold {f_idx} {model_name:<36s} RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} R2={m['r2']:.4f} R2seg={m['r2_segment']:.4f}")

    # --- static baselines
    Xtr, ytr = window_mean_matrix(ltr, wtr)
    Xiv, yiv = window_mean_matrix(liv, wiv)
    Xte, _ = window_mean_matrix(lte, wte)
    w_all = np.arange(len(wte.starts))
    add("Mean predictor (dummy)", w_all, np.full(len(w_all), float(ytr.mean())), 0.0, 0)
    best_a, best_e = 1.0, float("inf")
    for a in (0.1, 1.0, 10.0, 100.0, 1000.0):
        e = float(np.mean((Ridge(alpha=a).fit(Xtr, ytr).predict(Xiv) - yiv) ** 2))
        if e < best_e:
            best_a, best_e = a, e
    add(f"Linear ridge (alpha={best_a:g})", w_all, np.clip(Ridge(alpha=best_a).fit(Xtr, ytr).predict(Xte), 0, 1), 0.0, 0)

    # --- vanilla GRU baseline
    set_seed(tr["cv_seed"] + f_idx)
    gru = Trainer(cfg, "gru", len(schema), groups, tag=f"fold{f_idx}|gru")
    gru.fit(ltr, liv, wiv, tr["cv_epochs"], tr["cv_patience"])
    widx, pred = gru.predict(lte)
    add("Vanilla GRU baseline", widx, pred, gru.fit_seconds, gru.epochs_run)
    pd.DataFrame(gru.history).to_csv(_job_dir(paths, f"fold{f_idx}") / "history_gru.csv", index=False)
    gru.free()

    # --- proposed model
    set_seed(tr["cv_seed"] + f_idx)
    full = Trainer(cfg, "full", len(schema), groups, tag=f"fold{f_idx}|full")
    log(f"  [full] parameters: {full.n_params:,}")
    full.fit(ltr, liv, wiv, tr["cv_epochs"], tr["cv_patience"])
    widx, pred, gates = full.predict(lte, want_gates=True)
    add("A-CPF + Koopman + DFCG (proposed)", widx, pred, full.fit_seconds, full.epochs_run)
    pd.DataFrame(full.history).to_csv(_job_dir(paths, f"fold{f_idx}") / "history_full.csv", index=False)

    oof = pd.DataFrame({
        "fold": f_idx,
        "subject_id": [wte.seg_subject[i] for i in wte.win_seg[widx]],
        "scenario": [wte.seg_scenario[i] for i in wte.win_seg[widx]],
        "win_pos": wte.win_pos[widx], "y_true": wte.y_last(widx), "cpf_pred": pred})
    oof["driver_type"] = np.where(oof["subject_id"].str.upper().str.startswith("E"), "Expert", "Novice")
    for j, n in enumerate(names):
        oof[f"gate_{n}"] = gates[:, j]
    d = _job_dir(paths, f"fold{f_idx}")
    oof.to_csv(d / "oof.csv", index=False)
    pd.DataFrame(rows).to_csv(d / "metrics.csv", index=False)
    full.free()
    del ltr, liv, lte
    gc.collect()
    if USE_CUDA:
        torch.cuda.empty_cache()
    (d / "done.json").write_text(json.dumps({"version": VERSION, "hash": config_hash(),
                                             "seconds": time.time() - t_job}))
    log(f">>> FOLD {f_idx} finished in {(time.time() - t_job) / 60:.1f} min")


def run_final_job(meta: Dict, cfg: Dict, paths: Dict[str, Path]):
    t_job = time.time()
    tr = cfg["training"]
    sync = Path(meta["sync_dir"])
    subjects = meta["subjects"]
    train_subs, val_subs = stratified_split(subjects, tr["inner_val_frac"], tr["cv_seed"] + 99)
    log(f"\n>>> FINAL MODEL  train={train_subs}  early-stopping-val={val_subs}")
    schema = build_feature_schema(sync, subjects)
    names, groups = make_groups(schema)
    wtr = WindowData(sync, train_subs, cfg, schema, name="train")
    wiv = WindowData(sync, val_subs, cfg, schema, wtr.mean, wtr.std, name="val")
    ltr = WindowLoader(wtr, DEVICE, tr["batch_size"], shuffle=True, drop_last=len(wtr.starts) >= 2 * tr["batch_size"],
                       max_per_epoch=tr["max_train_windows_per_epoch"], seed=tr["cv_seed"] + 99)
    liv = WindowLoader(wiv, DEVICE, tr["eval_batch_size"], select=even_subsample(len(wiv.starts), tr["max_val_windows"]))
    set_seed(tr["cv_seed"] + 99)
    ck = Path(paths["checkpoints"]) / "final_best_model.pt"
    full = Trainer(cfg, "full", len(schema), groups, tag="final|full", ckpt_path=ck)
    full.fit(ltr, liv, wiv, tr["epochs"], tr["patience"])
    d = _job_dir(paths, "final")
    pd.DataFrame(full.history).to_csv(d / "history_final.csv", index=False)
    torch.save({"model_state": {k: v.cpu() for k, v in full.model.state_dict().items()},
                "feature_columns": schema, "mean": wtr.mean, "std": wtr.std, "groups": groups,
                "group_names": names, "config": cfg, "best_val_mse_last": full.best_val},
               Path(paths["checkpoints"]) / "final_bundle.pt")
    full.free()
    (d / "done.json").write_text(json.dumps({"version": VERSION, "hash": config_hash(),
                                             "seconds": time.time() - t_job}))
    log(f">>> FINAL MODEL finished in {(time.time() - t_job) / 60:.1f} min  (checkpoint: {ck})")


def execute_job(name: str, meta_path: str):
    meta = json.loads(Path(meta_path).read_text())
    paths = make_paths(meta["out_root"])
    cfg = copy.deepcopy(CONFIG)
    set_seed(GLOBAL_SEED)
    if name == "final":
        run_final_job(meta, cfg, paths)
    else:
        run_fold_job(int(name.replace("fold", "")), meta, cfg, paths)


# =============================================================================
# 11. JOB SCHEDULER (one process per GPU, or sequential in-process)
# =============================================================================
def job_done(paths: Dict[str, Path], name: str) -> bool:
    f = Path(paths["jobs"]) / name / "done.json"
    try:
        j = json.loads(f.read_text())
        return j.get("hash") == config_hash() and j.get("version") == VERSION
    except Exception:
        return False


def _script_path() -> Optional[str]:
    try:
        p = Path(__file__).resolve()
        return str(p) if p.exists() else None
    except NameError:
        return None


def run_jobs(jobs: List[str], meta_path: str, paths: Dict[str, Path], t0: float):
    pending = [j for j in jobs if not job_done(paths, j)]
    for j in jobs:
        if j not in pending:
            log(f"[RESUME] {j} already finished with identical config -> skipped")
    if not pending:
        return
    n_gpu = torch.cuda.device_count()
    n_workers = FORCE_WORKERS or (n_gpu if n_gpu >= 2 else 1)
    script = _script_path()
    budget = TIME_BUDGET_HOURS * 3600

    def over_budget() -> bool:
        return (time.time() - t0) > budget

    failed: List[str] = []
    if n_workers >= 2 and script and not NO_PARALLEL:
        log(f"[SCHED] {len(pending)} jobs on {n_workers} parallel workers (one per GPU): {pending}")
        running: Dict[int, Tuple[subprocess.Popen, str, Path, Any]] = {}
        last_print = 0.0
        while pending or running:
            for slot in range(n_workers):
                if slot not in running and pending:
                    if over_budget():
                        log(f"[BUDGET] {TIME_BUDGET_HOURS}h reached -> not starting: {pending}")
                        pending = []
                        break
                    name = pending.pop(0)
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(slot) if n_gpu >= 2 else ""
                    env["ACPF_WORKER"] = "1"
                    lf = Path(paths["logs"]) / f"{name}.log"
                    fh = open(lf, "w")
                    pr = subprocess.Popen([sys.executable, script, "--job", name, "--meta", meta_path],
                                          stdout=fh, stderr=subprocess.STDOUT, env=env)
                    running[slot] = (pr, name, lf, fh)
                    log(f"[SCHED] started {name} on GPU {slot}  (log: {lf})")
            time.sleep(5)
            for slot, (pr, name, lf, fh) in list(running.items()):
                rc = pr.poll()
                if rc is not None:
                    fh.close()
                    del running[slot]
                    if rc == 0 and job_done(paths, name):
                        log(f"[SCHED] {name} finished OK  (elapsed {(time.time() - t0) / 60:.1f} min)")
                    else:
                        tail = lf.read_text().strip().splitlines()[-12:]
                        log(f"[SCHED] {name} FAILED (rc={rc}); last log lines:\n    " + "\n    ".join(tail))
                        failed.append(name)
            if time.time() - last_print > 90 and running:
                last_print = time.time()
                for slot, (pr, name, lf, fh) in running.items():
                    lines = [l for l in lf.read_text().splitlines() if l.strip()]
                    log(f"[PROGRESS] GPU{slot} {name}: {lines[-1] if lines else '(starting)'}")
    else:
        log(f"[SCHED] sequential in-process run of {pending}")
        for name in pending:
            if over_budget():
                log(f"[BUDGET] {TIME_BUDGET_HOURS}h reached -> skipping {name}")
                continue
            try:
                execute_job(name, meta_path)
            except Exception as e:
                import traceback
                traceback.print_exc()
                failed.append(name)
    for name in failed:      # one retry, in-process, so a transient worker problem cannot ruin the run
        if over_budget():
            break
        log(f"[RETRY] {name} in-process")
        try:
            execute_job(name, meta_path)
        except Exception as e:
            import traceback
            traceback.print_exc()
            log(f"[ERROR] {name} failed twice: {type(e).__name__}: {e}")


# =============================================================================
# 12. ANALYSIS & FIGURES (out-of-fold predictions only)
# =============================================================================
def make_figures_and_tables(oof: pd.DataFrame, comp: pd.DataFrame, paths: Dict[str, Path], tag: str):
    fig_dir, res_dir = Path(paths["figures"]), Path(paths["results"])
    E, N = "#1f77b4", "#d62728"
    sfx = " [SYNTHETIC DEBUG DATA]" if tag else ""

    summ = comp.groupby("model")[["rmse", "mae", "r2", "r2_segment"]].agg(["mean", "std"]).round(4)
    summ.to_csv(res_dir / "benchmark_summary.csv")
    comp.to_csv(res_dir / "model_comparison_by_fold.csv", index=False)
    log("\n" + "=" * 90 + "\n  CROSS-VALIDATION SUMMARY (mean / std over folds, TEST subjects never seen in training)\n" + "=" * 90)
    log(summ.to_string())

    oof = oof.sort_values(["subject_id", "fold"]).copy()
    oof.to_csv(res_dir / "oof_predictions.csv", index=False)
    seg = oof.groupby(["subject_id", "driver_type", "scenario"])[["cpf_pred", "y_true"]].mean().reset_index()
    subj = oof.groupby(["subject_id", "driver_type"])["cpf_pred"].mean().reset_index()
    subj_true = oof.groupby(["subject_id"])["y_true"].mean().rename("label_mean").reset_index()
    subj = subj.merge(subj_true, on="subject_id")
    ex, no = subj[subj.driver_type == "Expert"], subj[subj.driver_type == "Novice"]
    stats: Dict[str, Any] = {"n_expert": len(ex), "n_novice": len(no)}
    if len(ex) >= 2 and len(no) >= 2:
        u, p = mannwhitneyu(ex["cpf_pred"], no["cpf_pred"], alternative="two-sided")
        stats.update({"mannwhitney_U": float(u), "p_value_subject_level": float(p),
                      "mean_cpf_expert": float(ex.cpf_pred.mean()), "mean_cpf_novice": float(no.cpf_pred.mean())})
        pl, pt = mannwhitneyu(ex["label_mean"], no["label_mean"], alternative="two-sided")
        stats["p_value_questionnaire_labels_subject_level"] = float(pt)
    (res_dir / "expert_vs_novice_stats.json").write_text(json.dumps(stats, indent=2))
    log(f"[STATS] subject-level expert vs novice: {stats}")
    gate_cols = [c for c in oof.columns if c.startswith("gate_")]
    if gate_cols:
        oof.groupby("driver_type")[gate_cols].mean().round(4).to_csv(res_dir / "mean_modality_gates.csv")

    # Fig 1: trajectories
    oof["t"] = oof.groupby("subject_id").cumcount()
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=False)
    for ax, grp, col, lbl in zip(axes, ("Expert", "Novice"), (E, N), ("Expert", "Novice")):
        for sid, d in oof[oof.driver_type == grp].groupby("subject_id"):
            ax.plot(d["cpf_pred"].rolling(25, min_periods=1).mean().values, color=col, alpha=0.35, lw=0.8)
        ax.set_ylim(0, 1); ax.set_ylabel("Predicted cognitive pressure"); ax.grid(alpha=0.3)
        ax.set_title(f"{lbl} drivers - out-of-fold CPF trajectories{sfx}", fontweight="bold")
    axes[1].set_xlabel("Window index (scenario order)")
    plt.tight_layout(); fig.savefig(fig_dir / "cpf_timeseries_expert_vs_novice.png", dpi=150); plt.close(fig)

    # Fig 2: distributions
    fig, ax = plt.subplots(figsize=(9, 5))
    for grp, col in (("Expert", E), ("Novice", N)):
        ax.hist(oof[oof.driver_type == grp]["cpf_pred"], bins=40, density=True, alpha=0.45, color=col, label=grp)
    ax.set_xlabel("Predicted CPF"); ax.set_ylabel("Density"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"CPF distribution (out-of-fold windows){sfx}", fontweight="bold")
    fig.savefig(fig_dir / "cpf_distribution_expert_vs_novice.png", dpi=150); plt.close(fig)

    # Fig 3: subject-level boxplot + MWU
    fig, ax = plt.subplots(figsize=(6, 6))
    data = [ex["cpf_pred"].values, no["cpf_pred"].values]
    bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="black", lw=2))
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Expert", "Novice"])
    for patch, c in zip(bp["boxes"], (E, N)):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    for i, d in enumerate(data, 1):
        ax.scatter(np.full(len(d), i) + np.random.RandomState(i).uniform(-0.08, 0.08, len(d)), d, color="k", s=14, zorder=3)
    if "p_value_subject_level" in stats:
        p = stats["p_value_subject_level"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.set_title(f"Subject-level Mann-Whitney U: p={p:.3g} ({sig}){sfx}", fontweight="bold", fontsize=10)
    ax.set_ylabel("Mean predicted CPF per subject"); ax.grid(alpha=0.3, axis="y")
    fig.savefig(fig_dir / "cpf_boxplot_significance.png", dpi=150); plt.close(fig)

    # Fig 4+6: scenario heatmap and bars
    seg["k"] = seg["scenario"].map(lambda s: scen_key(s) if scen_key(s) is not None else 99)
    piv = seg.groupby(["scenario", "driver_type"])["cpf_pred"].mean().unstack()
    order = sorted(piv.index, key=lambda s: (scen_key(s) if scen_key(s) is not None else 99, s))
    piv = piv.loc[order]
    cols = [c for c in ("Expert", "Novice") if c in piv.columns]
    if len(piv) > 1 and cols:
        fig, ax = plt.subplots(figsize=(7, max(4, len(piv) * 0.4)))
        im = ax.imshow(piv[cols].values, cmap="coolwarm", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
        ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=9)
        plt.colorbar(im, ax=ax, label="Mean predicted CPF"); ax.set_title(f"Scenario-level CPF{sfx}", fontweight="bold")
        plt.tight_layout(); fig.savefig(fig_dir / "cpf_scenario_heatmap.png", dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(12, 5))
        idx, w = np.arange(len(piv)), 0.38
        for j, (c, col) in enumerate(zip(cols, (E, N))):
            ax.bar(idx + (j - (len(cols) - 1) / 2) * w, piv[c].values, width=w, color=col, alpha=0.85, label=c)
        ax.set_xticks(idx); ax.set_xticklabels(piv.index, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Mean predicted CPF"); ax.legend(); ax.grid(alpha=0.3, axis="y")
        ax.set_title(f"CPF by driving scenario{sfx}", fontweight="bold")
        plt.tight_layout(); fig.savefig(fig_dir / "cpf_scenario_barchart.png", dpi=150); plt.close(fig)

    # Fig 5: per-subject grid
    subs = sorted(oof["subject_id"].unique())
    rows = (len(subs) + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(16, rows * 2.4), sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, sid in zip(axes, subs):
        d = oof[oof.subject_id == sid]
        ax.plot(d["cpf_pred"].values, color=E if sid.upper().startswith("E") else N, lw=1)
        ax.plot(d["y_true"].values, color="k", lw=0.8, alpha=0.6, ls="--")
        ax.set_title(sid, fontsize=9, fontweight="bold"); ax.set_ylim(0, 1); ax.grid(alpha=0.2)
    for ax in axes[len(subs):]:
        ax.set_visible(False)
    plt.suptitle(f"Per-subject out-of-fold prediction (coloured) vs questionnaire target (dashed){sfx}", fontsize=10)
    plt.tight_layout(); fig.savefig(fig_dir / "cpf_per_subject_trajectories.png", dpi=150); plt.close(fig)

    # Fig 7: segment-level predicted vs true
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(seg["y_true"], seg["cpf_pred"], c=np.where(seg.driver_type == "Expert", E, N), s=22, alpha=0.75)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    r2 = regression_metrics(seg["y_true"].values, seg["cpf_pred"].values)["r2"]
    ax.set_xlabel("Questionnaire target (scenario level)"); ax.set_ylabel("Predicted CPF (segment mean)")
    ax.set_title(f"Out-of-fold, per (subject, scenario)  R2={r2:.3f}{sfx}", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(fig_dir / "cpf_pred_vs_target_segments.png", dpi=150); plt.close(fig)

    # Fig 8: model comparison
    g = comp.groupby("model")["rmse"].agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(range(len(g)), g["mean"], xerr=g["std"].fillna(0), color="#4c72b0", alpha=0.85)
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g.index, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel("RMSE on unseen test subjects (mean +/- std over folds)"); ax.grid(alpha=0.3, axis="x")
    ax.set_title(f"Model comparison{sfx}", fontweight="bold")
    plt.tight_layout(); fig.savefig(fig_dir / "model_comparison_rmse.png", dpi=150); plt.close(fig)


def plot_histories(paths: Dict[str, Path]):
    hist = Path(paths["jobs"]) / "final" / "history_final.csv"
    if not hist.exists():
        return
    h = pd.read_csv(hist)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(h["epoch"], h["train_loss"], label="train (combined loss)")
    axes[0].plot(h["epoch"], h["val_mse_last"], label="val MSE (CPF, last step)")
    axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_title("Final model loss")
    axes[1].semilogy(h["epoch"], h["lr"], color="green"); axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3); axes[1].set_title("Learning rate (warm-up + cosine)")
    plt.tight_layout(); fig.savefig(Path(paths["figures"]) / "training_curves_final.png", dpi=150); plt.close(fig)


# =============================================================================
# 13. MASTER PIPELINE
# =============================================================================
def run_pipeline():
    t0 = time.time()
    log("=" * 90)
    log(f"  A-CPF + KOOPMAN + DFCG  v{VERSION}   |   smoke={SMOKE}")
    if USE_CUDA:
        log(f"  GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}   torch {torch.__version__}")
    else:
        log(f"  NO GPU detected (CPU mode, torch {torch.__version__}). On Kaggle enable Accelerator: GPU T4 x2.")
    log("=" * 90)

    raw, synth = find_dataset_path()
    base = "/kaggle/working/outputs" if IN_KAGGLE else "./outputs"
    out_root = base + ("_SYNTHETIC_DEBUG" if synth else "") + ("_SMOKE" if SMOKE else "")
    paths = make_paths(out_root)
    log(f"[OUT] {out_root}")

    labels = load_scale_labels(raw)
    stamp = Path(paths["synchronized"]) / ".sync_version"
    subjects_cfg = CONFIG["data"]["subjects"]
    have = all((Path(paths["synchronized"]) / f"{s}.parquet").exists() or
               (Path(paths["synchronized"]) / f"{s}.csv").exists() for s in subjects_cfg)
    if have and stamp.exists() and stamp.read_text() == f"{VERSION}|{raw}":
        log("[SYNC] reusing synchronized files from a previous run of this version")
    else:
        log("\n[SYNC] synchronising modalities ...")
        syncer = DataSynchronizer(raw, Path(paths["synchronized"]), labels, CONFIG)
        for sid in subjects_cfg:
            syncer.synchronize_subject(sid)
        stamp.write_text(f"{VERSION}|{raw}")

    usable = []
    for s in subjects_cfg:
        p = Path(paths["synchronized"]) / f"{s}.parquet"
        if p.exists() or p.with_suffix(".csv").exists():
            df = read_subject_cached(Path(paths["synchronized"]), s)
            if df[LABEL_COLS].notna().any(axis=1).mean() > 0.2:
                usable.append(s)
            else:
                log(f"  [WARN] {s}: almost no matching questionnaire labels -> excluded")
    n_e = sum(s.upper().startswith("E") for s in usable)
    n_n = len(usable) - n_e
    if len(usable) < 6 or n_e < 2 or n_n < 2:
        sample_ids = sorted({v for t in labels.values() for v in t["subject_id"].unique()})[:8]
        raise RuntimeError(
            f"Only {len(usable)} usable subjects ({n_e} expert / {n_n} novice) have labels. "
            f"Label-file subject IDs look like {sample_ids}; they must map to E01.. / N01.. IDs.")
    n_folds = max(2, min(CONFIG["training"]["n_folds"], len(usable) // 2))
    log(f"\n[SUBJECTS] usable={len(usable)} (expert {n_e} / novice {n_n}); {n_folds}-fold subject-level CV")

    meta = {"subjects": usable, "n_folds": n_folds, "sync_dir": str(paths["synchronized"]),
            "out_root": out_root, "synthetic": synth}
    meta_path = str(Path(paths["root"]) / "meta.json")
    Path(meta_path).write_text(json.dumps(meta, indent=1))
    jobs = (["final"] if RUN_FINAL_MODEL else []) + [f"fold{i}" for i in range(1, n_folds + 1)]
    run_jobs(jobs, meta_path, paths, t0)

    # ---- merge
    comp_parts, oof_parts, secs = [], [], {}
    for i in range(1, n_folds + 1):
        d = Path(paths["jobs"]) / f"fold{i}"
        if job_done(paths, f"fold{i}") and (d / "metrics.csv").exists():
            comp_parts.append(pd.read_csv(d / "metrics.csv"))
            oof_parts.append(pd.read_csv(d / "oof.csv"))
            secs[f"fold{i}"] = json.loads((d / "done.json").read_text())["seconds"]
    if not comp_parts:
        raise RuntimeError("No cross-validation fold finished. See logs in " + str(paths["logs"]))
    comp, oof = pd.concat(comp_parts, ignore_index=True), pd.concat(oof_parts, ignore_index=True)
    if len(comp_parts) < n_folds:
        log(f"[WARN] only {len(comp_parts)}/{n_folds} folds finished (time budget or failure).")
    make_figures_and_tables(oof, comp, paths, "SYNTH" if synth else "")
    plot_histories(paths)
    if job_done(paths, "final"):
        secs["final"] = json.loads((Path(paths["jobs"]) / "final" / "done.json").read_text())["seconds"]
    rep = {"version": VERSION, "synthetic_data": synth, "total_minutes": (time.time() - t0) / 60,
           "job_seconds": secs, "subjects": usable, "folds_finished": len(comp_parts)}
    (Path(paths["results"]) / "run_report.json").write_text(json.dumps(rep, indent=2))
    log("\n" + "=" * 90)
    log(f"  PIPELINE COMPLETE in {(time.time() - t0) / 60:.1f} min")
    log(f"  results : {paths['results']}\n  figures : {paths['figures']}\n  model   : {paths['checkpoints']}")
    log("=" * 90)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default=None)
    ap.add_argument("--meta", default=None)
    args, _unknown = ap.parse_known_args()
    if args.job:
        execute_job(args.job, args.meta)
    else:
        run_pipeline()  
