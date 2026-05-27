"""
pipeline.py  —  Unified CODE_LLM Pipeline  (MBPP+, Qwen2.5-Coder)
==================================================================
CHANGES vs previous version
-----------------------------
  1. ACTS_ROOTS updated to EMNLP paths.
  2. Activation-fetching loop made robust for run1/run2/… naming under
     <problem>/right/ and <problem>/wrong/ (no nested 'all/' assumed).
  3. Batched steering generation (STEER_BATCH_SIZE, default 4) – runs
     multiple prompts through the model simultaneously for ~4× throughput.
  4. Multi-GPU + Flash-Attention-2 model loading (device_map="auto",
     attn_implementation="flash_attention_2" when available).
  5. Delta baseline for steering is the OVERALL test-set baseline pass@1
     (not the pool-specific right/wrong baseline), so both forward and
     backward deltas are comparable on the same denominator.
  6. Layer ranking (and steering target selection) is by AUROC_val only,
     not by l*-score.  l* is still computed and logged but not used for
     layer selection.

Stages (run sequentially for each model):

  1. SPLIT        — Load pre-generated runs from acts root, apply 60/20/20
                    problem-level split (train / val / test).
  2. PROBING      — Layer-wise logistic probe trained on contrastive-TRAIN,
                    evaluated by AUROC on contrastive-VAL.
                    Top-5 layers by AUROC_val carry forward.
  3. SYMMETRY + L*-SCORE
                  — Computed for reference; top-5 for STEERING are chosen
                    by AUROC_val (not l*-score).
  4. STEERING     — Three intervention types on top-5 AUROC layers (TEST):
                      (a) Forward  Addition    h←h+α·d̂  α∈{0.5,1,2,5,10}
                          on WRONG runs from TEST.
                      (b) Backward Subtraction h←h+α·d̂ α∈{-0.5,…,-10}
                          on RIGHT runs from TEST.
                      (c) Direction Ablation   h←h-(h·d̂)·d̂
                          on RIGHT runs from TEST.
                    Δp@1 computed against overall TEST baseline (all problems),
                    not pool-specific baseline.
                    Q-1.5B: 5 steering runs/problem; Q-7B: 3.
                    Generation is batched (STEER_BATCH_SIZE).

Paths
-----
  Acts root (pre-generated):
    1.5B: /media/kpdubey/8.0 TB Volume/Shubham/MI/EMNLP/
                qwen-2.5-coder-1.5b-instruct/full_dataset_runs
    7B  : same structure under qwen-2.5-coder-7b-instruct/

  Run layout expected:
    <acts_root>/<problem_id>/right/run1/layer_01.h5
    <acts_root>/<problem_id>/wrong/run2/layer_01.h5
    … etc.

  All outputs written under OUT_ROOT (configurable).

Usage
-----
  python pipeline.py                          # run both models
  python pipeline.py --model qwen-coder-1.5b-instruct
  python pipeline.py --model qwen-coder-7b-instruct
  python pipeline.py --out-root /my/output/dir
  python pipeline.py --batch-size 8          # override steer batch size
"""

import argparse
import csv
import gc
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_RATIO    = 0.60
VAL_RATIO      = 0.20
SPLIT_SEED     = 42
TEMPERATURE    = 0.8
TOP_P          = 0.95
MAX_NEW_TOKENS = 512
EXEC_TIMEOUT   = 10

EXPECTED_RUNS_PER_PROBLEM = 5

# Probing
N_BOOTSTRAP  = 10
TOP_N_PROBE  = 5
GAP_THRESH   = 0.15
RNG_SEED     = 42

# Layer selection  — ranked by AUROC_val (l* still computed for reference)
TOP_K_STEER  = 5
W1, W2       = 0.6, 0.4      # l* weights (reference only)

# Steering alphas
ALPHAS_FWD   = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
ALPHAS_BWD   = [-0.5, -1.0, -2.0, -5.0, -10.0, -20.0, -50.0]
ALPHA_ABLATE = 1.0

# Batched generation for steering (≈4× GPU utilisation)
STEER_BATCH_SIZE = 128

N_STEER_RUNS = {
    "qwen-coder-1.5b-instruct": 3,
    "qwen-coder-7b-instruct":   3,
}

# ── BigCodeBench run paths ───────────────────────────────────────────────────
ACTS_ROOTS = {
    "qwen-coder-1.5b-instruct": (
        "/media/kpdubey/8.0 TB Volume/Shubham/MI/EMNLP/"
        "qwen-2.5-coder-1.5b-instruct/full_dataset_runs"
    ),
    "qwen-coder-7b-instruct": (
        "/media/kpdubey/8.0 TB Volume/Shubham/MI/EMNLP/"
        "qwen-2.5-coder-7b-instruct/full_dataset_runs"
    ),
}

MODEL_REGISTRY = {
    "qwen-coder-1.5b-instruct": {
        "hf_id":      "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "display":    "Qwen2.5-Coder-1.5B",
        "system_msg": "You are an expert Python programmer.",
        "chat":       True,
    },
    "qwen-coder-7b-instruct": {
        "hf_id":      "Qwen/Qwen2.5-Coder-7B-Instruct",
        "display":    "Qwen2.5-Coder-7B",
        "system_msg": "You are an expert Python programmer.",
        "chat":       True,
    },
}

# Directories inside acts_root that are NOT problem folders
SKIP_DIRS = {
    "probing", "steering", "plots", "pipeline2_results",
    "pipeline2_probing", "all", "contrastive", "comparison",
}

OUT_ROOT = Path(__file__).parent / "pipeline_results"


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADERS  (MBPP+ and BigCodeBench)
# ─────────────────────────────────────────────────────────────────────────────

def load_bigcodebench() -> list[dict]:
    """Load BigCodeBench full set (1 140 tasks, unittest-based)."""
    from datasets import load_dataset as hf_load
    print("  Loading bigcode/bigcodebench (full set, v0.1.4) …")
    ds = hf_load("bigcode/bigcodebench", split="v0.1.4")
    rows: list[dict] = []
    for item in ds:
        prompt    = item.get("complete_prompt", "")
        test_code = item.get("test", "")
        entry_pt  = item.get("entry_point", "task_func")
        test_runner = (
            "import unittest as __ut, io as __io, sys as __sys\n"
            f"{test_code}\n"
            "__suite  = __ut.TestLoader().loadTestsFromTestCase(TestCases)\n"
            "__result = __ut.TextTestRunner(stream=__io.StringIO(), verbosity=0)"
            ".run(__suite)\n"
            "if not __result.wasSuccessful():\n"
            "    __sys.exit(1)\n"
        )
        rows.append({
            "task_id":            item.get("task_id", f"BigCodeBench/{len(rows)}"),
            "prompt":             prompt,
            "entry_point":        entry_pt,
            "test":               None,
            "test_list":          [test_runner],
            "execution_mode":     "function",
            "canonical_solution": item.get("canonical_solution", ""),
        })
    print(f"  [DATA] Loaded {len(rows)} BigCodeBench problems.")
    return rows


def load_mbppplus() -> list[dict]:
    # 1. evalplus Python package (best coverage + plus test cases)
    try:
        from evalplus.data import get_mbpp_plus
        problems = get_mbpp_plus()
        return [{"task_id": k, **v} for k, v in problems.items()]
    except ImportError:
        pass

    from datasets import load_dataset

    # 2. Original MBPP full/test — covers exactly the stored dir IDs (11-510).
    #    This is the correct dataset when activations were collected from MBPP full.
    try:
        ds = load_dataset(
            "google-research-datasets/mbpp", "full", split="test",
            trust_remote_code=True,
        )
        problems = [dict(row) for row in ds]
        print(f"  [DATA] Loaded original MBPP full/test: {len(problems)} problems "
              f"(IDs {min(int(p['task_id']) for p in problems)}"
              f"-{max(int(p['task_id']) for p in problems)})")
        return problems
    except Exception as e:
        print(f"  [DATA] MBPP full/test unavailable: {e}")

    # 3. evalplus/mbppplus on HuggingFace (only ~45% overlap with stored dirs)
    try:
        ds = load_dataset("evalplus/mbppplus", split="test", trust_remote_code=True)
        problems = [dict(row) for row in ds]
        print(f"  [DATA] Loaded evalplus/mbppplus HF: {len(problems)} problems "
              f"(partial coverage — prefer MBPP full/test)")
        return problems
    except Exception as e:
        raise RuntimeError(
            f"Cannot load MBPP. Install evalplus or the datasets library.\n{e}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# H5 HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_h5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as hf:
        return hf["activation"][:].astype(np.float32)


def _safe_id(tid) -> str:
    return str(tid).replace("/", "_")


def _find_acts_root(acts_dir: str | Path) -> Path:
    """
    Return the directory that directly contains problem folders.
    If an 'all/' subfolder exists, use that; otherwise use acts_dir itself.
    """
    root = Path(acts_dir)
    if (root / "all").is_dir():
        return root / "all"
    return root


# ─────────────────────────────────────────────────────────────────────────────
# RUN-DIR SCANNING  (robust for run1/run2/… naming)
# ─────────────────────────────────────────────────────────────────────────────

def _iter_run_dirs(verdict_dir: Path) -> list[Path]:
    """
    Return all run sub-directories inside a verdict folder (right/ or wrong/).
    Handles any naming convention: run1, run_1, run_00, 0, 1, etc.
    Sorted deterministically.
    """
    if not verdict_dir.is_dir():
        return []
    return sorted(
        [p for p in verdict_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )


def detect_n_layers(acts_root: Path) -> int:
    """Detect number of layers by inspecting the first run directory found."""
    for prob_dir in sorted(acts_root.iterdir()):
        if not prob_dir.is_dir() or prob_dir.name in SKIP_DIRS:
            continue
        for verdict in ("right", "wrong"):
            for run_dir in _iter_run_dirs(prob_dir / verdict):
                h5s = sorted(run_dir.glob("layer_*.h5"))
                if h5s:
                    return len(h5s)
    raise RuntimeError(f"No layer H5 files found under {acts_root}")


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRITY VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_run_integrity(acts_root: Path, n_layers: int) -> dict:
    missing_runs   = []
    missing_h5s    = []
    total_problems = 0
    total_runs     = 0

    for prob_dir in sorted(acts_root.iterdir()):
        if not prob_dir.is_dir() or prob_dir.name in SKIP_DIRS:
            continue
        total_problems += 1
        n_right = 0
        n_wrong = 0

        for verdict in ("right", "wrong"):
            for run_dir in _iter_run_dirs(prob_dir / verdict):
                h5s = sorted(run_dir.glob("layer_*.h5"))
                if verdict == "right":
                    n_right += 1
                else:
                    n_wrong += 1
                total_runs += 1
                if len(h5s) != n_layers:
                    missing_h5s.append({
                        "problem":  prob_dir.name,
                        "verdict":  verdict,
                        "run_dir":  run_dir.name,
                        "found":    len(h5s),
                        "expected": n_layers,
                    })

        n_total = n_right + n_wrong
        if n_total != EXPECTED_RUNS_PER_PROBLEM:
            missing_runs.append({
                "problem": prob_dir.name,
                "n_right": n_right,
                "n_wrong": n_wrong,
                "n_total": n_total,
                "deficit": EXPECTED_RUNS_PER_PROBLEM - n_total,
            })

    return {
        "total_problems":          total_problems,
        "total_runs":              total_runs,
        "expected_total_runs":     total_problems * EXPECTED_RUNS_PER_PROBLEM,
        "n_problems_with_deficit": len(missing_runs),
        "n_runs_with_missing_h5":  len(missing_h5s),
        "problems_with_deficit":   missing_runs,
        "runs_with_missing_h5":    missing_h5s,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: SPLIT  (60 / 20 / 20)
# ─────────────────────────────────────────────────────────────────────────────

def scan_runs(acts_root: Path) -> dict[str, dict]:
    """
    Walk acts_root and collect per-problem run stats.
    Returns {task_id: {"n_right": int, "n_wrong": int, "n_total": int}}
    """
    problems: dict[str, dict] = {}
    for prob_dir in sorted(acts_root.iterdir()):
        if not prob_dir.is_dir() or prob_dir.name in SKIP_DIRS:
            continue
        tid = prob_dir.name
        n_r = len(_iter_run_dirs(prob_dir / "right"))
        n_w = len(_iter_run_dirs(prob_dir / "wrong"))
        problems[tid] = {
            "task_id": tid,
            "n_right": n_r,
            "n_wrong": n_w,
            "n_total": n_r + n_w,
        }
    return problems


def make_split(all_problem_ids: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Deterministic 60/20/20 split at problem level."""
    rng  = np.random.default_rng(SPLIT_SEED)
    ids  = sorted(all_problem_ids)
    perm = rng.permutation(len(ids))

    n     = len(ids)
    n_tr  = max(1, int(n * TRAIN_RATIO))
    n_val = max(1, int(n * VAL_RATIO))

    tr_idx  = set(perm[:n_tr].tolist())
    val_idx = set(perm[n_tr:n_tr + n_val].tolist())

    train = [ids[i] for i in range(n) if i in tr_idx]
    val   = [ids[i] for i in range(n) if i in val_idx]
    test  = [ids[i] for i in range(n) if i not in tr_idx and i not in val_idx]
    return train, val, test


def _problem_level_groups(
    ids: list[str],
    problems: dict[str, dict],
    N: int = EXPECTED_RUNS_PER_PROBLEM,
) -> dict[str, list[str]]:
    all_right   = []
    all_wrong   = []
    contrastive = []
    for tid in ids:
        p = problems[tid]
        n_r, n_tot = p["n_right"], p["n_total"]
        total = n_tot if n_tot > 0 else N
        if n_r == total:
            all_right.append(tid)
        elif n_r == 0:
            all_wrong.append(tid)
        else:
            contrastive.append(tid)
    return {
        "all_right_ids":   all_right,
        "all_wrong_ids":   all_wrong,
        "contrastive_ids": contrastive,
    }


def run_split(acts_dir: str, out_dir: Path) -> dict:
    split_path = out_dir / "split.json"
    if split_path.exists():
        print(f"  [SPLIT] Resuming — split.json found.")
        with open(split_path) as f:
            return json.load(f)

    acts_root = _find_acts_root(acts_dir)
    print(f"  [SPLIT] Scanning activations under {acts_root} …")

    n_layers_detected = detect_n_layers(acts_root)
    print(f"  [SPLIT] Detected {n_layers_detected} layers. Running integrity check …")
    integrity = validate_run_integrity(acts_root, n_layers_detected)

    if integrity["n_problems_with_deficit"] > 0:
        print(f"  [WARN] {integrity['n_problems_with_deficit']} problems have "
              f"fewer than {EXPECTED_RUNS_PER_PROBLEM} runs:")
        for rec in integrity["problems_with_deficit"][:20]:
            print(f"         {rec['problem']:30s}  "
                  f"right={rec['n_right']}  wrong={rec['n_wrong']}  "
                  f"total={rec['n_total']}  deficit={rec['deficit']}")
        if len(integrity["problems_with_deficit"]) > 20:
            print(f"         … and {len(integrity['problems_with_deficit'])-20} more.")
    else:
        print(f"  [SPLIT] ✓ All {integrity['total_problems']} problems have "
              f"exactly {EXPECTED_RUNS_PER_PROBLEM} runs each.")

    if integrity["n_runs_with_missing_h5"] > 0:
        print(f"  [WARN] {integrity['n_runs_with_missing_h5']} run dirs "
              f"have incomplete H5 files (expected {n_layers_detected}):")
        for rec in integrity["runs_with_missing_h5"][:10]:
            print(f"         {rec['problem']}/{rec['verdict']}/{rec['run_dir']}  "
                  f"found={rec['found']}  expected={rec['expected']}")
    else:
        print(f"  [SPLIT] ✓ All run dirs have complete H5 files "
              f"({n_layers_detected} layers each).")

    print(f"  [SPLIT] Total runs found: {integrity['total_runs']}  "
          f"(expected {integrity['expected_total_runs']})")

    problems = scan_runs(acts_root)
    all_ids  = sorted(problems.keys())
    n_total  = len(all_ids)
    print(f"  [SPLIT] Found {n_total} problems.")

    train_ids, val_ids, test_ids = make_split(all_ids)

    train_groups = _problem_level_groups(train_ids, problems)
    val_groups   = _problem_level_groups(val_ids,   problems)
    test_groups  = _problem_level_groups(test_ids,  problems)

    def _run_counts(ids):
        nr = sum(problems[t]["n_right"] for t in ids)
        nw = sum(problems[t]["n_wrong"] for t in ids)
        return nr, nw

    tr_nr,   tr_nw   = _run_counts(train_ids)
    val_nr,  val_nw  = _run_counts(val_ids)
    test_nr, test_nw = _run_counts(test_ids)

    test_pass_pool = [tid for tid in test_ids if problems[tid]["n_right"] > 0]
    test_fail_pool = [tid for tid in test_ids if problems[tid]["n_wrong"] > 0]

    split = {
        "acts_root":   str(acts_root),
        "train_ratio": TRAIN_RATIO,
        "val_ratio":   VAL_RATIO,
        "test_ratio":  round(1.0 - TRAIN_RATIO - VAL_RATIO, 4),
        "split_seed":  SPLIT_SEED,
        "n_total":     n_total,
        "integrity": {
            "total_runs":              integrity["total_runs"],
            "expected_total_runs":     integrity["expected_total_runs"],
            "n_problems_with_deficit": integrity["n_problems_with_deficit"],
            "n_runs_with_missing_h5":  integrity["n_runs_with_missing_h5"],
        },
        "train": {
            "n_problems":      len(train_ids),
            "all_ids":         train_ids,
            "all_right_ids":   train_groups["all_right_ids"],
            "all_wrong_ids":   train_groups["all_wrong_ids"],
            "contrastive_ids": train_groups["contrastive_ids"],
            "n_all_right":     len(train_groups["all_right_ids"]),
            "n_all_wrong":     len(train_groups["all_wrong_ids"]),
            "n_contrastive":   len(train_groups["contrastive_ids"]),
            "n_right_runs":    tr_nr,
            "n_wrong_runs":    tr_nw,
            "n_total_runs":    tr_nr + tr_nw,
        },
        "val": {
            "n_problems":      len(val_ids),
            "all_ids":         val_ids,
            "all_right_ids":   val_groups["all_right_ids"],
            "all_wrong_ids":   val_groups["all_wrong_ids"],
            "contrastive_ids": val_groups["contrastive_ids"],
            "n_all_right":     len(val_groups["all_right_ids"]),
            "n_all_wrong":     len(val_groups["all_wrong_ids"]),
            "n_contrastive":   len(val_groups["contrastive_ids"]),
            "n_right_runs":    val_nr,
            "n_wrong_runs":    val_nw,
            "n_total_runs":    val_nr + val_nw,
        },
        "test": {
            "n_problems":      len(test_ids),
            "all_ids":         test_ids,
            "all_right_ids":   test_groups["all_right_ids"],
            "all_wrong_ids":   test_groups["all_wrong_ids"],
            "contrastive_ids": test_groups["contrastive_ids"],
            "n_all_right":     len(test_groups["all_right_ids"]),
            "n_all_wrong":     len(test_groups["all_wrong_ids"]),
            "n_contrastive":   len(test_groups["contrastive_ids"]),
            "pass_pool_ids":   test_pass_pool,
            "fail_pool_ids":   test_fail_pool,
            "n_pass_pool":     len(test_pass_pool),
            "n_fail_pool":     len(test_fail_pool),
            "n_right_runs":    test_nr,
            "n_wrong_runs":    test_nw,
            "n_total_runs":    test_nr + test_nw,
        },
        "problems": {tid: problems[tid] for tid in all_ids},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)

    hdr = (f"\n  {'Split':<6}  {'Problems':>8}  "
           f"{'All-Right':>9}  {'All-Wrong':>9}  {'Contrastive':>11}  "
           f"{'R-runs':>7}  {'W-runs':>7}")
    print(hdr)
    print("  " + "-" * 72)

    def _row(label, ids, grp, nr, nw):
        return (f"  {label:<6}  {len(ids):>8}  "
                f"{len(grp['all_right_ids']):>9}  "
                f"{len(grp['all_wrong_ids']):>9}  "
                f"{len(grp['contrastive_ids']):>11}  "
                f"{nr:>7}  {nw:>7}")

    print(_row("Train", train_ids, train_groups, tr_nr, tr_nw))
    print(_row("Val",   val_ids,   val_groups,   val_nr, val_nw))
    print(_row("Test",  test_ids,  test_groups,  test_nr, test_nw))
    print("  " + "-" * 72)
    grand_nr = tr_nr + val_nr + test_nr
    grand_nw = tr_nw + val_nw + test_nw
    all_right_tot = sum(len(g["all_right_ids"]) for g in [train_groups, val_groups, test_groups])
    all_wrong_tot = sum(len(g["all_wrong_ids"]) for g in [train_groups, val_groups, test_groups])
    cont_tot      = sum(len(g["contrastive_ids"]) for g in [train_groups, val_groups, test_groups])
    print(f"  {'Total':<6}  {n_total:>8}  "
          f"{all_right_tot:>9}  {all_wrong_tot:>9}  {cont_tot:>11}  "
          f"{grand_nr:>7}  {grand_nw:>7}")

    print(f"\n  Notes:")
    print(f"    All-Right / All-Wrong / Contrastive are mutually exclusive "
          f"problem-level groups.")
    print(f"    TRAIN contrastive ({len(train_groups['contrastive_ids'])}) "
          f"→ direction estimation + probe training.")
    print(f"    VAL   contrastive ({len(val_groups['contrastive_ids'])}) "
          f"→ probe AUROC, Symmetry, l*-score.")
    print(f"    TEST  pass-pool   ({len(test_pass_pool)} problems, {test_nr} right runs) "
          f"→ backward subtraction + ablation.")
    print(f"    TEST  fail-pool   ({len(test_fail_pool)} problems, {test_nw} wrong runs) "
          f"→ forward addition.")
    print(f"\n  [SPLIT] → {split_path}")
    return split


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: PROBING
# ─────────────────────────────────────────────────────────────────────────────

def collect_acts_for_ids(
    acts_root: Path,
    task_ids:  set[str],
    layer_idx: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load layer_XX.h5 activations for the given task IDs.
    Iterates: acts_root / <problem> / (right|wrong) / <any_run_dir> / layer_XX.h5
    """
    fname      = f"layer_{layer_idx:02d}.h5"
    rights, wrongs = [], []

    for prob_dir in sorted(acts_root.iterdir()):
        if not prob_dir.is_dir() or prob_dir.name not in task_ids:
            continue
        for verdict, buf in (("right", rights), ("wrong", wrongs)):
            for run_dir in _iter_run_dirs(prob_dir / verdict):
                h5 = run_dir / fname
                if h5.exists():
                    try:
                        buf.append(_load_h5(h5))
                    except Exception as e:
                        tqdm.write(f"  [WARN] Failed to load {h5}: {e}")

    if not rights or not wrongs:
        return None, None
    return (np.stack(rights).astype(np.float32),
            np.stack(wrongs).astype(np.float32))


def compute_direction_bootstrapped(
    right: np.ndarray,
    wrong: np.ndarray,
    n_samples: int = N_BOOTSTRAP,
    seed: int = RNG_SEED,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    k   = min(len(right), len(wrong))
    if k < 2:
        diff = right.mean(0) - wrong.mean(0)
        norm = float(np.linalg.norm(diff))
        return ((diff / norm) if norm > 1e-9 else diff).astype(np.float32), norm

    dirs = []
    for _ in range(n_samples):
        ri = rng.choice(len(right), k, replace=False)
        wi = rng.choice(len(wrong), k, replace=False)
        dirs.append(right[ri].mean(0) - wrong[wi].mean(0))

    mean_d = np.stack(dirs).mean(0)
    norm   = float(np.linalg.norm(mean_d))
    return ((mean_d / norm) if norm > 1e-9 else mean_d).astype(np.float32), norm


def train_probe_auroc(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
) -> float | None:
    if len(np.unique(y_test)) < 2:
        return None
    scaler = StandardScaler()
    Xtr    = scaler.fit_transform(X_train)
    Xte    = scaler.transform(X_test)
    clf    = LogisticRegression(
        class_weight="balanced", max_iter=1000, C=1.0, solver="lbfgs"
    )
    clf.fit(Xtr, y_train)
    return float(roc_auc_score(y_test, clf.predict_proba(Xte)[:, 1]))


def run_probing(
    acts_dir: str,
    split:    dict,
    out_dir:  Path,
) -> tuple[list[dict], list[dict]]:
    """
    Stage 2: layer-wise probing.
    Top-5 layers selected by AUROC_val (descending).
    """
    probe_dir  = out_dir / "probing"
    done_flag  = probe_dir / "probe_analysis.json"

    if done_flag.exists():
        print("  [PROBE] Resuming — probe_analysis.json found.")
        with open(done_flag) as f:
            data = json.load(f)
        # Rank by AUROC_val
        top5 = sorted(
            [r for r in data["all_layers"] if r.get("auroc_val") is not None],
            key=lambda r: r["auroc_val"], reverse=True
        )[:TOP_N_PROBE]
        return data["all_layers"], top5

    probe_dir.mkdir(parents=True, exist_ok=True)
    acts_root = _find_acts_root(acts_dir)

    cont_train_ids = set(split["train"]["contrastive_ids"])
    cont_val_ids   = set(split["val"]["contrastive_ids"])

    print(f"  [PROBE] Contrastive train problems : {len(cont_train_ids)}")
    print(f"  [PROBE] Contrastive val   problems : {len(cont_val_ids)}")

    n_layers = detect_n_layers(acts_root)
    print(f"  [PROBE] Detected {n_layers} layers.")

    all_results: list[dict] = []

    for li in tqdm(range(n_layers), desc="  Probing", unit="layer"):
        lname = f"layer_{li:02d}"

        tr_r, tr_w = collect_acts_for_ids(acts_root, cont_train_ids, li)
        if tr_r is None or tr_w is None:
            all_results.append({
                "layer_idx": li, "layer_name": lname,
                "auroc_val": None, "n_right_train": 0, "n_wrong_train": 0,
            })
            continue

        val_r, val_w = collect_acts_for_ids(acts_root, cont_val_ids, li)
        auroc_val = None
        if val_r is not None and val_w is not None:
            X_tr  = np.vstack([tr_r, tr_w])
            y_tr  = np.concatenate([np.ones(len(tr_r)), np.zeros(len(tr_w))])
            X_val = np.vstack([val_r, val_w])
            y_val = np.concatenate([np.ones(len(val_r)), np.zeros(len(val_w))])
            try:
                auroc_val = train_probe_auroc(X_tr, y_tr, X_val, y_val)
            except Exception as exc:
                tqdm.write(f"  [WARN] {lname} probe AUROC failed: {exc}")

        direction, raw_norm = compute_direction_bootstrapped(tr_r, tr_w)

        all_results.append({
            "layer_idx":     li,
            "layer_name":    lname,
            "auroc_val":     round(auroc_val, 4) if auroc_val is not None else None,
            "n_right_train": int(len(tr_r)),
            "n_wrong_train": int(len(tr_w)),
            "n_right_val":   int(len(val_r)) if val_r is not None else 0,
            "n_wrong_val":   int(len(val_w)) if val_w is not None else 0,
            "raw_norm":      float(raw_norm),
            "_direction":    direction,
        })

    # ── Rank by AUROC_val ────────────────────────────────────────────────────
    valid = [r for r in all_results if r.get("auroc_val") is not None]
    top5  = sorted(valid, key=lambda r: r["auroc_val"], reverse=True)[:TOP_N_PROBE]
    top5  = sorted(top5,  key=lambda r: r["layer_idx"])   # keep layer order for display

    dir_dir = probe_dir / "directions"
    dir_dir.mkdir(exist_ok=True)
    for rec in top5:
        np.save(dir_dir / f"{rec['layer_name']}.npy", rec["_direction"])

    def _clean(r):
        return {k: v for k, v in r.items() if k != "_direction"}

    probe_data = {
        "acts_dir":               str(acts_dir),
        "n_cont_train_problems":  len(cont_train_ids),
        "n_cont_val_problems":    len(cont_val_ids),
        "layer_selection":        "auroc_val",
        "all_layers":             [_clean(r) for r in all_results],
        "top5_layers":            [_clean(r) for r in top5],
    }
    with open(done_flag, "w") as f:
        json.dump(probe_data, f, indent=2)

    print(f"\n  [PROBE] Top-{TOP_N_PROBE} layers by AUROC_val (contrastive-VAL):")
    print(f"  {'Layer':>10}  {'AUROC_val':>10}  {'nR_tr':>6}  {'nW_tr':>6}  "
          f"{'nR_val':>6}  {'nW_val':>6}")
    print("  " + "-" * 55)
    for r in sorted(top5, key=lambda x: -x["auroc_val"]):
        av = f"{r['auroc_val']:.4f}" if r["auroc_val"] is not None else "  N/A"
        print(f"  {r['layer_name']:>10}  {av:>10}  "
              f"{r['n_right_train']:>6}  {r['n_wrong_train']:>6}  "
              f"{r.get('n_right_val',0):>6}  {r.get('n_wrong_val',0):>6}")
    print(f"  [PROBE] → {done_flag}")

    return [_clean(r) for r in all_results], [_clean(r) for r in top5]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: SYMMETRY SCORE + L*-SCORE  (reference only — not used for selection)
# ─────────────────────────────────────────────────────────────────────────────

def _collect_side(
    acts_root: Path,
    task_ids:  list[str],
    layer_idx: int,
    side:      str,
) -> list[np.ndarray]:
    fname = f"layer_{layer_idx:02d}.h5"
    acts  = []
    for tid in sorted(task_ids):
        prob_dir = acts_root / tid
        for run_dir in _iter_run_dirs(prob_dir / side):
            h5 = run_dir / fname
            if h5.exists():
                try:
                    acts.append(_load_h5(h5).astype(np.float64))
                except Exception:
                    pass
    return acts


def compute_symmetry_for_layer(
    acts_root: Path,
    hard_ids:  list[str],
    easy_ids:  list[str],
    layer_idx: int,
) -> float | None:
    right_hard = _collect_side(acts_root, hard_ids, layer_idx, "right")
    wrong_hard = _collect_side(acts_root, hard_ids, layer_idx, "wrong")
    right_easy = _collect_side(acts_root, easy_ids, layer_idx, "right")
    wrong_easy = _collect_side(acts_root, easy_ids, layer_idx, "wrong")

    if not right_hard or not wrong_hard or not right_easy or not wrong_easy:
        return None

    V_rec   = np.mean(np.stack(right_hard), 0) - np.mean(np.stack(wrong_hard), 0)
    V_cor   = np.mean(np.stack(wrong_easy), 0) - np.mean(np.stack(right_easy), 0)
    V_rec_n = V_rec / (np.linalg.norm(V_rec) + 1e-12)
    V_cor_n = V_cor / (np.linalg.norm(V_cor) + 1e-12)
    return float(np.dot(V_rec_n, -V_cor_n))


def compute_lstar_score(auroc_val: float, sym: float) -> float:
    return W1 * auroc_val + W2 * (sym + 1.0) / 2.0


def run_symmetry_and_lstar(
    acts_dir:   str,
    split:      dict,
    probe_top5: list[dict],
    out_dir:    Path,
) -> list[dict]:
    """
    Stage 3: Sym and l*-score for each top-5 probe layer (reference only).
    Steering target layers are STILL selected by AUROC_val (probe_top5 order).
    """
    sym_dir   = out_dir / "symmetry"
    done_flag = sym_dir / "symmetry_results.json"

    if done_flag.exists():
        print("  [SYM] Resuming — symmetry_results.json found.")
        with open(done_flag) as f:
            data = json.load(f)
        # Return top layers ranked by AUROC_val (not l*)
        return sorted(
            data["all_top5"],
            key=lambda r: r.get("auroc_val") or 0.0,
            reverse=True,
        )[:TOP_K_STEER]

    sym_dir.mkdir(parents=True, exist_ok=True)
    acts_root = _find_acts_root(acts_dir)

    problems_meta = split.get("problems", {})
    cont_val_ids  = split["val"]["contrastive_ids"]

    hard_ids, easy_ids = [], []
    for tid in cont_val_ids:
        p = problems_meta.get(tid, {})
        n = p.get("n_right", 0)
        if n in (1, 2):
            hard_ids.append(tid)
        elif n in (3, 4):
            easy_ids.append(tid)

    print(f"  [SYM] Using contrastive-VAL problems for Hard/Easy grouping.")
    print(f"  [SYM] Hard (n_pass∈{{1,2}}): {len(hard_ids)} problems")
    print(f"  [SYM] Easy (n_pass∈{{3,4}}): {len(easy_ids)} problems")
    print(f"  [SYM] NOTE: layer selection for steering uses AUROC_val, not l*.")

    sym_rows = []
    for rec in tqdm(probe_top5, desc="  Symmetry", unit="layer"):
        li    = rec["layer_idx"]
        lname = rec["layer_name"]
        auc   = rec.get("auroc_val") or 0.0

        sym = compute_symmetry_for_layer(acts_root, hard_ids, easy_ids, li)
        if sym is None:
            print(f"  [SYM] {lname}: insufficient activations — skipping sym.")
            sym   = float("nan")
            lstar = auc  # fall back to AUROC
        else:
            lstar = compute_lstar_score(auc, sym)

        sym_rows.append({
            "layer_idx":   li,
            "layer_name":  lname,
            "auroc_val":   auc,
            "sym_score":   round(sym, 4) if not np.isnan(sym) else None,
            "lstar_score": round(lstar, 4),
        })

    # ── Steering targets: top-K by AUROC_val ────────────────────────────────
    top_layers_by_auroc = sorted(
        sym_rows, key=lambda r: r.get("auroc_val") or 0.0, reverse=True
    )[:TOP_K_STEER]

    data = {
        "n_cont_val_problems": len(cont_val_ids),
        "n_hard_problems":     len(hard_ids),
        "n_easy_problems":     len(easy_ids),
        "layer_selection":     "auroc_val",
        "all_top5":            sym_rows,
        "top_layers_auroc":    top_layers_by_auroc,
        # l*-ranked kept for reference
        "top_layers_lstar":    sorted(sym_rows, key=lambda r: r["lstar_score"], reverse=True)[:TOP_K_STEER],
    }
    with open(done_flag, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n  [SYM] Layer metrics (★ = selected for steering by AUROC_val):")
    print(f"  {'Layer':>10}  {'AUROC_val':>10}  {'Sym':>8}  {'l*-score':>10}  {'Selected':>8}")
    print("  " + "-" * 56)
    selected_names = {r["layer_name"] for r in top_layers_by_auroc}
    for r in sorted(sym_rows, key=lambda x: -(x.get("auroc_val") or 0)):
        sym_s  = f"{r['sym_score']:+.4f}" if r["sym_score"] is not None else "     N/A"
        marker = "  ★" if r["layer_name"] in selected_names else ""
        print(f"  {r['layer_name']:>10}  {r['auroc_val']:>10.4f}"
              f"  {sym_s:>8}  {r['lstar_score']:>10.4f}{marker}")
    print(f"  [SYM] → {done_flag}")

    return top_layers_by_auroc


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (multi-GPU + Flash-Attention-2)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_cfg: dict, device: str):
    """
    Load model with:
      - bfloat16
      - device_map='auto' + explicit max_memory per GPU  → forces Hugging Face
        to spread layers across ALL available GPUs even when the model fits on
        one (without max_memory it defaults to GPU 0 only for small models).
      - flash_attention_2 when available → faster attention, less VRAM
    """
    print(f"  Loading {model_cfg['display']} …")

    load_kwargs: dict = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
        print(f"  GPUs available: {n_gpu}  ({gpu_names})")

        # Allocate 90 % of each GPU's VRAM; remainder spills to CPU RAM.
        # This forces device_map='auto' to distribute layers across GPUs
        # even when the whole model would fit on GPU 0.
        max_memory: dict = {
            i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.90 / 1024**3)}GiB"
            for i in range(n_gpu)
        }
        max_memory["cpu"] = "48GiB"   # CPU RAM overflow buffer
        print(f"  max_memory per device: {max_memory}")

        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = max_memory
    else:
        load_kwargs["device_map"] = device

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["hf_id"],
        trust_remote_code=True,
        padding_side="left",   # left-pad for batch generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try flash attention 2 for faster attention + lower VRAM peak
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["hf_id"],
            attn_implementation="flash_attention_2",
            **load_kwargs,
        )
        print(f"  ✓ Flash-Attention-2 enabled.")
    except Exception as fa_err:
        print(f"  Flash-Attention-2 unavailable ({fa_err.__class__.__name__}) "
              f"— using default attention.")
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["hf_id"],
            **load_kwargs,
        )

    model.eval()

    # Report actual device map so we can verify distribution
    if hasattr(model, "hf_device_map"):
        device_counts: dict[str, int] = {}
        for dev in model.hf_device_map.values():
            device_counts[str(dev)] = device_counts.get(str(dev), 0) + 1
        print(f"  Layer distribution: {device_counts}")

    return model, tokenizer


def unload_model(model, tokenizer):
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_code_stub(problem: dict) -> str | None:
    """
    Build a function-signature stub from MBPP-style problems where `code` holds
    the reference solution.  Returns e.g.:
        def remove_Occ(s, ch):
            \"\"\"Write a python function to ...\"\"\"\
    so the model knows exactly which function name to implement.
    Returns None when the problem already has a code-stub `prompt` or no `code`.
    """
    code = (problem.get("code") or "").strip()
    text = (problem.get("text") or problem.get("prompt") or "").strip()
    if not code or not text:
        return None
    sig = None
    for line in code.split("\n"):
        s = line.strip()
        if s.startswith(("def ", "async def ")):
            sig = s
            break
    if not sig:
        return None
    return f"{sig}\n    \"\"\"{text}\"\"\""


def build_prompt(problem: dict, model_cfg: dict, tokenizer) -> str:
    task_text = problem.get("prompt", problem.get("text", ""))
    # When the prompt is a plain description, build a proper code stub so the
    # model uses the correct function name (required for test_list assertions).
    if not task_text.strip().startswith(("def ", "async def ")):
        stub = _make_code_stub(problem)
        if stub:
            task_text = stub
    messages  = [
        {"role": "system", "content": model_cfg["system_msg"]},
        {"role": "user",   "content": task_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def strip_code_fences(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    if t.startswith("```"):
        parts = t.split("\n", 1)
        t = parts[1] if len(parts) == 2 else ""
    if t.rstrip().endswith("```"):
        t = t[:t.rstrip().rfind("```")]
    return t


def _extract_code_from_response(text: str) -> str:
    """
    Extract Python code from a model response that may include markdown prose.
    Strategy: (1) find a ```python...``` or ```...``` block anywhere in text,
    (2) strip outer fences if text starts with them,
    (3) fall back to the first def/async def/class/import line onwards.
    """
    # (1) code fence anywhere in the response
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).rstrip()
    # (2) outer fence only
    t = strip_code_fences(text).strip()
    # (3) skip any leading prose by finding the first code-like line
    lines = t.split("\n")
    for i, line in enumerate(lines):
        if line.startswith(("def ", "async def ", "class ", "import ", "from ")):
            return "\n".join(lines[i:]).rstrip()
    return t


def _last_function_signature(prompt: str) -> str | None:
    for line in reversed(prompt.splitlines()):
        stripped = line.strip()
        if stripped.startswith(("def ", "async def ")):
            return stripped
    return None


def _prompt_body_indent(prompt: str) -> str:
    for line in reversed(prompt.splitlines()):
        if line.strip():
            indent_width = len(line) - len(line.lstrip(" "))
            return " " * indent_width
    return "    "


def _normalize_body_indentation(lines: list[str], body_indent: str) -> str:
    nonempty = [line for line in lines if line.strip()]
    if not nonempty:
        return ""
    common_indent = min(len(line) - len(line.lstrip(" ")) for line in nonempty)
    normalized_lines: list[str] = []
    for line in lines:
        if not line.strip():
            normalized_lines.append("")
            continue
        stripped = line[common_indent:] if len(line) >= common_indent else line.lstrip(" ")
        normalized_lines.append(body_indent + stripped)
    return "\n".join(normalized_lines).rstrip()


def _strip_repeated_signature_and_docstring(raw_text: str, prompt: str) -> str | None:
    signature = _last_function_signature(prompt)
    if not signature:
        return None
    body_indent = _prompt_body_indent(prompt)
    lines = raw_text.split("\n")
    signature_index = None
    for index, line in enumerate(lines):
        if line.strip() == signature:
            signature_index = index
            break
    if signature_index is None:
        return None
    remaining = lines[signature_index + 1:]
    while remaining and not remaining[0].strip():
        remaining = remaining[1:]
    if remaining and remaining[0].strip().startswith(('"""', "'''")):
        quote = '"""' if remaining[0].strip().startswith('"""') else "'''"
        remaining = remaining[1:]
        while remaining:
            line = remaining[0]
            remaining = remaining[1:]
            if quote in line:
                break
        while remaining and not remaining[0].strip():
            remaining = remaining[1:]
    truncated: list[str] = []
    for line in remaining:
        stripped = line.strip()
        if stripped in {"```", "<end_of_turn>", "<start_of_turn>model"}:
            break
        if line and not line.startswith((" ", "\t")):
            if stripped.startswith(("def ", "class ", "if __name__", "print(", "#")):
                break
        truncated.append(line)
    return _normalize_body_indentation(truncated, body_indent)


def truncate_to_function_continuation(text: str, prompt: str) -> str:
    normalized = strip_code_fences(text)
    if prompt in normalized:
        normalized = normalized.split(prompt, 1)[1]
    signature_stripped = _strip_repeated_signature_and_docstring(normalized, prompt)
    if signature_stripped:
        return signature_stripped
    lines: list[str] = []
    for line in normalized.split("\n"):
        stripped = line.strip()
        if stripped in {"```", "<end_of_turn>", "<start_of_turn>model"}:
            break
        if line and not line.startswith((" ", "\t")):
            if stripped.startswith(("def ", "class ", "if __name__", "print(", "#")):
                break
        lines.append(line)
    return "\n".join(lines).rstrip()


def decode_generation(raw: str, problem: dict) -> str:
    prompt_text = problem.get("prompt", problem.get("text", ""))
    prompt_stripped = prompt_text.strip()
    # When the prompt is a code stub (function signature), complete it.
    # When the prompt is a plain English description (MBPP+ HF format), the
    # model generates a complete function — extract and return it directly.
    if prompt_stripped.startswith(("def ", "async def ")):
        completion = truncate_to_function_continuation(raw, prompt_text)
        return prompt_text + "\n" + completion
    return _extract_code_from_response(raw)


def _run_subprocess(code: str, timeout: int = EXEC_TIMEOUT):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        proc = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode == 0, proc.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.unlink(fname)
        except Exception:
            pass


def evaluate_solution(full_solution: str, problem: dict) -> bool:
    test_cases = problem.get("test_list", problem.get("tests", []))
    if isinstance(test_cases, str):
        test_cases = [test_cases]
    if not test_cases:
        test_code = problem.get("test")
        entry_pt  = problem.get("entry_point")
        if test_code and entry_pt:
            test_cases = [f"{test_code}\n\ncheck({entry_pt})\n"]
        elif test_code:
            test_cases = [test_code]
        else:
            return False
    # Optional preamble fields present in different MBPP variants
    imports    = problem.get("test_imports", [])
    setup_code = problem.get("test_setup_code", "") or ""
    preamble   = ("\n".join(imports) + "\n" if imports else "") + (setup_code + "\n" if setup_code.strip() else "")
    for tc in test_cases:
        script = preamble + full_solution + "\n\n" + tc + "\n"
        ok, _err = _run_subprocess(script)
        if not ok:
            return False
    return True


def pass_at_k(n: int, c: int, k: int = 1) -> float:
    if n == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod([(n - c - i) / (n - i) for i in range(k)]))


# ─────────────────────────────────────────────────────────────────────────────
# STEERING HOOKS
# ─────────────────────────────────────────────────────────────────────────────

def _get_decoder_layer(model, layer_idx: int):
    for attr in ("model", "transformer", "gpt_neox", "language_model"):
        sub = getattr(model, attr, None)
        if sub is not None:
            for lattr in ("layers", "h", "blocks"):
                obj = getattr(sub, lattr, None)
                if isinstance(obj, torch.nn.ModuleList):
                    return obj[layer_idx]
    raise RuntimeError(f"Cannot locate decoder layer {layer_idx}.")


def _cast_direction(direction: torch.Tensor, model) -> torch.Tensor:
    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32
    # Also move to the same device as the first parameter
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    return direction.to(dtype=model_dtype, device=model_device)


class AdditionHook:
    """h ← h + α · d̂  at every decode step, last token only."""

    def __init__(self, model, layer_idx: int, direction: torch.Tensor, alpha: float):
        self._model     = model
        self._layer     = _get_decoder_layer(model, layer_idx)
        self._direction = direction.float().cpu()  # keep on CPU; cast per-call inside hook
        self._alpha     = alpha
        self._handle    = None

    def __enter__(self):
        alpha = self._alpha
        d_cpu = self._direction  # float32 on CPU

        def _hook(module, inp, out):
            h     = out[0] if isinstance(out, tuple) else out
            # Cast direction to the layer's actual device & dtype (handles multi-GPU)
            d     = d_cpu.to(dtype=h.dtype, device=h.device).squeeze()
            h_new = h.clone()
            h_new[:, -1, :] = h[:, -1, :] + alpha * d
            return (h_new,) + out[1:] if isinstance(out, tuple) else h_new

        self._handle = self._layer.register_forward_hook(_hook)
        return self

    def __exit__(self, *_):
        if self._handle:
            self._handle.remove()


class AblationHook:
    """h ← h − (h · d̂) · d̂  at last token only."""

    def __init__(self, model, layer_idx: int, direction: torch.Tensor):
        self._model     = model
        self._layer     = _get_decoder_layer(model, layer_idx)
        self._direction = direction.float().cpu()  # keep on CPU; cast per-call inside hook
        self._handle    = None

    def __enter__(self):
        d_cpu = self._direction

        def _hook(module, inp, out):
            h     = out[0] if isinstance(out, tuple) else out
            # Cast direction to the layer's actual device & dtype (handles multi-GPU)
            d      = d_cpu.to(dtype=h.dtype, device=h.device).squeeze()
            h_new  = h.clone()
            h_last = h[:, -1, :]
            proj   = (h_last * d).sum(dim=-1, keepdim=True)
            h_new[:, -1, :] = h_last - proj * d
            return (h_new,) + out[1:] if isinstance(out, tuple) else h_new

        self._handle = self._layer.register_forward_hook(_hook)
        return self

    def __exit__(self, *_):
        if self._handle:
            self._handle.remove()


# ─────────────────────────────────────────────────────────────────────────────
# BATCHED GENERATION WITH HOOK
# ─────────────────────────────────────────────────────────────────────────────

def _get_primary_device(model) -> torch.device:
    """Return the device of the first model parameter (works with device_map='auto')."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _generate_batch(
    model,
    tokenizer,
    model_cfg: dict,
    problems:  list[dict],
    seeds:     list[int],
    hook_cm,
) -> list[bool]:
    """
    Generate completions for a batch of problems under a single hook and
    evaluate each against its test cases.

    Returns a list of booleans (passed / failed) in the same order as problems.
    """
    device = _get_primary_device(model)

    prompts = [build_prompt(p, model_cfg, tokenizer) for p in problems]

    # Left-pad so all sequences align at the generated position
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

    # Set per-batch seeds via the first seed (deterministic enough for
    # a batch; individual seeds handled at the outer loop level)
    if seeds:
        torch.manual_seed(seeds[0])

    with hook_cm, torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    results    = []
    for i, problem in enumerate(problems):
        raw          = tokenizer.decode(out_ids[i][prompt_len:], skip_special_tokens=True)
        full_solution = decode_generation(raw, problem)
        try:
            passed = evaluate_solution(full_solution, problem)
        except Exception:
            passed = False
        results.append(passed)
    return results


def run_steering_experiment(
    model,
    tokenizer,
    model_cfg:    dict,
    test_problems: dict[str, dict],
    direction_np:  np.ndarray,
    layer_idx:     int,
    alpha:         float,
    run_pool:      list[tuple[str, int]],
    mode:          str,
    batch_size:    int = STEER_BATCH_SIZE,
) -> dict:
    """
    Run one steering experiment over a flat pool of (problem_id, run_i) pairs.

    run_pool is either wrong_pool (forward) or right_pool (backward/ablation).
    Each entry is one steered generation.  Pass rate is computed flat:
        n_passed / len(run_pool)
    not averaged per-problem.
    """
    direction = torch.from_numpy(direction_np.astype(np.float32))
    if direction.dim() == 1:
        direction = direction.unsqueeze(0).unsqueeze(0)
    dir_vec = direction.squeeze()

    n_total  = len(run_pool)
    n_passed = 0

    desc = f"  [{mode}] α={alpha:+.2g} l={layer_idx:02d}"

    with tqdm(total=n_total, desc=desc, leave=False, unit="run") as pbar:
        for batch_start in range(0, n_total, batch_size):
            batch       = run_pool[batch_start: batch_start + batch_size]
            batch_probs = [test_problems[tid]                            for tid, _     in batch]
            batch_seeds = [abs(hash(tid)) % (2**31) + run_i * 1000      for tid, run_i in batch]

            if mode == "ablation":
                hook_cm = AblationHook(model, layer_idx, dir_vec)
            else:
                hook_cm = AdditionHook(model, layer_idx, dir_vec, alpha)

            try:
                passed_flags = _generate_batch(
                    model, tokenizer, model_cfg,
                    batch_probs, batch_seeds, hook_cm,
                )
            except Exception as exc:
                tqdm.write(f"  [WARN] Batch generation failed: {exc}")
                passed_flags = [False] * len(batch)

            n_passed += sum(passed_flags)
            pbar.update(len(batch))

    pass_rate = n_passed / n_total if n_total > 0 else 0.0
    return {"n_passed": n_passed, "n_total": n_total, "pass_rate": pass_rate}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4: STEERING  (TEST split only)
# ─────────────────────────────────────────────────────────────────────────────

def _agg_pass1(per_prob: dict) -> float:
    vals = [v["pass@1"] for v in per_prob.values() if v.get("n_total", 0) > 0]
    return float(np.mean(vals)) if vals else 0.0


def compute_baseline_from_stored_runs(
    acts_root: Path,
    test_ids:  list[str],
) -> dict[str, dict]:
    """
    Compute baseline pass@1 per problem from pre-steering stored runs.
    Uses _iter_run_dirs so it handles run1/run2/… naming correctly.
    """
    results: dict[str, dict] = {}
    for tid in test_ids:
        prob_dir = acts_root / tid
        if not prob_dir.is_dir():
            continue
        n_r = len(_iter_run_dirs(prob_dir / "right"))
        n_w = len(_iter_run_dirs(prob_dir / "wrong"))
        n_tot = n_r + n_w
        results[tid] = {
            "n_right": n_r,
            "n_wrong": n_w,
            "n_total": n_tot,
            "pass@1":  pass_at_k(n_tot, n_r, 1) if n_tot > 0 else 0.0,
        }
    return results


def _detect_dataset_from_stored_ids(stored_ids: set[str]) -> str:
    """
    Guess which dataset the stored dirs came from based on directory name patterns.
    Returns 'bigcodebench' or 'mbpp'.
    """
    bcb_count  = sum(1 for s in stored_ids if "BigCodeBench" in s or s.startswith("BCB"))
    mbpp_count = sum(1 for s in stored_ids if "Mbpp" in s or s.isdigit())
    if bcb_count > mbpp_count:
        return "bigcodebench"
    return "mbpp"


def _build_problem_map(problems_list: list[dict], stored_ids: set[str]) -> dict[str, dict]:
    """
    Map dataset task_ids to stored directory names.

    Handles both MBPP+ and BigCodeBench:
      MBPP+         : task_id 26 / "Mbpp/26"  → dir "Mbpp_26"
      BigCodeBench  : task_id "BigCodeBench/0" → dir "BigCodeBench_0"

    Strategy
    --------
    1. Direct match after safe_id conversion  ("BigCodeBench/0" → "BigCodeBench_0" ✓).
    2. Numeric reverse-index for MBPP plain-integer IDs (26 → "Mbpp_26").
    3. Fall-through (unmatched) — recorded for diagnostics.
    """
    # Build: exact numeric part → stored_id  (for MBPP plain-integer IDs)
    num_to_stored: dict[str, str] = {}
    for sid in stored_ids:
        digits = re.sub(r"[^0-9]", "", sid)
        if digits:
            num_to_stored[digits] = sid

    candidate_map: dict[str, dict] = {}
    unmatched: list[str] = []

    for p in problems_list:
        raw_tid  = p["task_id"]
        safe_tid = _safe_id(raw_tid)   # "BigCodeBench/0"→"BigCodeBench_0", "Mbpp/26"→"Mbpp_26"

        # ── 1. Direct match ──────────────────────────────────────────────────
        if safe_tid in stored_ids:
            candidate_map[safe_tid] = p
            continue

        # ── 2. Numeric reverse-index (MBPP plain-integer IDs only) ───────────
        digits = re.sub(r"[^0-9]", "", str(raw_tid))
        if digits and digits in num_to_stored:
            candidate_map[num_to_stored[digits]] = p
            continue

        # ── 3. Fall-through (unmatched) ───────────────────────────────────────
        candidate_map[safe_tid] = p
        unmatched.append(str(raw_tid))

    if unmatched:
        print(f"  [MAP] {len(unmatched)} task_ids had no stored-dir match "
              f"(first 5: {unmatched[:5]})")
        print(f"  [MAP] Sample stored_ids: {sorted(stored_ids)[:5]}")

    return candidate_map


def run_steering(
    model_key:    str,
    acts_dir:     str,
    split:        dict,
    top_layers:   list[dict],
    out_dir:      Path,
    problem_map:  dict[str, dict],
    batch_size:   int,
    run_seed:     int = 0,
) -> list[dict]:
    """
    Stage 4: steering experiments, run-level split.

    For each test problem, its stored n_wrong runs → Forward Addition (improve them).
    For each test problem, its stored n_right runs → Backward Subtraction + Ablation.

    A problem with 3 right + 2 wrong contributes 2 runs to forward and 3 to backward.
    No problem-level pool membership required.
    Δfwd = steered_rate − 0.0  (wrong pool, baseline trivially 0).
    Δbwd = 1.0 − steered_rate  (right pool, baseline trivially 1).
    run_seed controls the generation seed offset so repeated runs differ.
    """
    steer_dir  = out_dir / "steering" / f"seed_{run_seed}"
    done_flag  = steer_dir / "steering_results.json"

    if done_flag.exists():
        print("  [STEER] Resuming — steering_results.json found.")
        with open(done_flag) as f:
            return json.load(f)

    steer_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = MODEL_REGISTRY[model_key]
    acts_root = _find_acts_root(acts_dir)

    test_all_ids = split["test"]["all_ids"]
    stored_probs = split["problems"]   # {tid: {"n_right": int, "n_wrong": int, ...}}

    # All test problems that exist in problem_map
    test_problems = {
        tid: problem_map[tid]
        for tid in test_all_ids
        if tid in problem_map
    }
    n_matched  = len(test_problems)
    n_expected = len(test_all_ids)
    if n_matched < n_expected:
        missing = sorted(set(test_all_ids) - set(test_problems.keys()))
        print(f"\n  [STEER] WARNING: {n_expected - n_matched} test problems "
              f"have no problem_map entry (first 5: {missing[:5]})")
    if n_matched == 0:
        print("  [STEER] FATAL: zero matched problems.")
        return []

    # Flat run pools: one entry per individual stored run, irrespective of problem.
    # wrong_pool = every run that failed in the test set.
    # right_pool = every run that succeeded in the test set.
    wrong_pool: list[tuple[str, int]] = []
    right_pool: list[tuple[str, int]] = []
    for tid in sorted(test_problems):
        sp = stored_probs.get(tid, {})
        for i in range(sp.get("n_wrong", 0)):
            wrong_pool.append((tid, run_seed * 10 + i))
        for i in range(sp.get("n_right", 0)):
            right_pool.append((tid, run_seed * 10 + i))

    n_wrong_runs = len(wrong_pool)
    n_right_runs = len(right_pool)

    if n_wrong_runs == 0 or n_right_runs == 0:
        print("  [STEER] FATAL: empty wrong or right pool in test set.")
        return []

    dir_dir    = out_dir / "probing" / "directions"
    directions: dict[str, np.ndarray] = {}
    for rec in top_layers:
        npy = dir_dir / f"{rec['layer_name']}.npy"
        if npy.exists():
            directions[rec["layer_name"]] = np.load(npy).astype(np.float32)
        else:
            print(f"  [WARN] Direction not found: {npy} — skipping layer.")

    model, tokenizer = load_model(model_cfg, device="cuda" if torch.cuda.is_available() else "cpu")

    # Guard: drop any top-layer entries whose index exceeds the model depth.
    _n_model_layers = None
    for _attr in ("model", "transformer", "gpt_neox", "language_model"):
        _sub = getattr(model, _attr, None)
        if _sub is not None:
            for _lattr in ("layers", "h", "blocks"):
                _obj = getattr(_sub, _lattr, None)
                if isinstance(_obj, torch.nn.ModuleList):
                    _n_model_layers = len(_obj)
                    break
        if _n_model_layers is not None:
            break
    if _n_model_layers is not None:
        _valid = [r for r in top_layers if r["layer_idx"] < _n_model_layers]
        _skipped = [r["layer_name"] for r in top_layers if r["layer_idx"] >= _n_model_layers]
        if _skipped:
            print(f"  [STEER] Dropping out-of-range layers (model has {_n_model_layers}): {_skipped}")
        top_layers = _valid
        directions = {k: v for k, v in directions.items() if k in {r["layer_name"] for r in top_layers}}

    # ── Analytical baselines from stored run outcomes ─────────────────────────
    # wrong_pool = runs that FAILED  → stored pass rate = 0.0 by definition
    # right_pool = runs that PASSED  → stored pass rate = 1.0 by definition
    # Using these avoids stochastic variance from re-generation and removes
    # the ~26-min α=0 pre-generation step.
    # Δfwd = steered_rate − 0.0  (positive = recovery of failing runs)
    # Δbwd = steered_rate − 1.0  (negative = degradation of passing runs)
    baseline_fwd_p1 = 0.0
    baseline_bwd_p1 = 1.0

    # Remove any stale checkpoint that stored the wrong α=0 baselines
    base_ck = steer_dir / "pool_baselines.json"
    if base_ck.exists():
        base_ck.unlink()

    print(f"\n  [STEER] Model              : {model_cfg['display']}")
    print(f"  [STEER] Wrong pool (fwd)   : {n_wrong_runs} runs  baseline=0.0%  (stored: all failed)")
    print(f"  [STEER] Right pool (bwd)   : {n_right_runs} runs  baseline=100.0%  (stored: all passed)")
    print(f"  [STEER] Δfwd = steered_rate − 0.0   (positive = recovery)")
    print(f"  [STEER] Δbwd = steered_rate − 1.0   (negative = degradation)")
    print(f"  [STEER] Layers             : {[r['layer_name'] for r in top_layers]}")
    print(f"  [STEER] Batch size         : {batch_size}")

    all_layer_results: list[dict] = []

    for layer_rec in tqdm(
        [r for r in top_layers if r["layer_name"] in directions],
        desc=f"  Layers [{model_cfg['display']}]",
        unit="layer",
    ):
        lname     = layer_rec["layer_name"]
        li        = layer_rec["layer_idx"]
        auroc_val = layer_rec.get("auroc_val")
        sym       = layer_rec.get("sym_score")
        lstar     = layer_rec.get("lstar_score")
        dir_np    = directions[lname]

        ck_path = steer_dir / f"{lname}_results.json"
        if ck_path.exists():
            print(f"  [STEER] {lname}: checkpoint found — loading.")
            with open(ck_path) as f:
                all_layer_results.append(json.load(f))
            continue

        tqdm.write(f"\n  ── {lname}  AUROC_val={auroc_val}  Sym={sym}  l*={lstar} ──")
        tqdm.write(f"  wrong_pool={n_wrong_runs} runs  baseline_fwd={baseline_fwd_p1*100:.1f}%  "
                   f"right_pool={n_right_runs} runs  baseline_bwd={baseline_bwd_p1*100:.1f}%")

        # (a) Forward Addition — steer wrong_pool, expect Δ > 0 (improvement)
        tqdm.write("    (a) Forward Addition  (wrong_pool, α > 0) …")
        fwd_results: list[dict] = []
        for alpha in ALPHAS_FWD:
            res = run_steering_experiment(
                model, tokenizer, model_cfg,
                test_problems, dir_np, li, alpha,
                wrong_pool, mode="forward", batch_size=batch_size,
            )
            rate = res["pass_rate"]
            dp1  = (rate - baseline_fwd_p1) * 100   # positive = improvement
            fwd_results.append({
                "alpha": alpha, "pass_rate": rate,
                "n_passed": res["n_passed"], "n_total": res["n_total"],
                "delta_p1": dp1,
            })
            tqdm.write(f"      α={alpha:+.1f}  rate={rate*100:.1f}%  Δfwd={dp1:+.1f}%")

        # (b) Backward Subtraction — steer right_pool, expect Δ < 0 (degradation)
        tqdm.write("    (b) Backward Subtraction  (right_pool, α < 0) …")
        bwd_results: list[dict] = []
        for alpha in ALPHAS_BWD:
            res = run_steering_experiment(
                model, tokenizer, model_cfg,
                test_problems, dir_np, li, alpha,
                right_pool, mode="backward", batch_size=batch_size,
            )
            rate = res["pass_rate"]
            dp1  = (rate - baseline_bwd_p1) * 100   # negative = degradation
            bwd_results.append({
                "alpha": alpha, "pass_rate": rate,
                "n_passed": res["n_passed"], "n_total": res["n_total"],
                "delta_p1": dp1,
            })
            tqdm.write(f"      α={alpha:.1f}  rate={rate*100:.1f}%  Δbwd={dp1:+.1f}%")

        # (c) Direction Ablation — steer right_pool, expect Δ < 0 (degradation)
        tqdm.write("    (c) Direction Ablation  (right_pool, h ← h − (h·d̂)d̂) …")
        res_abl = run_steering_experiment(
            model, tokenizer, model_cfg,
            test_problems, dir_np, li, ALPHA_ABLATE,
            right_pool, mode="ablation", batch_size=batch_size,
        )
        rate_abl = res_abl["pass_rate"]
        dp1_abl  = (rate_abl - baseline_bwd_p1) * 100   # negative = degradation
        tqdm.write(f"      Ablation  rate={rate_abl*100:.1f}%  Δabl={dp1_abl:+.1f}%")

        best_fwd = max((r["delta_p1"] for r in fwd_results), default=0.0)
        best_bwd = min((r["delta_p1"] for r in bwd_results), default=0.0)
        tqdm.write(f"  → Fwd best Δ={best_fwd:+.1f}%  Bwd best Δ={best_bwd:+.1f}%  "
                   f"Abl Δ={dp1_abl:+.1f}%")

        layer_result = {
            "layer_idx":         li,
            "layer_name":        lname,
            "auroc_val":         auroc_val,
            "sym_score":         sym,
            "lstar_score":       lstar,
            "baseline_fwd_p1":   baseline_fwd_p1,
            "baseline_bwd_p1":   baseline_bwd_p1,
            "n_wrong_runs":      n_wrong_runs,
            "n_right_runs":      n_right_runs,
            "forward":           fwd_results,
            "backward":          bwd_results,
            "ablation": {
                "alpha":     ALPHA_ABLATE,
                "pass_rate": rate_abl,
                "n_passed":  res_abl["n_passed"],
                "n_total":   res_abl["n_total"],
                "delta_p1":  dp1_abl,
            },
            "best_fwd_delta":    best_fwd,
            "best_bwd_delta":    best_bwd,
            "abl_delta":         dp1_abl,
        }

        all_layer_results.append(layer_result)
        with open(ck_path, "w") as f:
            json.dump(layer_result, f, indent=2)

    unload_model(model, tokenizer)

    with open(done_flag, "w") as f:
        json.dump(all_layer_results, f, indent=2)

    _write_steering_report(steer_dir, model_cfg["display"], all_layer_results)
    print(f"\n  [STEER] → {done_flag}")
    return all_layer_results


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def _write_steering_report(steer_dir: Path, display: str, layer_results: list[dict]):
    SEP  = "=" * 80
    SEP2 = "-" * 80
    lines = [
        SEP,
        f"  STEERING REPORT — {display}",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
        f"  Layer selection  : AUROC_val",
        f"  wrong_pool (fwd) : all test runs that FAILED  → baseline = 0.0  (stored outcome)",
        f"  right_pool (bwd) : all test runs that PASSED  → baseline = 1.0  (stored outcome)",
        f"  Δfwd = steered_rate − 0.0   (positive = recovery of failing runs)",
        f"  Δbwd = steered_rate − 1.0   (negative = degradation of passing runs)",
        f"  Pass rate = flat n_passed / n_total over the pool (not per-problem average)",
        SEP, "",
    ]

    for res in layer_results:
        auc_s  = f"{res['auroc_val']:.4f}"   if res["auroc_val"]   is not None else "N/A"
        sym_s  = f"{res['sym_score']:+.4f}"  if res.get("sym_score") is not None else "N/A"
        ls_s   = f"{res['lstar_score']:.4f}" if res.get("lstar_score") is not None else "N/A"
        n_wr   = res.get("n_wrong_runs", "?")
        n_rr   = res.get("n_right_runs", "?")
        lines += [
            SEP2,
            f"  Layer: {res['layer_name']}  AUROC_val={auc_s}  Sym={sym_s}  l*={ls_s}",
            SEP2,
            f"  wrong_pool: {n_wr} runs  baseline_fwd={res.get('baseline_fwd_p1',0)*100:.2f}%",
            f"  right_pool: {n_rr} runs  baseline_bwd={res.get('baseline_bwd_p1',1)*100:.2f}%",
            "",
            "  (a) Forward Addition  (wrong_pool, α > 0)",
            f"  {'α':>7}  {'rate':>8}  {'Δfwd':>8}",
        ]
        for row in res["forward"]:
            lines.append(f"  {row['alpha']:>7.1f}  "
                         f"{row['pass_rate']*100:>7.2f}%  {row['delta_p1']:>+7.2f}%")
        lines += [
            f"  Best Δfwd = {res['best_fwd_delta']:+.2f}%", "",
            "  (b) Backward Subtraction  (right_pool, α < 0)",
            f"  {'α':>7}  {'rate':>8}  {'Δbwd':>8}",
        ]
        for row in res["backward"]:
            lines.append(f"  {row['alpha']:>7.1f}  "
                         f"{row['pass_rate']*100:>7.2f}%  {row['delta_p1']:>+7.2f}%")
        lines += [
            f"  Best Δbwd = {res['best_bwd_delta']:+.2f}%", "",
            "  (c) Direction Ablation  (right_pool)",
            f"    rate={res['ablation']['pass_rate']*100:.2f}%  "
            f"Δabl={res['ablation']['delta_p1']:+.2f}%",
            "",
        ]

    lines.append(SEP)
    text = "\n".join(lines)
    print(text)
    (steer_dir / "steering_report.txt").write_text(text, encoding="utf-8")

    csv_path = steer_dir / "steering_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "layer", "auroc_val", "sym_score", "lstar_score",
            "baseline_fwd_p1", "n_wrong_runs",
            "baseline_bwd_p1", "n_right_runs",
            "best_fwd_alpha", "best_fwd_rate", "best_fwd_delta",
            "best_bwd_alpha", "best_bwd_rate", "best_bwd_delta",
            "abl_rate", "abl_delta",
        ])
        for res in layer_results:
            e1 = max(res["forward"],  key=lambda r: r["delta_p1"]) if res["forward"]  else {}
            e2 = min(res["backward"], key=lambda r: r["delta_p1"]) if res["backward"] else {}
            w.writerow([
                res["layer_name"], res["auroc_val"],
                res.get("sym_score"), res.get("lstar_score"),
                round(res.get("baseline_fwd_p1", 0) * 100, 2), res.get("n_wrong_runs", ""),
                round(res.get("baseline_bwd_p1", 1) * 100, 2), res.get("n_right_runs", ""),
                e1.get("alpha", ""), round(e1.get("pass_rate", 0) * 100, 2),
                round(e1.get("delta_p1", 0), 2),
                e2.get("alpha", ""), round(e2.get("pass_rate", 0) * 100, 2),
                round(e2.get("delta_p1", 0), 2),
                round(res["ablation"]["pass_rate"] * 100, 2),
                round(res["ablation"]["delta_p1"], 2),
            ])
    print(f"  [STEER] CSV → {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(model_key: str, out_root: Path, batch_size: int) -> None:
    model_cfg = MODEL_REGISTRY[model_key]
    acts_dir  = ACTS_ROOTS[model_key]
    display   = model_cfg["display"]

    if not Path(acts_dir).exists():
        print(f"  [ERROR] Acts dir not found: {acts_dir} — skipping {display}.")
        return

    existing = sorted(out_root.glob(f"{model_key}_*"))
    if existing:
        out_dir = existing[-1]
        print(f"  [INFO] Reusing output dir: {out_dir}")
    else:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = out_root / f"{model_key}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*72}")
    print(f"  MODEL : {display}")
    print(f"  Acts  : {acts_dir}")
    print(f"  Out   : {out_dir}")
    print(f"{'#'*72}")

    print(f"\n{'='*60}\n  STAGE 1 — SPLIT  (60 / 20 / 20)\n{'='*60}")
    split = run_split(acts_dir, out_dir)

    print(f"\n{'='*60}\n  STAGE 2 — PROBING  (ranked by AUROC_val)\n{'='*60}")
    all_probe_results, probe_top5 = run_probing(acts_dir, split, out_dir)

    if not probe_top5:
        print(f"  [ERROR] No probe results — cannot continue for {display}.")
        return

    print(f"\n{'='*60}\n  STAGE 3 — SYMMETRY + L*-SCORE (reference)\n{'='*60}")
    print("  Steering layers selected by AUROC_val, not l*.")
    top_layers = run_symmetry_and_lstar(acts_dir, split, probe_top5, out_dir)

    if not top_layers:
        print(f"  [ERROR] Layer selection failed — cannot steer {display}.")
        return

    print(f"\n{'='*60}\n  STAGE 4 — STEERING  (3 seeds)\n{'='*60}")
    print(f"  Fwd Addition   : n_wrong stored runs per problem (run-level).")
    print(f"  Bwd Subtraction: n_right stored runs per problem (run-level).")
    print(f"  Ablation       : n_right stored runs per problem (run-level).")
    print(f"  Delta baseline : stored overall TEST pass@1.")
    print(f"  Generation batch size: {batch_size}.")

    acts_root  = _find_acts_root(acts_dir)
    stored_ids = {
        d.name for d in acts_root.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS
    }

    # Auto-detect dataset from stored directory names
    dataset_kind = _detect_dataset_from_stored_ids(stored_ids)
    if dataset_kind == "bigcodebench":
        print("  Loading BigCodeBench problems …")
        problems_list = load_bigcodebench()
    else:
        print("  Loading MBPP+ problems …")
        problems_list = load_mbppplus()

    problem_map = _build_problem_map(problems_list, stored_ids)

    print(f"  [STEER] Dataset detected      : {dataset_kind}")
    print(f"  [STEER] Problems loaded       : {len(problems_list)}")
    print(f"  [STEER] Stored problem dirs   : {len(stored_ids)}")
    print(f"  [STEER] Matched (map size)    : {len(problem_map)}")

    for run_seed in range(3):
        print(f"\n{'─'*60}")
        print(f"  STEERING RUN  seed={run_seed}  ({run_seed+1}/3)")
        print(f"{'─'*60}")
        run_steering(
            model_key=model_key,
            acts_dir=acts_dir,
            split=split,
            top_layers=top_layers,
            out_dir=out_dir,
            problem_map=problem_map,
            batch_size=batch_size,
            run_seed=run_seed,
        )

    print(f"\n  ✓ {display} complete (3 steering seeds).\n")


def main():
    parser = argparse.ArgumentParser(
        description="Unified CODE_LLM pipeline: split → probe → sym+l* → steer"
    )
    parser.add_argument(
        "--model", default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help="Run only this model (default: both).",
    )
    parser.add_argument(
        "--out-root", default=str(OUT_ROOT),
        help=f"Root output directory (default: {OUT_ROOT}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=STEER_BATCH_SIZE,
        help=f"Steering generation batch size (default: {STEER_BATCH_SIZE}). "
             f"Increase for more GPU utilisation.",
    )
    args     = parser.parse_args()
    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    models = (
        [args.model] if args.model
        else ["qwen-coder-1.5b-instruct", "qwen-coder-7b-instruct"]
    )

    for mk in models:
        try:
            run_pipeline(mk, out_root, args.batch_size)
        except Exception as exc:
            import traceback
            print(f"\n[ERROR] {MODEL_REGISTRY[mk]['display']} failed: {exc}")
            traceback.print_exc()

    print(f"\n{'='*60}\n  ALL DONE  →  {out_root}\n{'='*60}")


if __name__ == "__main__":
    main()
