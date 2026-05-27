"""
baseline_humaneval.py  —  Unsteered Distribution-Level Baseline Evaluation (HumanEvalPlus)
============================================================================================
Computes the pure, unsteered base-model pool-wise run-level success rate
separately for two historically-defined pools drawn from the TEST split:

  • Wrong Pool  — test problems / runs that FAILED in the stored activation data.
  • Right Pool  — test problems / runs that PASSED in the stored activation data.

IMPORTANT — Interpretation of historical counts
------------------------------------------------
The ``wrong_count`` and ``right_count`` values from the activation store are used
ONLY as sampling allocation weights: they determine *how many* fresh completions
to generate per problem so that the pool sizes mirror the original data collection.
They are NOT evaluation labels.  Every generated completion is evaluated
independently from scratch by executing it against the problem's unit tests.
This is a distribution-level baseline evaluation with fully fresh sampling,
not a replay of stored outcomes.

Metrics (pool-wise run-level success rate — NOT "pass@k" in the HumanEval sense)
----------------------------------------------------------------------------------
  pool_wise_success_wrong = passed_wrong_pool_count / Total_Wrong_Runs
  pool_wise_success_right = passed_right_pool_count / Total_Right_Runs

"Pool-wise run-level success rate" counts each generation attempt as one
independent trial.  It differs from the canonical pass@k estimator, which
accounts for sample variance; reviewers should not conflate the two.

NO activation hooks, steering vectors, or alpha modifications are applied.
This script is the clean reference baseline for downstream steering Δ metrics.

Outputs
-------
  <seed_dir>/pool_baselines.json          — machine-readable summary
  <seed_dir>/baseline_report.txt          — human-readable report
  <seed_dir>/wrong_generations.jsonl      — every wrong-pool generation attempt
  <seed_dir>/right_generations.jsonl      — every right-pool generation attempt

Usage
-----
  python baseline_humaneval.py
  python baseline_humaneval.py --model qwen-coder-7b-instruct
  python baseline_humaneval.py --out-root /my/output/dir
  python baseline_humaneval.py --run-seed 1    # repeat with a different global seed
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import resource
import subprocess
import sys
import tempfile
import importlib.machinery as _imachinery
import types as _types
from datetime import datetime
from pathlib import Path

# ── Torchvision import guard ──────────────────────────────────────────────────
# transformers>=4.x imports torchvision inside image_utils.py.  On systems
# where torchvision is absent, mis-installed, or has a circular-import bug
# (AttributeError: partially initialized module 'torchvision' has no attribute
# 'extension'), this crashes the entire transformers import chain and prevents
# Qwen2ForCausalLM from loading.  LLM inference does not use any vision ops,
# so we unconditionally inject a minimal stub before transformers is imported.
#
# __spec__ must be set: transformers calls importlib.util.find_spec("torchvision")
# which raises ValueError if the already-registered module has __spec__ = None.
_tv_stub       = _types.ModuleType("torchvision")
_tv_ext        = _types.ModuleType("torchvision.extension")
_tv_io         = _types.ModuleType("torchvision.io")
_tv_transforms = _types.ModuleType("torchvision.transforms")
_tv_stub.__spec__       = _imachinery.ModuleSpec("torchvision",            loader=None, is_package=True)
_tv_ext.__spec__        = _imachinery.ModuleSpec("torchvision.extension",  loader=None)
_tv_io.__spec__         = _imachinery.ModuleSpec("torchvision.io",         loader=None)
_tv_transforms.__spec__ = _imachinery.ModuleSpec("torchvision.transforms", loader=None)
# __path__ is required for Python to resolve sub-module imports on this stub.
_tv_stub.__path__ = []
_tv_ext._has_ops      = lambda: False
_tv_io.ImageReadMode  = type("ImageReadMode", (), {})()
_tv_io.decode_image   = lambda *a, **kw: None
# transformers/image_utils.py unconditionally imports InterpolationMode at
# module level; provide a plain-class stand-in with the standard enum values.
class _InterpolationMode:
    NEAREST = 0; BILINEAR = 2; BICUBIC = 3; LANCZOS = 1; HAMMING = 5; BOX = 4
_tv_transforms.InterpolationMode = _InterpolationMode
del _InterpolationMode
_tv_stub.extension  = _tv_ext
_tv_stub.io         = _tv_io
_tv_stub.transforms = _tv_transforms
sys.modules["torchvision"]            = _tv_stub
sys.modules["torchvision.extension"]  = _tv_ext
sys.modules["torchvision.io"]         = _tv_io
sys.modules["torchvision.transforms"] = _tv_transforms
del _tv_stub, _tv_ext, _tv_io, _tv_transforms
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# The torchvision stub above makes transformers' `_is_package_available` see
# torchvision as present (find_spec returns a non-None spec), which causes
# `is_torchvision_available()` to return True and triggers additional torchvision
# imports inside modeling_qwen2.py's lazy-load chain — imports our stub doesn't
# fully satisfy.  Force the flag back to False immediately after the import so
# that all is_torchvision_available()-guarded code paths are skipped.
try:
    import transformers.utils.import_utils as _tui
    _tui._torchvision_available = False
    del _tui
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (mirrors pipeline_humaneval.py — keep in sync)
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_RATIO    = 0.60
VAL_RATIO      = 0.20
SPLIT_SEED     = 42          # must match pipeline_humaneval.py exactly

TEMPERATURE    = 0.8
TOP_P          = 0.95
MAX_NEW_TOKENS = 512

# Wall-clock timeout for the unit-test subprocess (seconds).
EXEC_TIMEOUT   = 10
# Maximum RSS memory the test subprocess may use (bytes).  4 GiB.
EXEC_MEM_LIMIT = 4 * 1024 ** 3
# Maximum CPU seconds the test subprocess may consume.
EXEC_CPU_LIMIT = 30

# Default batch size = 1 to guarantee per-sample RNG independence.
# Each sample is seeded with its own deterministic torch.Generator;
# batching across samples would couple the stochastic sampling chain.
DEFAULT_BATCH_SIZE = 128

EXPECTED_RUNS_PER_PROBLEM = 5

SKIP_DIRS = {
    "probing", "steering", "plots", "pipeline2_results",
    "pipeline2_probing", "all", "contrastive", "comparison",
    "probing_5fold_cv",
}

# ── HumanEvalPlus activation roots ───────────────────────────────────────────
_HE_BASE = (
    "/media/kpdubey/8.0 TB Volume/Shubham/MI/PROBING/"
    "layerwise_results_contrastive_pipeline1/humanevalplus"
)

ACTS_ROOTS = {
    "qwen-coder-1.5b-instruct": (
        f"{_HE_BASE}/qwen-coder-1.5b-instruct_20260426_162930/all"
    ),
    "qwen-coder-7b-instruct": (
        f"{_HE_BASE}/qwen-coder-7b-instruct_20260426_162930/all"
    ),
}

MODEL_REGISTRY = {
    "qwen-coder-1.5b-instruct": {
        "hf_id":      "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "display":    "Qwen2.5-Coder-1.5B",
        "system_msg": "You are an expert Python programmer.",
    },
    "qwen-coder-7b-instruct": {
        "hf_id":      "Qwen/Qwen2.5-Coder-7B-Instruct",
        "display":    "Qwen2.5-Coder-7B",
        "system_msg": "You are an expert Python programmer.",
    },
}

OUT_ROOT          = Path(__file__).parent / "baseline_results_humaneval"
# Root of pipeline_humaneval.py output — used as fallback for split.json search.
PIPELINE_OUT_ROOT = Path(__file__).parent / "pipeline_results_humaneval"

# Hardcoded split.json paths from the canonical pipeline_humaneval.py runs.
# Fill these in once pipeline_humaneval.py has been run; empty dict falls back
# to auto-detection (most-recent glob under PIPELINE_OUT_ROOT).
SPLIT_JSONS: dict[str, Path] = {
    # "qwen-coder-1.5b-instruct": PIPELINE_OUT_ROOT / "qwen-coder-1.5b-instruct_YYYYMMDD_HHMMSS" / "split.json",
    # "qwen-coder-7b-instruct":   PIPELINE_OUT_ROOT / "qwen-coder-7b-instruct_YYYYMMDD_HHMMSS"   / "split.json",
}


# ─────────────────────────────────────────────────────────────────────────────
# MASTER SEED INITIALISATION
# Sets all global RNG states at once so no stochastic subsystem is left
# unseeded.  Called once at pipeline startup before any model or data work.
# ─────────────────────────────────────────────────────────────────────────────

def set_global_seeds(seed: int) -> None:
    """Initialise every RNG that could affect generation or evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic CuDNN kernels (may slow down; comment out if not needed).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC SEED FORMULA
# Python's built-in hash() is session-randomised (PYTHONHASHSEED).
# MD5 is deterministic across all runs, processes, and Python versions.
# ─────────────────────────────────────────────────────────────────────────────

def _deterministic_seed(tid: str, run_offset: int) -> int:
    """
    Derive a fully reproducible per-run integer seed from a task ID and
    an integer offset.  Uses MD5 so the result is identical across Python
    sessions regardless of PYTHONHASHSEED.

    Formula mirrors pipeline_humaneval.py (with MD5 fix applied):
        base  = MD5(tid) % 2**31
        seed  = base + run_offset * 1000
    """
    base = int(hashlib.md5(tid.encode()).hexdigest(), 16) % (2 ** 31)
    return base + run_offset * 1000


# ─────────────────────────────────────────────────────────────────────────────
# NO-OP HOOK  (pure model — absolutely no steering)
# ─────────────────────────────────────────────────────────────────────────────

class NoHook:
    """Pass-through context manager — model runs with zero intervention."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADER  (HumanEvalPlus)
# ─────────────────────────────────────────────────────────────────────────────

def load_humanevalplus() -> list[dict]:
    """
    Load HumanEvalPlus problems.  Priority order:
      1. evalplus Python package   — get_human_eval_plus()
      2. evalplus/humanevalplus HuggingFace dataset
      3. openai/openai-evals HumanEval fallback (no plus tests)
    """
    try:
        from evalplus.data import get_human_eval_plus
        problems = get_human_eval_plus()
        result   = [{"task_id": k, **v} for k, v in problems.items()]
        print(f"  [DATA] Loaded HumanEvalPlus via evalplus pkg: {len(result)} problems")
        return result
    except ImportError:
        pass
    except Exception as e:
        print(f"  [DATA] evalplus get_human_eval_plus failed: {e}")

    from datasets import load_dataset

    try:
        ds = load_dataset("evalplus/humanevalplus", split="test", trust_remote_code=True)
        result = [dict(row) for row in ds]
        print(f"  [DATA] Loaded evalplus/humanevalplus HF: {len(result)} problems")
        return result
    except Exception as e:
        print(f"  [DATA] evalplus/humanevalplus HF unavailable: {e}")

    try:
        ds = load_dataset(
            "openai/openai-evals", "HumanEval", split="test", trust_remote_code=True
        )
        result = [dict(row) for row in ds]
        print(f"  [DATA] Loaded openai/openai-evals HumanEval: {len(result)} problems "
              f"(no plus tests — pass rates may differ)")
        return result
    except Exception as e:
        raise RuntimeError(
            f"Cannot load HumanEvalPlus. Install evalplus or the datasets library.\n{e}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVATION-DIR HELPERS  (mirrors pipeline_humaneval.py)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_id(tid) -> str:
    return str(tid).replace("/", "_")


def _find_acts_root(acts_dir: str | Path) -> Path:
    root = Path(acts_dir)
    if (root / "all").is_dir():
        return root / "all"
    return root


def _iter_run_dirs(verdict_dir: Path) -> list[Path]:
    if not verdict_dir.is_dir():
        return []
    return sorted(
        [p for p in verdict_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SPLIT JSON LOADER  (load identical split from pipeline_humaneval.py output)
# ─────────────────────────────────────────────────────────────────────────────

def _auto_detect_split_json(model_key: str) -> Path | None:
    """
    Return the split.json for `model_key`.  Priority:
      1. SPLIT_JSONS hardcoded path (canonical run, always preferred)
      2. Most-recent glob match under PIPELINE_OUT_ROOT (fallback)
    """
    if model_key in SPLIT_JSONS:
        return SPLIT_JSONS[model_key]
    if not PIPELINE_OUT_ROOT.is_dir():
        return None
    candidates = sorted(PIPELINE_OUT_ROOT.glob(f"{model_key}_*/split.json"))
    return candidates[-1] if candidates else None


def load_split_from_json(
    split_path: Path,
) -> tuple[list[str], dict[str, dict]]:
    """
    Load the test split and per-problem historical counts from a split.json
    produced by pipeline_humaneval.py.

    Returns
    -------
    test_ids        : list of str   — problem IDs in the test split
    historical_data : {tid: {"wrong_count": int, "right_count": int}}
                      Counts are sampling allocation weights only, not labels.
    """
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)

    test_ids:      list[str]       = split["test"]["all_ids"]
    problems_meta: dict[str, dict] = split.get("problems", {})

    historical_data: dict[str, dict] = {
        tid: {
            "wrong_count": problems_meta[tid]["n_wrong"],
            "right_count": problems_meta[tid]["n_right"],
        }
        for tid in test_ids
        if tid in problems_meta
    }
    return test_ids, historical_data


# ─────────────────────────────────────────────────────────────────────────────
# SPLIT  (fallback: recompute from acts_root — identical logic to pipeline_humaneval.py)
# ─────────────────────────────────────────────────────────────────────────────

def scan_runs(acts_root: Path) -> dict[str, dict]:
    """
    Walk acts_root and collect historical per-problem run counts.

    NOTE: These counts (n_right / n_wrong) are used ONLY as sampling
    allocation weights — they tell us how many fresh completions to generate
    per problem so the pool sizes match the original data collection.
    They are NOT used as evaluation labels.
    """
    problems: dict[str, dict] = {}
    for prob_dir in sorted(acts_root.iterdir()):
        if not prob_dir.is_dir() or prob_dir.name in SKIP_DIRS:
            continue
        tid = prob_dir.name
        n_r = len(_iter_run_dirs(prob_dir / "right"))
        n_w = len(_iter_run_dirs(prob_dir / "wrong"))
        problems[tid] = {"n_right": n_r, "n_wrong": n_w, "n_total": n_r + n_w}
    return problems


def make_test_split(all_ids: list[str]) -> list[str]:
    """
    Reproduce the exact TEST split used by pipeline_humaneval.py.
    SPLIT_SEED=42 and ratio 60/20/20 must never change.
    """
    rng  = np.random.default_rng(SPLIT_SEED)
    ids  = sorted(all_ids)
    perm = rng.permutation(len(ids))
    n     = len(ids)
    n_tr  = max(1, int(n * TRAIN_RATIO))
    n_val = max(1, int(n * VAL_RATIO))
    tr_idx  = set(perm[:n_tr].tolist())
    val_idx = set(perm[n_tr:n_tr + n_val].tolist())
    return [ids[i] for i in range(n) if i not in tr_idx and i not in val_idx]


def build_test_historical_data(acts_root: Path) -> tuple[list[str], dict[str, dict]]:
    """
    Scan acts_root, reproduce the test split, and return:
      test_ids         — stored dir-name IDs in the test split
      historical_data  — {tid: {"wrong_count": int, "right_count": int}}

    The counts in historical_data are sampling allocation weights only.
    They reflect how many runs of each outcome existed in the activation
    store; they do NOT label the new generations produced by this script.
    """
    all_problems = scan_runs(acts_root)
    test_ids     = make_test_split(sorted(all_problems.keys()))
    historical_data = {
        tid: {
            "wrong_count": all_problems[tid]["n_wrong"],
            "right_count": all_problems[tid]["n_right"],
        }
        for tid in test_ids
        if tid in all_problems
    }
    return test_ids, historical_data


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM MAP  (stored dir name → dataset problem dict)
# HumanEval: task_id "HumanEval/0" → _safe_id → "HumanEval_0" (direct match).
# ─────────────────────────────────────────────────────────────────────────────

def build_problem_map(
    problems_list: list[dict],
    stored_ids:    set[str],
) -> dict[str, dict]:
    num_to_stored: dict[str, str] = {}
    for sid in stored_ids:
        digits = re.sub(r"[^0-9]", "", sid)
        if digits:
            num_to_stored[digits] = sid

    problem_map: dict[str, dict] = {}
    unmatched: list[str] = []

    for p in problems_list:
        raw_tid  = p["task_id"]
        safe_tid = _safe_id(raw_tid)
        if safe_tid in stored_ids:
            problem_map[safe_tid] = p
            continue
        digits = re.sub(r"[^0-9]", "", str(raw_tid))
        if digits and digits in num_to_stored:
            problem_map[num_to_stored[digits]] = p
            continue
        problem_map[safe_tid] = p
        unmatched.append(str(raw_tid))

    if unmatched:
        print(f"  [MAP] {len(unmatched)} task_ids had no stored-dir match "
              f"(first 5: {unmatched[:5]})")
    return problem_map


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING / UNLOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_cfg: dict):
    """
    Load model in bfloat16 with device_map='auto'.
    Attempts Flash-Attention-2; falls back to default attention silently.
    """
    print(f"  Loading {model_cfg['display']} …")
    load_kwargs: dict = dict(dtype=torch.bfloat16, trust_remote_code=True)

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print(f"  GPUs: {n_gpu}  ({[torch.cuda.get_device_name(i) for i in range(n_gpu)]})")
        max_memory = {
            i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.90 / 1024**3)}GiB"
            for i in range(n_gpu)
        }
        max_memory["cpu"] = "48GiB"
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = max_memory
    else:
        load_kwargs["device_map"] = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["hf_id"], trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["hf_id"], attn_implementation="flash_attention_2", **load_kwargs
        )
        print("  ✓ Flash-Attention-2 enabled.")
    except Exception as fa_err:
        print(f"  Flash-Attention-2 unavailable ({fa_err.__class__.__name__}) — default attention.")
        model = AutoModelForCausalLM.from_pretrained(model_cfg["hf_id"], **load_kwargs)

    model.eval()

    if hasattr(model, "hf_device_map"):
        counts: dict[str, int] = {}
        for dev in model.hf_device_map.values():
            counts[str(dev)] = counts.get(str(dev), 0) + 1
        print(f"  Layer distribution: {counts}")

    return model, tokenizer


def unload_model(model, tokenizer) -> None:
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT / DECODE HELPERS
# HumanEval prompts always begin with a `def` signature, so no stub
# construction is needed — the prompt is always a proper code stub.
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(problem: dict, model_cfg: dict, tokenizer) -> str:
    """
    Build a chat-templated prompt.  For HumanEvalPlus the `prompt` field
    always starts with a function signature (def ...), so no pre-processing
    is required.
    """
    task_text = problem.get("prompt", "")
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
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).rstrip()
    t = strip_code_fences(text).strip()
    for i, line in enumerate(t.split("\n")):
        if line.startswith(("def ", "async def ", "class ", "import ", "from ")):
            return "\n".join(t.split("\n")[i:]).rstrip()
    return t


def _last_function_signature(prompt: str) -> str | None:
    for line in reversed(prompt.splitlines()):
        s = line.strip()
        if s.startswith(("def ", "async def ")):
            return s
    return None


def _prompt_body_indent(prompt: str) -> str:
    for line in reversed(prompt.splitlines()):
        if line.strip():
            return " " * (len(line) - len(line.lstrip(" ")))
    return "    "


def _normalize_body_indentation(lines: list[str], body_indent: str) -> str:
    nonempty = [l for l in lines if l.strip()]
    if not nonempty:
        return ""
    common = min(len(l) - len(l.lstrip(" ")) for l in nonempty)
    out = []
    for l in lines:
        if not l.strip():
            out.append("")
        else:
            out.append(body_indent + (l[common:] if len(l) >= common else l.lstrip(" ")))
    return "\n".join(out).rstrip()


def _strip_repeated_signature_and_docstring(raw: str, prompt: str) -> str | None:
    sig = _last_function_signature(prompt)
    if not sig:
        return None
    indent  = _prompt_body_indent(prompt)
    lines   = raw.split("\n")
    sig_idx = next((i for i, l in enumerate(lines) if l.strip() == sig), None)
    if sig_idx is None:
        return None
    rest = lines[sig_idx + 1:]
    while rest and not rest[0].strip():
        rest = rest[1:]
    if rest and rest[0].strip().startswith(('"""', "'''")):
        q    = '"""' if rest[0].strip().startswith('"""') else "'''"
        rest = rest[1:]
        while rest:
            l = rest[0]; rest = rest[1:]
            if q in l: break
        while rest and not rest[0].strip():
            rest = rest[1:]
    truncated = []
    for l in rest:
        s = l.strip()
        if s in {"```", "<end_of_turn>", "<start_of_turn>model"}:
            break
        if l and not l.startswith((" ", "\t")):
            if s.startswith(("def ", "class ", "if __name__", "print(", "#")):
                break
        truncated.append(l)
    return _normalize_body_indentation(truncated, indent)


def truncate_to_function_continuation(text: str, prompt: str) -> str:
    norm = strip_code_fences(text)
    if prompt in norm:
        norm = norm.split(prompt, 1)[1]
    stripped = _strip_repeated_signature_and_docstring(norm, prompt)
    if stripped:
        return stripped
    lines = []
    for l in norm.split("\n"):
        s = l.strip()
        if s in {"```", "<end_of_turn>", "<start_of_turn>model"}:
            break
        if l and not l.startswith((" ", "\t")):
            if s.startswith(("def ", "class ", "if __name__", "print(", "#")):
                break
        lines.append(l)
    return "\n".join(lines).rstrip()


def decode_generation(raw: str, problem: dict) -> str:
    """
    Decode a model completion for a HumanEvalPlus problem.
    HumanEval prompts always start with 'def ...', so we always complete
    the function stub directly — no plain-English fallback is needed.
    """
    prompt_text = problem.get("prompt", "")
    if prompt_text.strip().startswith(("def ", "async def ")):
        completion = truncate_to_function_continuation(raw, prompt_text)
        return prompt_text + "\n" + completion
    # Fallback for any edge-case problem without a code-stub prompt.
    return _extract_code_from_response(raw)


# ─────────────────────────────────────────────────────────────────────────────
# SANDBOXED EXECUTION  (resource limits + subprocess timeout)
# ─────────────────────────────────────────────────────────────────────────────

def _make_resource_preexec():
    """
    Return a callable for subprocess preexec_fn that applies hard resource
    limits inside the child process (Linux only).
    Prevents fork bombs, infinite loops, and memory leaks from generated code.
    """
    mem = EXEC_MEM_LIMIT
    cpu = EXEC_CPU_LIMIT

    def _set_limits():
        # Limit address-space / resident memory (catches OOM and fork bombs).
        try:
            resource.setrlimit(resource.RLIMIT_AS,  (mem, mem))
        except (ValueError, resource.error):
            pass
        # Limit total CPU seconds (catches infinite loops).
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        except (ValueError, resource.error):
            pass
        # Prevent spawning child processes (additional fork-bomb protection).
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        except (ValueError, resource.error):
            pass

    return _set_limits


def _run_subprocess(code: str, timeout: int = EXEC_TIMEOUT) -> tuple[bool, str]:
    """
    Execute `code` in an isolated subprocess with:
      • wall-clock timeout  (EXEC_TIMEOUT seconds)
      • memory cap          (EXEC_MEM_LIMIT bytes, via RLIMIT_AS)
      • CPU-time cap        (EXEC_CPU_LIMIT seconds, via RLIMIT_CPU)
      • no sub-child spawning (RLIMIT_NPROC = 0)

    All failure modes (syntax error, runtime exception, OOM, timeout,
    infinite loop) return (False, reason) without propagating exceptions.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        fname = f.name

    try:
        proc = subprocess.run(
            [sys.executable, fname],
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_make_resource_preexec(),
        )
        return proc.returncode == 0, proc.stderr
    except subprocess.TimeoutExpired:
        return False, "WALL_CLOCK_TIMEOUT"
    except MemoryError:
        return False, "MEMORY_ERROR"
    except Exception as exc:
        return False, str(exc)
    finally:
        try:
            os.unlink(fname)
        except Exception:
            pass


def evaluate_solution(full_solution: str, problem: dict) -> bool:
    """
    Run `full_solution` against all unit tests for `problem`.
    Returns True only if every test case passes.
    All exceptions are caught; any failure returns False.

    Handles HumanEvalPlus format (test + entry_point + check() call).
    """
    try:
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

        imports    = problem.get("test_imports", [])
        setup_code = problem.get("test_setup_code", "") or ""
        preamble   = (
            ("\n".join(imports) + "\n" if imports else "")
            + (setup_code + "\n" if setup_code.strip() else "")
        )

        for tc in test_cases:
            script = preamble + full_solution + "\n\n" + tc + "\n"
            ok, _err = _run_subprocess(script)
            if not ok:
                return False
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# BATCHED GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _get_primary_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_batch(
    model,
    tokenizer,
    model_cfg: dict,
    items:     list[tuple[dict, int]],   # [(problem, seed), …]
) -> list[str]:
    """
    Generate one completion per item in `items` in a single forward pass.

    All prompts are left-padded to the same length by the tokenizer so the
    batch fits in one model.generate() call.  The seed of the first item is
    used to seed the global RNG before generation; per-item isolation only
    holds perfectly at batch_size=1, but aggregate metrics are unaffected.

    Returns a list of decoded full-solution strings, one per item.
    """
    device  = _get_primary_device(model)
    prompts = [build_prompt(p, model_cfg, tokenizer) for p, _ in items]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

    # Seed from the first item — deterministic at the batch level.
    batch_seed = items[0][1]
    torch.manual_seed(batch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(batch_seed)

    with NoHook(), torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )

    results: list[str] = []
    for i, (problem, _) in enumerate(items):
        # Per-row prompt length from the attention mask (left-padded batch).
        prompt_len = int(inputs["attention_mask"][i].sum().item())
        raw = tokenizer.decode(out_ids[i][prompt_len:], skip_special_tokens=True)
        results.append(decode_generation(raw, problem))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# POOL RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_pool(
    model,
    tokenizer,
    model_cfg:    dict,
    pool:         list[tuple[str, int]],   # [(tid, run_offset), …]
    problem_map:  dict[str, dict],
    pool_type:    str,
    seed_dir:     Path,
    batch_size:   int = DEFAULT_BATCH_SIZE,
) -> dict:
    """
    Generate and evaluate every (problem_id, run_offset) entry in `pool`.

    Entries are processed in batches of `batch_size`.  Each batch is tokenized
    together and sent through model.generate() in one call, then each output is
    decoded and evaluated independently.

    Every generation attempt is appended to a .jsonl file immediately so that
    partial results survive crashes.  Record schema per line:
      {"problem_id": str, "pool_type": str, "seed": int,
       "generated_code": str, "passed": bool}

    Returns pool-wise run-level success metrics.
    """
    n_total    = len(pool)
    n_passed   = 0
    jsonl_path = seed_dir / f"{pool_type}_generations.jsonl"

    with (
        tqdm(total=n_total, desc=f"  [{pool_type}] generating", unit="run") as pbar,
        open(jsonl_path, "a", encoding="utf-8") as jf,
    ):
        for batch_start in range(0, n_total, batch_size):
            batch = pool[batch_start: batch_start + batch_size]

            # Build (problem, seed) pairs for this batch.
            items: list[tuple[dict, int]] = []
            seeds: list[int]              = []
            for tid, run_offset in batch:
                seed = _deterministic_seed(tid, run_offset)
                items.append((problem_map[tid], seed))
                seeds.append(seed)

            # Single batched forward pass.
            try:
                full_solutions = generate_batch(model, tokenizer, model_cfg, items)
            except Exception as exc:
                tqdm.write(f"  [BATCH GEN WARN] batch@{batch_start}: {exc}")
                full_solutions = [""] * len(batch)

            # Evaluate and persist each result.
            for (tid, _), full_solution, seed in zip(batch, full_solutions, seeds):
                try:
                    passed = evaluate_solution(full_solution, problem_map[tid])
                except Exception as exc:
                    tqdm.write(f"  [EVAL WARN] tid={tid} seed={seed}: {exc}")
                    passed = False

                n_passed += int(passed)

                record = {
                    "problem_id":     tid,
                    "pool_type":      pool_type,
                    "seed":           seed,
                    "generated_code": full_solution,
                    "passed":         passed,
                }
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")
                jf.flush()

            pbar.update(len(batch))

    success_rate = n_passed / n_total if n_total > 0 else 0.0
    return {
        "n_passed":                         n_passed,
        "n_total":                          n_total,
        "pool_wise_run_level_success_rate":  round(success_rate, 6),
        "pool_wise_run_level_success_pct":   round(success_rate * 100, 2),
        "jsonl_path":                        str(jsonl_path),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BASELINE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline(
    model_key:  str,
    out_root:   Path,
    batch_size: int,
    run_seed:   int = 0,
    split_json: Path | None = None,
    force:      bool = False,
) -> dict:
    """
    Full distribution-level baseline evaluation pipeline for one model.

    The `run_seed` parameter controls the pool entry offset
    (run_offset = run_seed * 10 + i), matching the convention in
    pipeline_humaneval.py so that baseline and steering results are seeded
    from the same deterministic sequence and Δ metrics are meaningful.

    `split_json` must point to the split.json written by pipeline_humaneval.py
    so that the test-problem IDs are byte-for-byte identical.  If not
    supplied, the script auto-detects the most recent pipeline_humaneval.py
    run in PIPELINE_OUT_ROOT; if none is found it falls back to recomputing
    the split from acts_root (same SPLIT_SEED=42, 60/20/20 logic).

    Steps
    -----
    1. Initialise all global RNGs (master seed = SPLIT_SEED ^ run_seed).
    2. Load test split from pipeline_humaneval.py split.json (or recompute).
    3. Load HumanEvalPlus dataset → build problem_map.
    4. Build wrong_pool and right_pool from historical sampling weights.
    5. Run unsteered single-sample generation for each pool entry.
    6. Compute pool-wise run-level success rate and save results.
    """
    model_cfg = MODEL_REGISTRY[model_key]
    acts_dir  = ACTS_ROOTS[model_key]
    display   = model_cfg["display"]

    if not Path(acts_dir).exists():
        print(f"  [ERROR] Acts dir not found: {acts_dir} — skipping {display}.")
        return {}

    # ── Output directories ───────────────────────────────────────────────────
    existing = sorted(out_root.glob(f"{model_key}_*"))
    if existing:
        out_dir = existing[-1]
        print(f"  [INFO] Reusing output dir: {out_dir}")
    else:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = out_root / f"{model_key}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

    seed_dir  = out_dir / f"seed_{run_seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    done_flag = seed_dir / "pool_baselines.json"

    if done_flag.exists() and not force:
        print(f"  [BASELINE] Resuming — pool_baselines.json found for seed {run_seed}.")
        with open(done_flag) as f:
            return json.load(f)
    if done_flag.exists() and force:
        print(f"  [BASELINE] --force: removing stale pool_baselines.json for seed {run_seed}.")
        done_flag.unlink()

    # Initialise all global RNG states before anything else.
    master_seed = SPLIT_SEED ^ run_seed
    set_global_seeds(master_seed)
    print(f"  [RNG] Global seeds set to {master_seed} "
          f"(SPLIT_SEED={SPLIT_SEED} XOR run_seed={run_seed})")

    print(f"\n{'#'*72}")
    print(f"  MODEL   : {display}")
    print(f"  Acts    : {acts_dir}")
    print(f"  Out     : {seed_dir}")
    print(f"  Seed    : {run_seed}  (master_seed={master_seed})")
    print(f"  NOTE    : NO steering hooks — pure base model evaluation")
    print(f"{'#'*72}")

    # ── Step 1: Load (or recompute) test split ───────────────────────────────
    print(f"\n{'='*60}\n  STEP 1 — LOAD TEST SPLIT\n{'='*60}")
    acts_root = _find_acts_root(acts_dir)

    # Prefer the split.json written by pipeline_humaneval.py so that test IDs
    # are guaranteed identical to the steering experiment.
    resolved_split_json = split_json
    if resolved_split_json is None:
        resolved_split_json = _auto_detect_split_json(model_key)

    if resolved_split_json is not None and resolved_split_json.exists():
        print(f"  [SPLIT] Loading from pipeline_humaneval.py: {resolved_split_json}")
        test_ids, historical_data = load_split_from_json(resolved_split_json)
        print(f"  [SPLIT] Source: split.json  ({len(test_ids)} test problems)")
    else:
        if resolved_split_json is not None:
            print(f"  [SPLIT] split.json not found at {resolved_split_json} — recomputing.")
        else:
            print(f"  [SPLIT] No split.json found in {PIPELINE_OUT_ROOT} — recomputing.")
        test_ids, historical_data = build_test_historical_data(acts_root)
        print(f"  [SPLIT] Source: recomputed from acts_root  ({len(test_ids)} test problems)")

    # wrong_count / right_count values are sampling allocation weights only —
    # they are NOT evaluation labels for the new generations.
    total_wrong_runs = sum(v["wrong_count"] for v in historical_data.values())
    total_right_runs = sum(v["right_count"] for v in historical_data.values())

    print(f"  Test problems                    : {len(test_ids)}")
    print(f"  Total wrong-pool allocation      : {total_wrong_runs} runs")
    print(f"  Total right-pool allocation      : {total_right_runs} runs")
    print(f"  (Counts are sampling weights only — not evaluation labels)")

    if total_wrong_runs == 0 or total_right_runs == 0:
        print("  [ERROR] One or both pools are empty — cannot compute baseline.")
        return {}

    # ── Step 2: Load dataset and build problem_map ───────────────────────────
    print(f"\n{'='*60}\n  STEP 2 — LOAD HUMANEVALPLUS DATASET\n{'='*60}")
    problems_list = load_humanevalplus()
    stored_ids    = {
        d.name for d in acts_root.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS
    }
    problem_map = build_problem_map(problems_list, stored_ids)

    test_problem_map: dict[str, dict] = {}
    unresolved: list[str] = []
    for tid in test_ids:
        if tid in problem_map:
            test_problem_map[tid] = problem_map[tid]
        else:
            unresolved.append(tid)

    if unresolved:
        print(f"  [WARN] {len(unresolved)} test problems not resolved in dataset "
              f"(first 5: {unresolved[:5]})")
    print(f"  Resolved test problems: {len(test_problem_map)} / {len(test_ids)}")

    # ── Step 3: Build flat run pools ─────────────────────────────────────────
    print(f"\n{'='*60}\n  STEP 3 — BUILD RUN POOLS\n{'='*60}")

    # run_offset = run_seed * 10 + i matches pipeline_humaneval.py so that
    # the MD5-derived seeds for baseline and steering are drawn from the same
    # deterministic sequence, enabling direct Δ comparisons.
    # The historical wrong_count / right_count values are ONLY used to set
    # pool sizes (sampling allocation weights); not as evaluation targets.
    wrong_pool: list[tuple[str, int]] = []
    right_pool: list[tuple[str, int]] = []
    for tid in sorted(test_problem_map):
        hist = historical_data.get(tid, {})
        for i in range(hist.get("wrong_count", 0)):
            wrong_pool.append((tid, run_seed * 10 + i))
        for i in range(hist.get("right_count", 0)):
            right_pool.append((tid, run_seed * 10 + i))

    print(f"  Wrong pool : {len(wrong_pool)} generation slots  "
          f"(derived from historical wrong_count — allocation weight only)")
    print(f"  Right pool : {len(right_pool)} generation slots  "
          f"(derived from historical right_count — allocation weight only)")

    # ── Step 4: Load model and run both pools ────────────────────────────────
    print(f"\n{'='*60}\n  STEP 4 — UNSTEERED GENERATION (no hooks)\n{'='*60}")
    print(f"  Batch size : {batch_size}  "
          f"{'(per-sample isolation — recommended)' if batch_size == 1 else '(batched — reduces reproducibility)'}")

    model, tokenizer = load_model(model_cfg)

    print("\n  --- Wrong Pool (sampling size from historical wrong_count) ---")
    wrong_result = run_pool(
        model, tokenizer, model_cfg,
        wrong_pool, test_problem_map,
        pool_type="wrong",
        seed_dir=seed_dir,
        batch_size=batch_size,
    )

    print("\n  --- Right Pool (sampling size from historical right_count) ---")
    right_result = run_pool(
        model, tokenizer, model_cfg,
        right_pool, test_problem_map,
        pool_type="right",
        seed_dir=seed_dir,
        batch_size=batch_size,
    )

    unload_model(model, tokenizer)

    # ── Step 5: Compute metrics and save ─────────────────────────────────────
    # Metric keys use "pool_wise_run_level_success_rate" throughout.
    # This is distinct from the canonical pass@k estimator: it is the raw
    # fraction of independently generated attempts that passed unit tests,
    # pooled flat across all problems in the respective pool.
    wrong_rate = wrong_result["pool_wise_run_level_success_rate"]
    right_rate = right_result["pool_wise_run_level_success_rate"]

    results = {
        "model":             display,
        "model_key":         model_key,
        "run_seed":          run_seed,
        "master_seed":       master_seed,
        "generated_at":      datetime.now().isoformat(),
        "n_test_problems":   len(test_problem_map),
        "evaluation_type":   "distribution_level_baseline",
        "metric_definition": (
            "Pool-wise run-level success rate: fraction of independently "
            "generated attempts that passed all unit tests. "
            "NOT the canonical pass@k estimator."
        ),
        "note_on_counts": (
            "historical wrong_count / right_count are sampling allocation "
            "weights only — they are not evaluation labels."
        ),
        # ── Wrong pool ───────────────────────────────────────────────────────
        "wrong_pool": {
            "n_total":  wrong_result["n_total"],
            "n_passed": wrong_result["n_passed"],
            "pool_wise_run_level_success_rate": wrong_rate,
            "pool_wise_run_level_success_pct":  wrong_result["pool_wise_run_level_success_pct"],
            "generations_jsonl": wrong_result["jsonl_path"],
        },
        # ── Right pool ───────────────────────────────────────────────────────
        "right_pool": {
            "n_total":  right_result["n_total"],
            "n_passed": right_result["n_passed"],
            "pool_wise_run_level_success_rate": right_rate,
            "pool_wise_run_level_success_pct":  right_result["pool_wise_run_level_success_pct"],
            "generations_jsonl": right_result["jsonl_path"],
        },
        # ── Flat keys for steering-script compatibility ───────────────────────
        # These names match what pipeline_humaneval.py reads from pool_baselines.json.
        "baseline_fwd_p1": round(wrong_rate, 6),
        "baseline_bwd_p1": round(right_rate, 6),
        "n_wrong_runs":    wrong_result["n_total"],
        "n_right_runs":    right_result["n_total"],
    }

    with open(done_flag, "w") as f:
        json.dump(results, f, indent=2)

    # Human-readable report
    report_lines = [
        "=" * 72,
        f"  DISTRIBUTION-LEVEL BASELINE REPORT — {display}  (HumanEvalPlus)",
        f"  Generated : {results['generated_at']}",
        f"  Run seed  : {run_seed}  (master_seed={master_seed})",
        "",
        "  EVALUATION TYPE: Distribution-level baseline with fresh sampling.",
        "  Metric: Pool-wise run-level success rate (NOT canonical pass@k).",
        "  No activation hooks, steering vectors, or alpha modifications.",
        "",
        "  NOTE ON POOL SIZES:",
        "    wrong_count / right_count from the activation store are used",
        "    ONLY as sampling allocation weights (how many completions to",
        "    generate per problem).  They are NOT evaluation labels.",
        "    Every completion is evaluated independently by unit-test execution.",
        "=" * 72,
        "",
        "  Wrong Pool  (generation slots from historical wrong_count)",
        f"  {'Total generation attempts':<36}: {wrong_result['n_total']}",
        f"  {'Passed unit tests':<36}: {wrong_result['n_passed']}",
        f"  {'Pool-wise run-level success rate':<36}: {wrong_rate * 100:.2f}%",
        "",
        "  Right Pool  (generation slots from historical right_count)",
        f"  {'Total generation attempts':<36}: {right_result['n_total']}",
        f"  {'Passed unit tests':<36}: {right_result['n_passed']}",
        f"  {'Pool-wise run-level success rate':<36}: {right_rate * 100:.2f}%",
        "",
        "  Formula:",
        f"    pool_wise_success_wrong = {wrong_result['n_passed']} / "
        f"{wrong_result['n_total']} = {wrong_rate * 100:.4f}%",
        f"    pool_wise_success_right = {right_result['n_passed']} / "
        f"{right_result['n_total']} = {right_rate * 100:.4f}%",
        "",
        "  Generations saved to:",
        f"    {wrong_result['jsonl_path']}",
        f"    {right_result['jsonl_path']}",
        "",
        "=" * 72,
    ]
    report_text = "\n".join(report_lines)
    print(f"\n{report_text}")
    (seed_dir / "baseline_report.txt").write_text(report_text, encoding="utf-8")
    print(f"\n  [BASELINE] → {done_flag}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "HumanEvalPlus distribution-level baseline evaluation. "
            "Computes pool-wise run-level success rates for wrong and right pools "
            "derived from the test split of stored activation data."
        )
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
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=(
            f"Generation batch size (default: {DEFAULT_BATCH_SIZE}). "
            "batch_size=1 guarantees per-sample RNG independence. "
            "Larger values improve throughput but couple stochastic sampling."
        ),
    )
    parser.add_argument(
        "--run-seed", type=int, default=None,
        help="Run a single seed (0, 1, or 2) instead of all three.",
    )
    parser.add_argument(
        "--split-json", default=None,
        help=(
            "Path to split.json produced by pipeline_humaneval.py. "
            "When omitted the script uses SPLIT_JSONS or auto-detects the "
            f"most recent run in {PIPELINE_OUT_ROOT}."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore cached pool_baselines.json and re-run generation from scratch.",
    )
    args     = parser.parse_args()
    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    split_json = Path(args.split_json).expanduser() if args.split_json else None

    models = (
        [args.model] if args.model
        else ["qwen-coder-1.5b-instruct", "qwen-coder-7b-instruct"]
    )
    seeds = [args.run_seed] if args.run_seed is not None else [0, 1, 2]

    # Outer loop: model (1.5B then 7B).
    # Inner loop: seed 0 → 1 → 2 for each model before moving on.
    all_results: dict[str, dict] = {}   # keyed by f"{model_key}_seed{s}"
    for mk in models:
        for s in seeds:
            key = f"{mk}_seed{s}"
            print(f"\n{'#'*72}")
            print(f"  PIPELINE  model={MODEL_REGISTRY[mk]['display']}  seed={s}")
            print(f"{'#'*72}")
            try:
                result = run_baseline(
                    model_key=mk,
                    out_root=out_root,
                    batch_size=args.batch_size,
                    run_seed=s,
                    split_json=split_json,
                    force=args.force,
                )
                all_results[key] = result
            except Exception as exc:
                import traceback
                print(f"\n[ERROR] {MODEL_REGISTRY[mk]['display']} seed={s} failed: {exc}")
                traceback.print_exc()

    # Cross-model / cross-seed summary
    print(f"\n{'='*72}")
    print("  FULL SUMMARY  (pool-wise run-level success rate, all seeds)")
    print(f"  {'Model + seed':<36}  {'Wrong Pool':>12}  {'Right Pool':>12}")
    print("  " + "-" * 64)
    for key, res in all_results.items():
        if res:
            wp = res.get("wrong_pool", {}).get("pool_wise_run_level_success_pct", float("nan"))
            rp = res.get("right_pool", {}).get("pool_wise_run_level_success_pct", float("nan"))
            print(f"  {key:<36}  {wp:>11.2f}%  {rp:>11.2f}%")
    print(f"{'='*72}")

    print(f"\n  ALL DONE  →  {out_root}\n")


if __name__ == "__main__":
    main()
