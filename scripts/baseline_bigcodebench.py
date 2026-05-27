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

_tv_stub       = _types.ModuleType("torchvision")
_tv_ext        = _types.ModuleType("torchvision.extension")
_tv_io         = _types.ModuleType("torchvision.io")
_tv_transforms = _types.ModuleType("torchvision.transforms")
_tv_stub.__spec__       = _imachinery.ModuleSpec("torchvision",            loader=None, is_package=True)
_tv_ext.__spec__        = _imachinery.ModuleSpec("torchvision.extension",  loader=None)
_tv_io.__spec__         = _imachinery.ModuleSpec("torchvision.io",         loader=None)
_tv_transforms.__spec__ = _imachinery.ModuleSpec("torchvision.transforms", loader=None)
_tv_stub.__path__ = []
_tv_ext._has_ops      = lambda: False
_tv_io.ImageReadMode  = type("ImageReadMode", (), {})()
_tv_io.decode_image   = lambda *a, **kw: None
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

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import transformers.utils.import_utils as _tui
    _tui._torchvision_available = False
    del _tui
except Exception:
    pass

TRAIN_RATIO    = 0.60
VAL_RATIO      = 0.20
SPLIT_SEED     = 42

TEMPERATURE    = 0.8
TOP_P          = 0.95
MAX_NEW_TOKENS = 512

EXEC_TIMEOUT   = 10
EXEC_MEM_LIMIT = 4 * 1024 ** 3
EXEC_CPU_LIMIT = 30

DEFAULT_BATCH_SIZE = 128

EXPECTED_RUNS_PER_PROBLEM = 5

SKIP_DIRS = {
    "probing", "steering", "plots", "pipeline2_results",
    "pipeline2_probing", "all", "contrastive", "comparison",
    "probing_5fold_cv",
}

_DATA_ROOT = Path(__file__).parent.parent / "data" / "bigcodebench"
ACTS_ROOTS = {
    "qwen-coder-1.5b-instruct": str(_DATA_ROOT / "qwen-coder-1.5b-instruct" / "all"),
    "qwen-coder-7b-instruct":   str(_DATA_ROOT / "qwen-coder-7b-instruct" / "all"),
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

OUT_ROOT          = Path(__file__).parent / "baseline_results_bigcodebench"
PIPELINE_OUT_ROOT = Path(__file__).parent / "pipeline_results_bigcodebench"

SPLIT_JSONS: dict[str, Path] = {}


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _deterministic_seed(tid: str, run_offset: int) -> int:
    base = int(hashlib.md5(tid.encode()).hexdigest(), 16) % (2 ** 31)
    return base + run_offset * 1000


class NoHook:
    def __enter__(self): return self
    def __exit__(self, *_): pass


def load_bigcodebench() -> list[dict]:
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


def _auto_detect_split_json(model_key: str) -> Path | None:
    if model_key in SPLIT_JSONS:
        return SPLIT_JSONS[model_key]
    if not PIPELINE_OUT_ROOT.is_dir():
        return None
    candidates = sorted(PIPELINE_OUT_ROOT.glob(f"{model_key}_*/split.json"))
    return candidates[-1] if candidates else None


def load_split_from_json(
    split_path: Path,
) -> tuple[list[str], dict[str, dict]]:
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


def scan_runs(acts_root: Path) -> dict[str, dict]:
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


def load_model(model_cfg: dict):
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


def build_prompt(problem: dict, model_cfg: dict, tokenizer) -> str:
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
    return _extract_code_from_response(raw)


def _make_resource_preexec():
    mem = EXEC_MEM_LIMIT
    cpu = EXEC_CPU_LIMIT

    def _set_limits():
        try:
            resource.setrlimit(resource.RLIMIT_AS,  (mem, mem))
        except (ValueError, resource.error):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        except (ValueError, resource.error):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        except (ValueError, resource.error):
            pass

    return _set_limits


def _run_subprocess(code: str, timeout: int = EXEC_TIMEOUT) -> tuple[bool, str]:
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


def _get_primary_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_batch(
    model,
    tokenizer,
    model_cfg: dict,
    items:     list[tuple[dict, int]],
) -> list[str]:
    device  = _get_primary_device(model)
    prompts = [build_prompt(p, model_cfg, tokenizer) for p, _ in items]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

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
        prompt_len = int(inputs["attention_mask"][i].sum().item())
        raw = tokenizer.decode(out_ids[i][prompt_len:], skip_special_tokens=True)
        results.append(decode_generation(raw, problem))
    return results


def run_pool(
    model,
    tokenizer,
    model_cfg:    dict,
    pool:         list[tuple[str, int]],
    problem_map:  dict[str, dict],
    pool_type:    str,
    seed_dir:     Path,
    batch_size:   int = DEFAULT_BATCH_SIZE,
) -> dict:
    n_total    = len(pool)
    n_passed   = 0
    jsonl_path = seed_dir / f"{pool_type}_generations.jsonl"

    with (
        tqdm(total=n_total, desc=f"  [{pool_type}] generating", unit="run") as pbar,
        open(jsonl_path, "a", encoding="utf-8") as jf,
    ):
        for batch_start in range(0, n_total, batch_size):
            batch = pool[batch_start: batch_start + batch_size]

            items: list[tuple[dict, int]] = []
            seeds: list[int]              = []
            for tid, run_offset in batch:
                seed = _deterministic_seed(tid, run_offset)
                items.append((problem_map[tid], seed))
                seeds.append(seed)

            try:
                full_solutions = generate_batch(model, tokenizer, model_cfg, items)
            except Exception as exc:
                tqdm.write(f"  [BATCH GEN WARN] batch@{batch_start}: {exc}")
                full_solutions = [""] * len(batch)

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


def run_baseline(
    model_key:  str,
    out_root:   Path,
    batch_size: int,
    run_seed:   int = 0,
    split_json: Path | None = None,
    force:      bool = False,
) -> dict:
    model_cfg = MODEL_REGISTRY[model_key]
    acts_dir  = ACTS_ROOTS[model_key]
    display   = model_cfg["display"]

    if not Path(acts_dir).exists():
        print(f"  [ERROR] Acts dir not found: {acts_dir} — skipping {display}.")
        return {}

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

    print(f"\n{'='*60}\n  STEP 1 — LOAD TEST SPLIT\n{'='*60}")
    acts_root = _find_acts_root(acts_dir)

    resolved_split_json = split_json
    if resolved_split_json is None:
        resolved_split_json = _auto_detect_split_json(model_key)

    if resolved_split_json is not None and resolved_split_json.exists():
        print(f"  [SPLIT] Loading from pipeline_bigcodebench.py: {resolved_split_json}")
        test_ids, historical_data = load_split_from_json(resolved_split_json)
        print(f"  [SPLIT] Source: split.json  ({len(test_ids)} test problems)")
    else:
        if resolved_split_json is not None:
            print(f"  [SPLIT] split.json not found at {resolved_split_json} — recomputing.")
        else:
            print(f"  [SPLIT] No split.json found in {PIPELINE_OUT_ROOT} — recomputing.")
        test_ids, historical_data = build_test_historical_data(acts_root)
        print(f"  [SPLIT] Source: recomputed from acts_root  ({len(test_ids)} test problems)")

    total_wrong_runs = sum(v["wrong_count"] for v in historical_data.values())
    total_right_runs = sum(v["right_count"] for v in historical_data.values())

    print(f"  Test problems                    : {len(test_ids)}")
    print(f"  Total wrong-pool allocation      : {total_wrong_runs} runs")
    print(f"  Total right-pool allocation      : {total_right_runs} runs")
    print(f"  (Counts are sampling weights only — not evaluation labels)")

    if total_wrong_runs == 0 or total_right_runs == 0:
        print("  [ERROR] One or both pools are empty — cannot compute baseline.")
        return {}

    print(f"\n{'='*60}\n  STEP 2 — LOAD BIGCODEBENCH DATASET\n{'='*60}")
    problems_list = load_bigcodebench()
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

    print(f"\n{'='*60}\n  STEP 3 — BUILD RUN POOLS\n{'='*60}")

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
        "wrong_pool": {
            "n_total":  wrong_result["n_total"],
            "n_passed": wrong_result["n_passed"],
            "pool_wise_run_level_success_rate": wrong_rate,
            "pool_wise_run_level_success_pct":  wrong_result["pool_wise_run_level_success_pct"],
            "generations_jsonl": wrong_result["jsonl_path"],
        },
        "right_pool": {
            "n_total":  right_result["n_total"],
            "n_passed": right_result["n_passed"],
            "pool_wise_run_level_success_rate": right_rate,
            "pool_wise_run_level_success_pct":  right_result["pool_wise_run_level_success_pct"],
            "generations_jsonl": right_result["jsonl_path"],
        },
        "baseline_fwd_p1": round(wrong_rate, 6),
        "baseline_bwd_p1": round(right_rate, 6),
        "n_wrong_runs":    wrong_result["n_total"],
        "n_right_runs":    right_result["n_total"],
    }

    with open(done_flag, "w") as f:
        json.dump(results, f, indent=2)

    report_lines = [
        "=" * 72,
        f"  DISTRIBUTION-LEVEL BASELINE REPORT — {display}  (BigCodeBench)",
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "BigCodeBench distribution-level baseline evaluation. "
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
            "Path to split.json produced by pipeline_bigcodebench.py. "
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

    all_results: dict[str, dict] = {}
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
