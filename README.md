# Discovering and Steering Code Correctness through Residual Stream Directions in LLMs
Code repository for EMNLP-2026 Submission

## Repository Structure

```
emnlp_codellm/
├── data/
│   ├── humanevalplus/
│   │   ├── qwen-2.5-coder-1.5b-instruct/full_dataset_runs/
│   │   └── qwen-2.5-coder-7b-instruct/full_dataset_runs/
│   ├── mbppplus/
│   │   ├── qwen-2.5-coder-1.5b-instruct/full_dataset_runs/
│   │   └── qwen-2.5-coder-7b-instruct/full_dataset_runs/
│   └── bigcodebench/
│       ├── qwen-coder-1.5b-instruct/full_dataset_runs/   
│       └── qwen-coder-7b-instruct/full_dataset_runs/
├── scripts/
│   ├── pipeline_humanevalplus.py
│   ├── pipeline_mbppplus.py
│   ├── pipeline_bigcodebench.py
│   ├── baseline_humanevalplus.py
│   ├── baseline_mbppplus.py
│   └── baseline_bigcodebench.py
└── requirements.txt
```

**Activation data layout** (per dataset):
```
data/<dataset>/<model>/<base_dir>/<problem_id>/right/run1/layer_XX.h5
                                               <problem_id>/wrong/run2/layer_XX.h5
```

## Setup

```bash
pip install -r requirements.txt
```
## Pipeline Overview

Each pipeline script runs four stages sequentially for each model:

| Stage | Description |
|-------|-------------|
| 1. Split | 60/20/20 problem-level split (SPLIT_SEED=42) into train/val/test |
| 2. Probing | Layer-wise logistic probe on contrastive-TRAIN; ranked by AUROC on contrastive-VAL |
| 3. Symmetry + L\*-score | Computed for reference; top layers selected by AUROC_val |
| 4. Steering | Forward addition, backward subtraction, direction ablation on extended test pool |


## Scripts

### Pipeline scripts

```bash
python scripts/pipeline_humanevalplus.py
python scripts/pipeline_humanevalplus.py --model qwen-coder-1.5b-instruct
python scripts/pipeline_humanevalplus.py --out-root /my/output/dir
python scripts/pipeline_humanevalplus.py --batch-size 64
```

Same interface for `pipeline_mbppplus.py` and `pipeline_bigcodebench.py`.

Outputs per model (written under `scripts/pipeline_results_*/`):
- `split.json` — 60/20/20 split with per-problem run counts and group assignments
- `probing/probe_analysis.json` — per-layer AUROC and direction vectors
- `probing/directions/layer_XX.npy` — unit steering vectors
- `symmetry/symmetry_results.json` — symmetry and l\*-scores
- `steering/seed_{0,1,2}/steering_results.json` — per-layer, per-alpha results

### Baseline scripts

```bash
python scripts/baseline_humanevalplus.py
python scripts/baseline_humanevalplus.py --model qwen-coder-7b-instruct
python scripts/baseline_humanevalplus.py --run-seed 1
python scripts/baseline_humanevalplus.py --split-json /path/to/split.json
```

Same interface for `baseline_mbppplus.py` and `baseline_bigcodebench.py`.

Outputs per seed (written under `scripts/baseline_results_*/`):
- `seed_{N}/pool_baselines.json` — pool-wise run-level success rates
- `seed_{N}/wrong_generations.jsonl` — every wrong-pool generation
- `seed_{N}/right_generations.jsonl` — every right-pool generation

## Models

| Key | HuggingFace ID |
|-----|----------------|
| `qwen-coder-1.5b-instruct` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| `qwen-coder-7b-instruct` | `Qwen/Qwen2.5-Coder-7B-Instruct` |

## Datasets

| Script | Dataset |
|--------|---------|
| `*_humanevalplus.py` | HumanEvalPlus (evalplus pkg / HuggingFace) |
| `*_mbppplus.py` | MBPP+ (evalplus pkg / google-research-datasets/mbpp) |
| `*_bigcodebench.py` | BigCodeBench v0.1.4 (bigcode/bigcodebench) |

## Steering Interventions

| Type | Operation | Pool | Expected delta |
|------|-----------|------|----------------|
| Forward addition | h ← h + α·d̂ | wrong runs | positive (recovery) |
| Backward subtraction | h ← h + α·d̂, α < 0 | right runs | negative (degradation) |
| Direction ablation | h ← h − (h·d̂)·d̂ | right runs | negative (degradation) |

Alpha (steering strength): forward `[0.5, 1, 2, 5, 10]`, backward `[-0.5, -1, -2, -5, -10]`.
