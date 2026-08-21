# Usage Guide

## Purpose

Operational runbook for `embedding-fine-tune-hf` covering environment setup, training execution, and validation.

## Prerequisites

- Python 3.12 recommended
- CUDA-ready environment when using GPU training
- `.env` variables configured from `.env.example`

## Install

```bash
python -m pip install uv==0.10.12
uv pip install --torch-backend=cu129 -r requirements.txt
```

or

```bash
uv pip install --torch-backend=cu129 .
uv pip install --torch-backend=cu129 -e .
```

## Environment Variables

Set at least:

- `PROJECT_DIR`
- `CONNECTED_DIR`
- `DEVICES`
- `HF_HOME`
- `USER_NAME`

## Main Execution Mode

`main.py` supports:

- `train`

Run:

```bash
python main.py mode=train
```

Equivalent script:

```bash
bash scripts/train/train.sh
```

## Common Runtime Options

- `is_sft`, `is_preprocessed`, `left_padding`
- `is_quantized`, `is_peft`
- `strategy=deepspeed` for multi-GPU full fine-tuning
- `max_length`, `upload_user`, `model_type`

## Validation Checklist

1. `python main.py mode=train` starts without config resolution error.
2. Output artifacts are generated under configured output paths.
3. Run logs include expected model/data configuration.
4. Changes are reflected in `CHANGELOG.md` before release.
