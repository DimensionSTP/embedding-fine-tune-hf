# Training Modes Contract

## Scope

Contract for `embedding-fine-tune-hf` training execution.

## Entry Point

- Command: `python main.py mode=train`
- Script equivalent: `bash scripts/train/train.sh`

## Required Inputs

Environment variables:

- `PROJECT_DIR`
- `CONNECTED_DIR`
- `DEVICES`
- `HF_HOME`
- `USER_NAME`

Config expectations:

- `mode` must be `train`
- Optional runtime knobs include `is_sft`, `is_preprocessed`, `left_padding`, `is_quantized`, `is_peft`, `strategy`, `max_length`

## Output Contract

- Training must produce checkpoint/artifact outputs at config-defined directories.
- Log output must contain selected model/data/runtime settings for reproducibility.

## Failure Contract

- Unsupported mode must raise a clear runtime error.
- Missing required environment variables must fail fast before long training loops.

## Release/Docs Sync Rule

If mode names, required env vars, or output path conventions change, update together:

- `README.md`
- `USAGE_GUIDE.md`
- this contract file
- `CHANGELOG.md`
