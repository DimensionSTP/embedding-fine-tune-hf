# Training Modes Contract

## Modes

- `mode=train`: run embedding fine-tuning pipeline.

## Inputs

- dataset configured under `configs/dataset/*`
- model/trainer configs under `configs/*`

## Outputs

- checkpoints: `${CONNECTED_DIR}/.../checkpoints` (config-dependent)
- logs: `logs/` and optional `wandb` outputs

## Notes

- Keep CLI overrides synchronized with config keys.
