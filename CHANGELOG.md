# Changelog

All notable changes to this repository are documented in this file.

## [v1.0.10] - 2026-06-12

- Sync `packages.txt` with the current validated runtime freeze after the transformer and runtime package refresh.
- Update embedding fine-tuning `requirements.txt` pins for `huggingface-hub==1.19.0` and `transformers==5.11.0` to match the validated environment.
- Record refreshed runtime packages including `trl==1.6.0`, `vllm==0.19.0`, `liger_kernel==0.8.0`, and MLflow-related packages in the full environment snapshot for reproducible setup.

## [v1.0.9] - 2026-05-14

- Sync `packages.txt` with the current validated runtime freeze after the distributed-runtime package refresh.
- Record `mpi4py==4.1.1` and `mpich==5.0.1` in the full environment snapshot for reproducible MPI-capable runtime reconstruction.
- Keep direct embedding fine-tuning dependency manifests unchanged because the new packages are not imported by the project code or required as direct install dependencies.

## [v1.0.8] - 2026-05-12

- Sync `packages.txt` with the current validated runtime freeze after the dependency refresh that includes `sentence-transformers==5.4.1`, `transformers==5.8.0`, and related framework pins.
- Update embedding fine-tuning `requirements.txt` pins for `huggingface-hub`, `numpy`, `sentence-transformers`, and `transformers` to match the validated environment.
- Keep install and reproducibility manifests aligned for embedding training workflows under the refreshed local runtime baseline.

## [v1.0.7] - 2026-04-06

- Sync `packages.txt` and `requirements.txt` with the current validated runtime freeze after compatibility-driven dependency adjustments.
- Update embedding runtime dependency pins to keep install manifests aligned with the latest validated local environment.

## [v1.0.6] - 2026-04-03

- Sync `packages.txt` and `requirements.txt` with the validated environment freeze.
- Update runtime dependency pins for the embedding fine-tuning pipeline to match the current validated baseline.

## [v1.0.5] - 2026-03-24

- Remove stale references to deleted Korean documentation files from the training contract.
- Align release/documentation sync guidance with the current English documentation set.
- Prepare a patch release for documentation-contract consistency after KO document removal.

## [v1.0.4] - 2026-03-23

- Remove outdated Korean docs (`*_ko.md`) to prevent EN/KO content drift; keep EN docs as canonical source for now.
- Revert Hydra entry-point defaults to prior runtime behavior for W&B local directory compatibility.
- Synchronize Korean documentation formatting and heading consistency.
- Update dependency snapshots (`packages.txt`, `requirements.txt`) for environment reproducibility.

## [2026-03-19]

- Add collaboration metadata: `CONTRIBUTING.md`, `SECURITY.md`, `CODEOWNERS`.
- Add GitHub templates: PR template, issue templates, and docs/link CI workflow.
- Add `.env.example` template for onboarding and local setup.
- Add packaging metadata via `pyproject.toml`.
- Add repository execution/output contract docs.
- Add README installation guidance for pyproject-based installs.
- Add Python compile smoke workflow for CI baseline checks.

## [v1.0.3] - 2026-03-19

- Packaging and execution-contract baseline update with compile-check CI and documentation hardening.
- Refer to the GitHub Release note for full details and migration context.
