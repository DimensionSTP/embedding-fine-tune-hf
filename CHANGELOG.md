# Changelog

All notable changes to this repository are documented in this file.

## [v1.0.7] - 2026-04-06

- Sync `packages.txt` and `requirements.txt` with the current `joshpp` runtime freeze after compatibility-driven dependency adjustments.
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
