# Repository guidance

This repository is a compact, private handoff of the LG Aimers experiment history.

- Preserve negative, failed, and invalidated results alongside successful results.
- Never commit competition data, credentials, prediction arrays, trained models, or submission ZIP files.
- Use only the official files placed locally under `open/data/` when reproducing experiments.
- Do not automate DACON submission.
- Do not weaken `submission/verify_submission.py` or evaluation-row independence checks.
- When results are rerun, use a new experiment ID or archive the previous result first.
- Treat `catalog/experiments.csv` and `experiments/EXPERIMENT_REGISTRY.md` as indexes; the original JSON/CSV evidence remains authoritative.

The original project-specific agent guide is retained at
[`docs/ORIGINAL_AGENT_GUIDE.md`](docs/ORIGINAL_AGENT_GUIDE.md) for provenance.

