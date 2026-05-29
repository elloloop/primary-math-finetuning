# CLAUDE.md

## How I expect you to write code

**No shortcuts. "Simple" never means "sloppy."** A small diff that hardcodes,
duplicates, or skips a test isn't simpler — it's deferred cost.

1. **Fix causes, not symptoms.** Find the root cause before fixing. If you're
   applying a workaround, say so explicitly and explain why. Never swallow an
   exception or silence an error to make a problem disappear.

2. **Think about consequences.** Before changing shared or widely-used code,
   trace its callers and the invariants they rely on. A fix that's locally
   correct but breaks something elsewhere — now or later — is not a fix.

3. **SOLID, sensibly.** One responsibility per class/widget/function. Separate
   pure logic from I/O so it can be tested. Inject dependencies that cross a
   boundary so they're mockable. Don't add abstractions for things that don't
   cross a boundary.

4. **DRY about knowledge, not appearance.** Don't duplicate a rule or decision.
   Code that merely looks similar but changes for different reasons stays
   separate. When unsure, prefer duplication over a premature/wrong abstraction.

5. **No hardcoded values.** No magic numbers or strings inline — give them
   names. Environment/tenant/feature-specific values go in typed config in
   application code, never scattered literals, never the database.

6. **Readable & maintainable.** Clear names, short flat functions, early
   returns over deep nesting. Comments explain *why*, not *what*. Match the
   existing style of the file you're editing.

7. **Testable, and prove it.** Ship a test for behavior you add or change. If
   something is hard to test, that's a design smell — restructure until it
   isn't. "Works but can't be tested" means it isn't done.

A change is done only when: the cause (not a symptom) is fixed, no new hardcoded
values, a test covers it, and the analyzer/formatter are clean.

## Project facts

> Keep these current as the repo evolves; only write what you've confirmed.

- **Setup command:** `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- **Analyze/lint command:** `flake8 src/ scripts/ config/ tests/ --max-line-length 120 --ignore E203,W503,E402` (pinned `flake8==7.0.0`)
- **Test command (all):** `pytest tests/ -v --tb=short`
- **Test command (single):** `pytest tests/test_data.py -v` or `pytest tests/test_data.py::test_name`
- **Format command:** `black src/ scripts/ config/ tests/` (CI checks with `black --check --diff`; pinned `black==23.12.1`)
- **Run an app:** train: `python scripts/train.py --data_path data/examples/train.json`; evaluate: `python scripts/evaluate.py --model_path outputs/models/default/final --benchmark gsm8k`; serve: FastAPI + vLLM via `inference/server.py` (configured by `MODEL_PATH`/`LORA_PATH` env vars)
- **Repo layout:** `src/` (library: `data/`, `models/`, `training/`, `evaluation/`); `config/` (typed config dicts); `scripts/` (CLI entrypoints); `train/`, `eval/`, `inference/` (per-image Dockerfiles + requirements); `tests/` (pytest); `data/`, `outputs/`, `notebooks/`, `docs/`
- **State management / data layer conventions:** configuration lives as typed Python dicts in the `config/` package (`MODEL_CONFIG`, `LORA_CONFIG`, `TRAINING_ARGS`, `PHASES`); runtime overrides come from environment variables (e.g. `MODEL_PATH`, `HF_HOME`), not scattered literals
- **Generated files NOT to hand-edit:** everything under `outputs/` (models/logs/results/visualizations), `data/raw/` and `data/processed/`, `__pycache__/`, `.pytest_cache/` — all gitignored
- **Other gotchas worth recording:** dependency versions are pinned in `requirements.txt` (torch 2.1.2, transformers 4.46.0, peft, trl, etc.) — keep pins consistent; GPU/CUDA-bound (Docker `runtime: nvidia`, RunPod deployment via `runpod_startup.sh`); CI installs CPU-only torch for tests; releases build 3 GHCR images (train / inference / eval); Python `>=3.10`, CI runs on 3.10
