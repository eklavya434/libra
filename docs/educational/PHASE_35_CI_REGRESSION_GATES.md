# Phase 35: Continuous Integration, Pre-Commit Hooks & Automated Performance Regression Gates

## 1. Overview & Pedagogical Purpose
As educational LLM architectures grow in complexity (adding RoPE scaling, PagedAttention, KV caches, multimodal projectors, and quantized layers), subtle regressions can easily accumulate:
1. **Silent Latency Degeneracy**: A seemingly harmless change to an attention loop or caching tensor can degrade generation speed from 30 tokens/sec down to 2 tokens/sec without failing any functional unit test.
2. **Broken Gradient Descent**: Architectural changes to layer normalization, residual scales, or masking can break backpropagation or freeze gradient flow.
3. **Accidental Credential & Footprint Leaks**: Developers can easily commit `.env` files or download multi-gigabyte models into git tracking, violating the strict 15 GB quota.

Phase 35 implements comprehensive, zero-cost continuous integration and automated performance gating to guarantee code quality, inference speed, and repository hygiene.

---

## 2. Core Architecture & Pipeline Components

```
                   +---------------------------------------+
                   |           Local Git Commit            |
                   +-------------------+-------------------+
                                       |
                   +-------------------v-------------------+
                   |         .githooks/pre-commit          |
                   |  - Staged files check (No .env/keys)  |
                   |  - Storage quota check (< 15 GB)      |
                   |  - Ruff lint & format check           |
                   +-------------------+-------------------+
                                       | git push
                   +-------------------v-------------------+
                   |       GitHub Actions Matrix           |
                   |       (.github/workflows/ci.yml)      |
                   +---------+---------+---------+---------+
                             |         |         |         |
                  +----------+    +----+----+    +----+    +---------+
                  |               |              |                   |
            +-----v-----+   +-----v-----+  +-----v-----+       +-----v-----+
            |   lint    |   |  backend  |  | frontend  |       |  deploy   |
            |   Ruff    |   |  pytest   |  | standalone|       |   audit   |
            |  342 files|   | 392 tests |  |   build   |       |  compose  |
            +-----------+   +-----+-----+  +-----------+       +-----------+
                                  |
                            +-----v-----+
                            | regression|
                            |   gate    |
                            | >10 t/s   |
                            | <15GB disk|
                            +-----------+
```

---

## 3. Key Technical Decisions & Components

### A. Automated CPU Performance Regression Gate
Located in [`packages/evaluation/regression_gate.py`](file:///c:/Users/eklav/Desktop/Libra/packages/evaluation/regression_gate.py), the gate evaluates:
- **Autoregressive Throughput**: Generates tokens with `generate_with_cache` and enforces a minimum tokens/sec speed threshold.
- **Forward Pass Latency**: Profiles multi-layer attention forward evaluations.
- **Single-Step Optimization Sanity**: Evaluates loss before and after an AdamW optimizer step, ensuring $\mathcal{L}_{\text{after}} < \mathcal{L}_{\text{before}}$.
- **Storage Footprint**: Sums all tracked directories (`models`, `data`, `checkpoints`, `.venv`) to ensure the 15 GB storage quota is never breached.

### B. Pre-Commit Secret and Weight Guard
Located in [`scripts/pre_commit_check.py`](file:///c:/Users/eklav/Desktop/Libra/scripts/pre_commit_check.py) and registered via [`scripts/install_hooks.py`](file:///c:/Users/eklav/Desktop/Libra/scripts/install_hooks.py):
- Scans `git diff --cached --name-only` for sensitive files (`.env`, `credentials.json`, `id_rsa`).
- Rejects files larger than 25 MB from regular Git commits (requiring volume mounts or Git LFS).
- Enforces zero lint errors and correct formatting.

### C. Zero-Cost ($0 / ₹0) GitHub Actions CI
The CI matrix in [`.github/workflows/ci.yml`](file:///c:/Users/eklav/Desktop/Libra/.github/workflows/ci.yml) runs entirely on standard GitHub-hosted `ubuntu-latest` free runners:
- Uses `--extra-index-url https://download.pytorch.org/whl/cpu` to avoid multi-gigabyte CUDA downloads.
- Employs GitHub Actions dependency caching for `pip` and `npm`.

---

## 4. Educational Summary (LIBRA Protocol)

- **WHAT**:
  - `packages/evaluation/regression_gate.py`: `PerformanceRegressionGate` and `RegressionReport` measuring CPU tokens/sec, latency, step loss reduction, and disk quota.
  - `scripts/pre_commit_check.py`: Cross-platform pre-commit validator.
  - `.githooks/pre-commit` & `scripts/install_hooks.py`: Git hook configuration.
  - `.github/workflows/ci.yml`: 5-job GitHub Actions CI matrix (`lint`, `backend-test`, `frontend-build`, `deployment-audit`, `regression-gate`).
  - `pyproject.toml`: Refined Ruff linting rules with 100% clean formatting.
  - `scripts/run_phase35_regression_gate.py`: CLI benchmark script.

- **WHY**:
  - Catches performance degradation before code is merged.
  - Protects private credentials and enforces the 15 GB quota locally before commits reach Git.
  - Guarantees $0 operation by leveraging free CI runners and CPU PyTorch wheels.

- **HOW**:
  - Git hooks intercept commits via `core.hooksPath = .githooks`.
  - The regression gate initializes a standard `ModernTransformerLM` and measures timing using high-resolution monotonic clocks (`time.perf_counter()`).
  - GitHub Actions runs parallel jobs with dependency caching.

- **TEST**:
  - 6 new automated unit tests in `tests/evaluation/test_regression_gate.py` and `tests/unit/test_ci_config.py`.
  - Full test suite: **392 passed tests** in ~48s (100% pass rate).
  - All 342 Python files pass Ruff checks with zero errors.
  - Pre-commit check and regression gate CLI execute cleanly.

- **NEXT**:
  - **Phase 36: Final Capstone Integration, System Polish & Comprehensive Release Verification**.
