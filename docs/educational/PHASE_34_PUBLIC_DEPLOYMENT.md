# Phase 34: Public Deployment & Containerized Lab Infrastructure (Docker & Cloud Readiness)

## 1. Overview & Pedagogical Purpose
Until Phase 34, Project Libra was developed exclusively within a bare-metal local virtual environment (`.venv`) and local Node.js runtime, in accordance with **Prime Directive 5** (*"No Docker Before Phase 34"*). This saved disk space and memory during the iterative architecture phases (Phases 0 through 33).

Phase 34 containerizes the entire stack, providing:
1. **Reproducibility**: An identical Linux execution environment regardless of host operating system (Windows, macOS, Linux).
2. **Security**: Non-root system users (`libra:1000`, `nextjs:1001`) preventing container escape vulnerabilities.
3. **Layer Caching & Minimal Footprint**: Multi-stage builds that isolate compilation tools from the final runtime images, reducing container size by over 80%.
4. **Zero-Cost ($0 / ₹0) Cloud Readiness**: Structured for deployment on free-tier hosting platforms (Hugging Face Spaces, Fly.io, Render, Oracle Cloud Free Tier).

---

## 2. Core Architecture & Container Topology

```
                   +---------------------------------------+
                   |           User Web Browser            |
                   +-------------------+-------------------+
                                       |
                   +-------------------+-------------------+
                   |         Docker Bridge Network         |
                   |            (libra-network)            |
                   +---------+-------------------+---------+
                             |                   |
            Port 3000        |                   |  Port 8000
    +------------------------v---+           +---v-----------------------+
    |       libra-frontend       |           |       libra-backend       |
    |  Node 20 Alpine Standalone |           |   Python 3.11 Slim (CPU)  |
    |  User: nextjs (UID 1001)   |           |   User: libra (UID 1000)  |
    |  Healthcheck: wget /       |           |   Healthcheck: urllib /   |
    +----------------------------+           +-------------+-------------+
                                                           |
                                             +-------------+-------------+
                                             |  Host Volume Mounts       |
                                             |  - ./models -> /app/models|
                                             |  - ./data   -> /app/data  |
                                             |  - ./checkpoints          |
                                             +---------------------------+
```

---

## 3. Key Technical Decisions & Innovations

### A. CPU-First PyTorch Wheel Isolation
Standard PyTorch wheels downloaded via `pip install torch` bundle multi-gigabyte NVIDIA CUDA/cuDNN binaries, inflating container images to 8GB–12GB.
In `infra/docker/Dockerfile.backend`, we explicitly configure:
```dockerfile
RUN pip install --no-cache-dir --prefix=/install -r requirements-docker.txt
```
where `requirements-docker.txt` specifies `--extra-index-url https://download.pytorch.org/whl/cpu`. This keeps the backend image at ~1.2GB total.

### B. Next.js Standalone Build
In `apps/frontend/next.config.mjs`, we configure `output: "standalone"`.
During `next build`, Next.js analyzes module imports and creates a minimal `.next/standalone/server.js` with only the exact `node_modules` required for execution. The frontend runner stage needs no build tools or dev dependencies, reducing the frontend container to ~250MB.

### C. Volume Isolation for Model Quota Compliance
Baking model weights or checkpoint files into a Docker image violates the strict 15 GB storage quota and slows down container builds. All models, checkpoints, and sqlite databases are mounted as Docker volumes:
```yaml
volumes:
  - ./models:/app/models
  - ./data:/app/data
  - ./checkpoints:/app/checkpoints
```

### D. Multi-Stage Dockerfile Security
Both containers employ multi-stage patterns:
- **Build stage**: Contains compilers, build-essential, npm devDependencies.
- **Runtime stage**: Discards compilers, creates an unprivileged non-root user (`libra` or `nextjs`), and runs the service.

---

## 4. Educational Summary (LIBRA Protocol)

- **WHAT**:
  - `infra/docker/Dockerfile.backend`: Multi-stage CPU-optimized Python 3.11 image with non-root security.
  - `infra/docker/Dockerfile.frontend`: Multi-stage Next.js 14 standalone image.
  - `infra/docker/requirements-docker.txt`: Pinned CPU dependencies.
  - `docker-compose.yml`: Multi-service orchestration with health checks, network isolation, and volume mounts.
  - `.dockerignore`: Exclusion list preventing build context leaks.
  - `apps/frontend/next.config.mjs`: Standalone production build configuration.
  - `packages/core/deployment_validator.py`: Programmatic configuration audit tool.
  - `scripts/verify_docker_setup.py`: CLI container readiness verification script.

- **WHY**:
  - Delivers complete cloud readiness and reproducible zero-cost deployment while honoring Directive 5.
  - Enforces container security best practices (non-root, healthchecks).
  - Eliminates image bloat via standalone compilation and CPU PyTorch wheels.

- **HOW**:
  - Multi-stage Docker builds separate build environments from slim execution containers.
  - Next.js standalone mode bundles runtime dependencies into `server.js`.
  - Docker Compose coordinates healthcheck-dependent service startup.

- **TEST**:
  - 6 new automated unit tests in `tests/unit/test_deployment_config.py`.
  - Next.js production build succeeded with standalone output (`apps/frontend/.next/standalone/server.js`).
  - Full test suite: **386 passed tests** in ~46s.
  - CLI verification script confirms all configurations valid.

- **NEXT**:
  - **Phase 35: Continuous Integration, Pre-Commit Hooks & Automated Performance Regression Gates**.
