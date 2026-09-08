.PHONY: help install dev-backend dev-frontend test lint format clean docker-build docker-up docker-down docker-verify

help:
	@echo "Libra Developer Commands:"
	@echo "  make install       - Install Python and Node.js dependencies"
	@echo "  make dev-backend   - Run FastAPI backend on port 8000"
	@echo "  make dev-frontend  - Run Next.js frontend on port 3000"
	@echo "  make test          - Run all tests"
	@echo "  make lint          - Check code quality"
	@echo "  make format        - Format code"
	@echo "  make clean         - Remove temporary cache files"
	@echo "  make docker-verify - Validate container configuration & Dockerfiles"
	@echo "  make docker-build  - Build Docker container images"
	@echo "  make docker-up     - Start containerized stack in background"
	@echo "  make docker-down   - Stop and tear down containerized stack"

install:
	.\.venv\Scripts\python -m pip install -e .[dev]
	cd apps/frontend && npm install

dev-backend:
	.\.venv\Scripts\uvicorn apps.backend.main:app --reload --port 8000

dev-frontend:
	cd apps/frontend && npm run dev

test:
	.\.venv\Scripts\pytest -v tests/

lint:
	.\.venv\Scripts\ruff check .

format:
	.\.venv\Scripts\ruff format .

clean:
	powershell -Command "Get-ChildItem -Include __pycache__,.pytest_cache,.ruff_cache -Recurse -Force | Remove-Item -Recurse -Force"

docker-verify:
	.\.venv\Scripts\python scripts/verify_docker_setup.py

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down
