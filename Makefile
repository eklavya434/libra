.PHONY: help install dev-backend dev-frontend test lint format clean

help:
	@echo "Libra Developer Commands:"
	@echo "  make install       - Install Python and Node.js dependencies"
	@echo "  make dev-backend   - Run FastAPI backend on port 8000"
	@echo "  make dev-frontend  - Run Next.js frontend on port 3000"
	@echo "  make test          - Run all tests"
	@echo "  make lint          - Check code quality"
	@echo "  make format        - Format code"
	@echo "  make clean         - Remove temporary cache files"

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
