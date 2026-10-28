.PHONY: install install-all lint format typecheck test test-ml cov serve pipeline clean

PY ?= python

install:
	uv pip install -e ".[dev]"

install-all:
	uv pip install -e ".[all]"

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

typecheck:
	mypy

test:
	pytest -m "not ml and not slow"

test-ml:
	pytest -m "not torch"
	pytest -m torch

cov:
	pytest --cov=storygraph --cov-report=term-missing --cov-report=xml

serve:
	uvicorn storygraph.serving.api:app --host 0.0.0.0 --port 8080 --reload

pipeline:
	$(PY) scripts/run_pipeline.py --config configs/pipeline.yaml --input data/sample/catalog.jsonl --output artifacts/signals.jsonl

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov coverage.xml artifacts
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
