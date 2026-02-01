.PHONY: install dev data-download preprocess pretrain evaluate test lint clean

# --- Setup ---
install:
	pip install -e .

dev:
	pip install -e ".[dev,notebooks,data]"
	pre-commit install

# --- Data Pipeline ---
data-download:
	python scripts/data_curation/01_download_sea_ad.py
	python scripts/data_curation/02_download_geo_datasets.py
	python scripts/data_curation/03_download_allen_brain.py

preprocess:
	python scripts/data_curation/04_preprocess_expression.py
	python scripts/data_curation/05_preprocess_chromatin.py
	python scripts/data_curation/06_harmonize_batches.py
	python scripts/data_curation/07_create_splits.py

# --- Training ---
pretrain:
	python scripts/training/01_pretrain_knight_min.py

finetune-cellstate:
	python scripts/training/02_finetune_cellstate.py

finetune-perturbation:
	python scripts/training/03_finetune_perturbation.py

# --- Evaluation ---
evaluate:
	python scripts/evaluation/01_benchmark_cellstate.py
	python scripts/evaluation/02_benchmark_perturbation.py
	python scripts/evaluation/03_benchmark_harmonization.py

# --- Dev ---
test:
	pytest tests/ -v --tb=short

lint:
	ruff check knight/ scripts/ tests/
	ruff format --check knight/ scripts/ tests/

format:
	ruff check --fix knight/ scripts/ tests/
	ruff format knight/ scripts/ tests/

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
