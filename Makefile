.DEFAULT_GOAL := help
PY := python

.PHONY: help install lint format test cov run clean docker-build docker-run

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with dev dependencies
	$(PY) -m pip install -e ".[dev]"
	pre-commit install

lint: ## Check formatting and lint rules
	ruff check src tests scripts
	black --check src tests scripts

format: ## Auto-fix formatting and lint issues
	ruff check --fix src tests scripts
	black src tests scripts

test: ## Run the test suite with coverage
	pytest

run: ## Run the full pipeline (prepare -> forecast -> segment)
	coffee-intel run-all

report: ## Render docs/results.md to PDF (needs `pip install -e ".[docs]"` and Chrome/Edge)
	$(PY) scripts/build_report_pdf.py docs/results.md

clean: ## Remove caches and generated artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	rm -rf reports/figures/*.png reports/metrics/*.csv reports/metrics/*.json
	rm -rf data/interim/* data/processed/*
	find . -type d -name __pycache__ -exec rm -rf {} +

all: lint test ## Lint then test

docker-build: ## Build the Docker image
	docker build -t coffee-intel:local .

docker-run: ## Run the pipeline inside Docker (mounts ./data and ./reports)
	docker run --rm -v "$(PWD)/data:/app/data" -v "$(PWD)/reports:/app/reports" coffee-intel:local run-all
