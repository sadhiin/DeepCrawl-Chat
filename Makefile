# Makefile for DeepCrawl-Chat

# ------------------------------------------------------------
# Project configuration
# ------------------------------------------------------------
PYTHON ?= python3
UV ?= uv

# ------------------------------------------------------------
# Development environment
# ------------------------------------------------------------
.PHONY: install sync dev
install: sync

# Sync the environment and install all dependencies (including dev deps)
sync:
	$(UV) sync --dev

# Run an interactive development shell
 dev:
	$(UV) run $(PYTHON) -m ipython

# ------------------------------------------------------------
# Linting & formatting
# ------------------------------------------------------------
.PHONY: lint format
lint:
	$(UV) run ruff check src tests

format:
	$(UV) run ruff format src tests

# ------------------------------------------------------------
# Testing
# ------------------------------------------------------------
.PHONY: test
test:
	$(UV) run pytest

# ------------------------------------------------------------
# Application entry points
# ------------------------------------------------------------
.PHONY: api run-api
api:
	$(UV) run uvicorn src.deepcrawl_chat.api.main:app --reload

run-api: api

# ------------------------------------------------------------
# Crawler CLI
# ------------------------------------------------------------
.PHONY: crawl
crawl:
	$(UV) run python deep_crawler.py https://python.langchain.com --depth 1

# ------------------------------------------------------------
# Indexing CLI (example placeholder)
# ------------------------------------------------------------
.PHONY: index
index:
	$(UV) run python -m src.deepcrawl_chat.cli.index_crawl data/crawls/example_crawl.csv

# ------------------------------------------------------------
# Docker
# ------------------------------------------------------------
DOCKER_IMAGE ?= deepcrawl-chat
.PHONY: docker-build docker-run
docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-run:
	docker run -p 8000:8000 --env-file .env $(DOCKER_IMAGE)

# ------------------------------------------------------------
# Clean up
# ------------------------------------------------------------
.PHONY: clean
clean:
	rm -rf .venv __pycache__ *.egg-info build dist
