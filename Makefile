# =========================================================================
# DeepCrawl-Chat - Makefile
#
# This Makefile provides utility commands for development, testing, and
# deployment of the DeepCrawl-Chat system.
# =========================================================================

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

# Python settings
PYTHON := python3
PYTHON_VERSION := 3.12
PIP := $(PYTHON) -m pip

# Package management tool (uv is faster than pip)
UV := uv

# Project directories
SRC_DIR := src
TESTS_DIR := tests
CONF_DIR := conf
DOCS_DIR := docs

# Testing settings
PYTEST := $(PYTHON) -m pytest
PYTEST_ARGS := -v
COVERAGE := $(PYTHON) -m pytest --cov=$(SRC_DIR)

# Linting/formatting settings
BLACK := $(PYTHON) -m black
ISORT := $(PYTHON) -m isort
FLAKE8 := $(PYTHON) -m flake8
MYPY := $(PYTHON) -m mypy



# Clean build artifacts and temporary files
.PHONY: clean
clean:
	@echo "==> Cleaning build artifacts..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.pyd" -delete
	@find . -type f -name ".coverage" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@find . -type d -name "*.egg" -exec rm -rf {} +
	@echo "==> Cleaning complete"



# Help message showing available targets
.PHONY: help
help:
	@echo "DeepCrawl-Chat Makefile"
	@echo ""
	@echo "Development Environment:"
	@echo "  setup             - Set up development environment with virtual environment"
	@echo "  install           - Install dependencies from requirements.txt"
	@echo "  install-dev       - Install development dependencies"
	@echo "  update-deps       - Update dependencies in requirements.txt"
	@echo "  clean             - Clean build artifacts and temporary files"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint              - Run linting checks"
	@echo "  format            - Format code using Black and isort"
	@echo ""
	@echo "Docker:"
	@echo "  build-docker      - Build Docker image"
	@echo "  docker-up         - Start all Docker containers in detached mode"
	@echo "  docker-up-logs    - Start Docker containers with logs"
	@echo "  docker-down       - Stop Docker containers"
	@echo "  docker-logs       - View logs for a container (use: make docker-logs service=image-generation)"
	@echo ""
	@echo "For more information, see README.md"