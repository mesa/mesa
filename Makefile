# Makefile for Mesa LLM Assistant

.PHONY: help install install-dev test lint format type-check clean run docs docker-build docker-run

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  type-check   Run type checking with mypy"
	@echo "  clean        Clean up temporary files"
	@echo "  run          Run the development server"
	@echo "  docs         Build documentation"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run Docker container"

# Installation
install:
	pip install -r mesa_llm/requirements.txt

install-dev:
	pip install -r mesa_llm/requirements.txt
	pip install -e ".[dev,docs]"

# Testing
test:
	pytest mesa_llm/tests/ -v --tb=short

test-coverage:
	pytest mesa_llm/tests/ --cov=mesa_llm --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 mesa_llm/
	black --check mesa_llm/
	isort --check-only mesa_llm/

format:
	black mesa_llm/
	isort mesa_llm/

type-check:
	mypy mesa_llm/

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Development server
run:
	python -m mesa_llm.main

run-dev:
	API_RELOAD=true python -m mesa_llm.main

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8080

# Docker
docker-build:
	docker build -t mesa-llm-assistant .

docker-run:
	docker run -p 8000:8000 --env-file .env mesa-llm-assistant

# Package building
build:
	python setup.py sdist bdist_wheel

upload-test:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*

# Development workflow
dev-setup: install-dev
	cp mesa_llm/.env.example .env
	@echo "Development environment set up!"
	@echo "1. Edit .env file with your API keys"
	@echo "2. Run 'make run-dev' to start the development server"

# CI/CD helpers
ci-test: install-dev lint type-check test

# Example usage
example-basic:
	python mesa_llm/examples/basic_usage.py

example-api:
	python mesa_llm/examples/api_client.py

# Health check
health-check:
	curl -f http://localhost:8000/api/v1/health || exit 1