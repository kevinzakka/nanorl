SHELL := /bin/bash

.PHONY: help check format test
.DEFAULT: help

help:
	@echo "Usage: make <target>"
	@echo
	@echo "Available targets:"
	@echo "  help: Show this help"
	@echo "  format: Run type checking and code styling inplace"
	@echo "  test: Run all tests"

format:
	black .
	ruff . --fix
	mypy --ignore-missing-imports .

test:
	pytest -n auto
