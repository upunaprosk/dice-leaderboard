.PHONY: install test test-integration build check

install:
	uv sync --locked --extra dev

test:
	uv run --extra dev python -m pytest -m "not integration"

test-integration:
	uv run --extra dev python -m pytest -m integration -v

build:
	uv build

check:
	uv lock --check
	uv run --extra dev python -m pytest -m "not integration"
	uv build