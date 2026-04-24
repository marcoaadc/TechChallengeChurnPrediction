.PHONY: install lint format test run

install:
	poetry install

lint:
	poetry run ruff check src/ tests/
	poetry run ruff format --check src/ tests/

format:
	poetry run ruff check --fix src/ tests/
	poetry run ruff format src/ tests/

test:
	poetry run pytest tests/ -v

run:
	poetry run uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
