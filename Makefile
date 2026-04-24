.PHONY: install lint format test run docker

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

docker:
	docker build -t churn-api .
	@echo "Run with: docker run -p 8000:8000 churn-api"
