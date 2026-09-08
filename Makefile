.PHONY: install test lint analyse dashboard docker

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

analyse:
	aspect-sentiment data/sample_reviews.csv --output-dir outputs/sample

dashboard:
	streamlit run app.py

docker:
	docker compose up --build
