PY=python

.PHONY: setup data test app mlflow fmt lint

setup:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt
	pre-commit install

data:
	$(PY) -m src.simulate --outdir data --users 100000

test:
	pytest

app:
	streamlit run app/streamlit_app.py

mlflow:
	mlflow ui --host 127.0.0.1 --port 5000

fmt:
	isort .
	black .

lint:
	flake8 .
