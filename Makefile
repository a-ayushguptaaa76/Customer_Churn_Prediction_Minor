install:
	python -m pip install -r requirements.txt

train:
	python -m src.train --data data/customer_data.csv

test:
	pytest -q

api:
	uvicorn app.api:app --reload

docker:
	docker compose up --build
