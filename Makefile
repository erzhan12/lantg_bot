lint:
	ruff check .

runserver:
	uvicorn main:app --reload --port 8080

run:
	uv run main.py