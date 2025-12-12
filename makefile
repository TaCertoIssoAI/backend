run:
	uvicorn app.main:app --reload --port 8000

run-prod:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 7

run-load-test:
	PROD_SERVICE_URL="http://localhost:8000" python integration_tests/prod_load_test.py