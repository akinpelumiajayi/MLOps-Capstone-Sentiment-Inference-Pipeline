# Developer entry points. Run `make help` for the list.
# On Windows use Git Bash or WSL (make is not available in plain PowerShell).

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

PYTHON  ?= python
VENV    ?= .venv
BIN     := $(if $(filter Windows_NT,$(OS)),$(VENV)/Scripts,$(VENV)/bin)
API_URL ?= http://localhost:8000
CLUSTER ?= mlops
NS      ?= sentiment

export PYTHONPATH := src
export MLFLOW_DISABLE_AGENT_HINT := 1

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---- setup ------------------------------------------------------------------------
.PHONY: install
install: ## Create .venv with all dev dependencies and git hooks
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt -e .
	$(BIN)/pre-commit install

# ---- quality ----------------------------------------------------------------------
.PHONY: lint format typecheck test check
lint: ## Ruff lint + format check
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format: ## Auto-fix lint issues and format
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

typecheck: ## Mypy (strict)
	$(BIN)/mypy

test: ## Tests with coverage
	$(BIN)/pytest --cov --cov-report=term-missing

check: lint typecheck test ## Everything CI's quality job runs

# ---- ML pipeline ------------------------------------------------------------------
.PHONY: train gate metrics compare mlflow-ui drift
train: ## Run the DVC pipeline (download -> prepare -> train -> evaluate -> compare)
	PATH="$(BIN):$$PATH" $(BIN)/dvc repro

gate: ## Model quality gate against the committed metrics
	git show HEAD:reports/metrics.json > /tmp/baseline_metrics.json 2>/dev/null || true
	$(BIN)/python -m sentiment.gate --baseline /tmp/baseline_metrics.json

metrics: ## Show metrics and diff vs. last commit
	$(BIN)/dvc metrics show
	$(BIN)/dvc metrics diff

mlflow-ui: ## Browse local experiments at http://localhost:5000
	$(BIN)/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

drift: ## Offline drift check: make drift INPUT=path/to/batch.csv
	$(BIN)/python -m sentiment.drift --input $(INPUT)

# ---- local services ---------------------------------------------------------------
.PHONY: serve up down logs smoke loadtest
serve: ## Run the API with auto-reload
	$(BIN)/uvicorn sentiment.api.main:app --reload --port 8000

.env:
	cp .env.example .env
	@echo "Created .env from .env.example; change GRAFANA_ADMIN_PASSWORD."

up: .env ## Full stack with docker compose (trains first if no model)
	docker compose up --build -d
	@echo "API http://localhost:8000/docs | UI :8501 | MLflow :5000 | Prometheus :9090 | Grafana :3000"

down: ## Stop the stack
	docker compose down

logs: ## Tail stack logs
	docker compose logs -f api trainer

smoke: ## Smoke-test a running API (API_URL=...)
	$(BIN)/python scripts/smoke_test.py $(API_URL)

loadtest: ## 2-minute headless load test against API_URL, CSVs in reports/
	$(BIN)/locust -f loadtest/locustfile.py --host $(API_URL) --headless -u 50 -r 10 -t 2m \
		--csv reports/loadtest --only-summary

# ---- Kubernetes (kind) ------------------------------------------------------------
.PHONY: helm-lint kind-up kind-deploy hpa-demo kind-down
helm-lint: ## Lint and render the chart
	helm lint helm/sentiment-api --set image.repository=x --set image.tag=y --strict
	helm template sentiment-api helm/sentiment-api -f helm/sentiment-api/values-kind.yaml > /dev/null

kind-up: ## Create a kind cluster with metrics-server (needed for HPA)
	./scripts/kind-up.sh $(CLUSTER)

kind-deploy: ## Build the image, load it into kind, helm install, smoke test
	./scripts/kind-deploy.sh $(CLUSTER) $(NS)

hpa-demo: ## Generate load and watch the HPA scale out
	./scripts/hpa-demo.sh $(NS)

kind-down: ## Delete the kind cluster
	kind delete cluster --name $(CLUSTER)

# ---- housekeeping -----------------------------------------------------------------
.PHONY: clean
clean: ## Remove caches (keeps data, models and MLflow runs)
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml junit.xml
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
