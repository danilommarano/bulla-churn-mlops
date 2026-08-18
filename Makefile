.DEFAULT_GOAL := help

.PHONY: help setup test lint format format-check ci train pipeline serve feast-materialize monitor monitor-drift presentation-metrics docker-build docker-run

help: ## Lista os targets disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Prepara o ambiente do zero (deps + .env)
	uv sync
	@test -f .env || cp .env.example .env
	@echo "Ambiente pronto. Edite .env se precisar."

test: ## Roda a suíte de testes
	uv run pytest -v

lint: ## Checa estilo com ruff
	uv run ruff check src tests

format: ## Formata com ruff
	uv run ruff format src tests

format-check: ## Checa formatação sem alterar (usado no CI)
	uv run ruff format --check src tests

ci: ## Roda a sequência de CI localmente (espelha o GitHub Actions)
	$(MAKE) lint
	$(MAKE) format-check
	$(MAKE) test
	$(MAKE) docker-build
	$(MAKE) pipeline

train: ## Treina o modelo e registra no MLflow (backend SQLite local)
	uv run python -m churn.training.train

pipeline: ## Roda o DAG de treino KFP local ponta a ponta (prepare -> split -> train -> evaluate -> register)
	uv run python -m churn.orchestration.dag

feast-materialize: ## Popula e materializa a feature store Feast (offline -> online)
	uv run python -m churn.feature_store.materialize

serve: ## Sobe a API FastAPI (modelo @production do MLflow local)
	uv run uvicorn churn.serving.api:app --host 0.0.0.0 --port 8000

monitor: ## Gera o relatório de drift/qualidade (holdout saudável; gate deve passar)
	uv run python -m churn.monitoring

monitor-drift: ## Idem com drift simulado (demonstra detecção; gate falha de propósito)
	uv run python -m churn.monitoring --simulate-drift

presentation-metrics: ## Exporta métricas do @production para o site (presentation/src/data/metrics.json)
	uv run python -m churn.reporting

docker-build: ## Constrói a imagem Docker da API
	docker build -f docker/Dockerfile -t churn-api .

docker-run: ## Roda o container com o registry MLflow local montado (precisa de `make train` antes)
	docker run --rm -p 8000:8000 \
		-v "$(CURDIR)/mlflow.db:/app/mlflow.db" \
		-v "$(CURDIR)/mlruns:$(CURDIR)/mlruns" \
		-v "$(CURDIR)/feature_repo:$(CURDIR)/feature_repo" \
		churn-api
