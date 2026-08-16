.DEFAULT_GOAL := help

.PHONY: help setup test lint format train serve docker-build docker-run

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

train: ## Treina o modelo e registra no MLflow (backend SQLite local)
	uv run python -m churn.training.train

serve: ## Sobe a API FastAPI (modelo @production do MLflow local)
	uv run uvicorn churn.serving.api:app --host 0.0.0.0 --port 8000

docker-build: ## Constrói a imagem Docker da API
	docker build -f docker/Dockerfile -t churn-api .

docker-run: ## Roda o container com o registry MLflow local montado (precisa de `make train` antes)
	docker run --rm -p 8000:8000 \
		-v "$(CURDIR)/mlflow.db:/app/mlflow.db" \
		-v "$(CURDIR)/mlruns:$(CURDIR)/mlruns" \
		churn-api
