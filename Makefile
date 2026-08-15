.DEFAULT_GOAL := help

.PHONY: help setup test lint format

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
