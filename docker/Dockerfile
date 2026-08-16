FROM python:3.12-slim

# uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Dependency layer (cached unless pyproject/lock change)
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# The MLflow registry (mlflow.db + mlruns/) is mounted at runtime, not baked in.
ENV CHURN_MLFLOW_TRACKING_URI=sqlite:///mlflow.db
EXPOSE 8000

CMD ["uv", "run", "--no-dev", "uvicorn", "churn.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
