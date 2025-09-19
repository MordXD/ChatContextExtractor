FROM python:3.11-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1
RUN pip install --upgrade pip poetry
COPY pyproject.toml /app/
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi
COPY app /app/app
EXPOSE 8080
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
