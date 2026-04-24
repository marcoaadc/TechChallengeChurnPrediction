# Stage 1: Builder — exporta dependências e instala
FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir poetry==1.8.5

WORKDIR /build

COPY pyproject.toml poetry.lock ./

RUN poetry export -f requirements.txt --without dev --output requirements.txt \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runner — imagem mínima de produção
FROM python:3.11-slim AS runner

COPY --from=builder /install /usr/local

RUN useradd --create-home appuser
USER appuser

WORKDIR /app

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
