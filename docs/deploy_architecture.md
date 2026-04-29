# Arquitetura de Deploy — Churn Prediction API

## 1. Decisão: Real-Time via API REST

### Arquitetura escolhida
Inferência em **tempo real** através de uma API REST (FastAPI), servindo predições individuais sob demanda.

### Alternativa considerada
**Batch processing:** executar predições periodicamente (ex: diariamente) para toda a base de clientes e armazenar os resultados em um banco de dados.

### Justificativa da escolha

| Critério | Real-Time (escolhido) | Batch |
|----------|----------------------|-------|
| **Latência** | ~30ms por request | Minutos/horas (processamento em lote) |
| **Atualização** | Instantânea (dados atuais do cliente) | Defasada (último batch) |
| **Caso de uso** | Atendimento ao vivo, decisão imediata | Relatórios periódicos, campanhas |
| **Complexidade** | API + modelo em memória | Scheduler + storage + pipeline |
| **SLO de latência** | p95 < 200ms (atendido) | N/A |

A escolha por real-time é motivada pelo requisito de negócio: a equipe de retenção e o atendimento ao cliente precisam de predições **durante a interação com o cliente**, não no dia seguinte. O SLO de latência p95 < 200ms exige resposta imediata.

## 2. Diagrama de Arquitetura

```
┌──────────┐     HTTP POST      ┌─────────────────────────────────────┐
│  Cliente  │ ─────────────────→ │           FastAPI (Uvicorn)         │
│ (curl/UI) │                    │                                     │
└──────────┘     JSON Response   │  ┌─────────┐   ┌──────────────┐    │
       ↑      ←───────────────── │  │ Pydantic │──→│ Feature Eng. │    │
       │                         │  │ Validação│   └──────┬───────┘    │
       │                         │  └─────────┘          │            │
       │                         │                ┌──────▼───────┐    │
       │                         │                │ Preprocessor │    │
       │                         │                │  (.joblib)   │    │
       │                         │                └──────┬───────┘    │
       │                         │                ┌──────▼───────┐    │
       │                         │                │  ChurnMLP    │    │
       │                         │                │  (.pt model) │    │
       │                         │                └──────┬───────┘    │
       │                         │                ┌──────▼───────┐    │
       │                         │                │  Threshold   │    │
       │                         │                │  (0.24)      │    │
       └─────────────────────────│                └──────────────┘    │
                                 └─────────────────────────────────────┘
```

### Fluxo de uma request

1. **Cliente** envia POST `/predict` com JSON dos dados do cliente (18 campos)
2. **Pydantic** valida os campos (tipos, ranges, valores obrigatórios)
3. **Feature Engineering** calcula 4 features derivadas (total_services_count, tenure_to_charges_ratio, has_no_support, is_new_customer)
4. **Preprocessor** (ColumnTransformer) aplica StandardScaler + OneHotEncoder → 44 features
5. **ChurnMLP** recebe tensor float32, retorna logit → sigmoid → probabilidade
6. **Threshold** (0.24, calibrado por custo) decide se `churn_prediction = True/False`
7. **Response** retorna `churn_probability` e `churn_prediction`

## 3. Containerização

### Docker Multi-Stage Build

```dockerfile
# Stage 1: Builder — exporta dependências
FROM python:3.11-slim AS builder
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt --without dev --output requirements.txt
    && pip install --prefix=/install -r requirements.txt

# Stage 2: Runner — imagem mínima
FROM python:3.11-slim AS runner
COPY --from=builder /install /usr/local
RUN useradd --create-home appuser
USER appuser
COPY src/ ./src/
COPY models/ ./models/
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Decisões de segurança
- **Usuário não-root** (`appuser`): evita execução com privilégios elevados
- **Imagem slim**: superfície de ataque reduzida
- **Sem dependências de dev**: pytest, ruff, jupyter não entram na imagem de produção
- **Multi-stage build**: artefatos de build não ficam na imagem final

## 4. Endpoints

| Endpoint | Método | Descrição | Código de sucesso |
|----------|--------|-----------|-------------------|
| `/health` | GET | Health check — verifica se modelo está carregado | 200 |
| `/predict` | POST | Predição de churn para um cliente | 200 |
| `/docs` | GET | Documentação Swagger (automática) | 200 |

### Códigos de erro

| Código | Situação |
|--------|----------|
| 422 | Dados de entrada inválidos (validação Pydantic ou preprocessamento) |
| 503 | Modelo não carregado (arquivo .pt não encontrado) |
| 500 | Erro interno durante inferência |

## 5. Observabilidade

- **Logging estruturado:** JSON via `python-json-logger` (timestamp, level, message)
- **Middleware de latência:** loga método HTTP, path e tempo em ms para cada request
- **Health check:** endpoint `/health` retorna status do modelo para probes de liveness/readiness

## 6. Considerações para Produção

### Escalabilidade
- Uvicorn suporta múltiplos workers (`--workers N`) para paralelismo
- Modelo carregado em memória no startup (cold start ~1s)
- Stateless: cada request é independente, permite horizontal scaling

### Alta disponibilidade
- Container pode ser orquestrado via Kubernetes ou ECS
- Health check em `/health` para readiness/liveness probes
- Fallback: se API estiver fora do ar, usar último resultado de batch como contingência
