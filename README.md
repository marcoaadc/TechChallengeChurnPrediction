# Tech Challenge - Churn Prediction

Projeto de previsão de Churn (cancelamento de clientes) utilizando uma Rede Neural do tipo MLP (Multi-Layer Perceptron) desenvolvida em PyTorch.

## Problema

Churn é a taxa de cancelamento de clientes em um determinado período. Prever quais clientes têm maior probabilidade de cancelar permite que a empresa tome ações preventivas de retenção, reduzindo custos e aumentando a receita.

Este projeto implementa uma API de classificação binária que recebe os dados de um cliente e retorna a probabilidade de churn.

## Stack

- **Modelo:** PyTorch (MLP)
- **Rastreamento de experimentos:** MLflow
- **API:** FastAPI + Uvicorn
- **Testes:** pytest + pandera
- **Linting:** ruff
- **Gerenciamento de dependências:** Poetry

## Instalação

```bash
# Instalar o Poetry (caso ainda não tenha)
pip install poetry

# Instalar as dependências do projeto
make install
# ou: poetry install

# Ativar o ambiente virtual
poetry shell
```

## Comandos de Desenvolvimento

```bash
make install   # Instala dependências
make lint      # Verifica linting (ruff)
make format    # Formata código automaticamente
make test      # Roda testes (pytest)
make run       # Inicia a API (uvicorn)
```

## API

### Iniciar o servidor

```bash
make run
# A API estará disponível em http://localhost:8000
# Documentação interativa (Swagger): http://localhost:8000/docs
```

### Endpoints

#### `GET /health`

Verifica se a API e o modelo estão operacionais.

```bash
curl http://localhost:8000/health
```

#### `POST /predict`

Recebe dados de um cliente e retorna a probabilidade de churn.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 840.50
  }'
```

## Arquitetura

```
Cliente (curl/UI)
    │
    ▼  POST /predict (JSON)
┌─────────────────────────────┐
│        FastAPI (Uvicorn)     │
│  ┌────────────────────────┐ │
│  │ Pydantic (validação)   │ │
│  └──────────┬─────────────┘ │
│  ┌──────────▼─────────────┐ │
│  │ Feature Engineering    │ │
│  │ (4 features derivadas) │ │
│  └──────────┬─────────────┘ │
│  ┌──────────▼─────────────┐ │
│  │ Preprocessor (.joblib) │ │
│  │ StandardScaler + OHE   │ │
│  └──────────┬─────────────┘ │
│  ┌──────────▼─────────────┐ │
│  │ ChurnMLP (.pt)         │ │
│  │ [64,32] + threshold    │ │
│  └──────────┬─────────────┘ │
│             ▼               │
│   {probability, prediction} │
└─────────────────────────────┘
```

- **Inferência real-time** via API REST (latência p95 < 200ms)
- **Containerização** com Docker multi-stage build (python:3.11-slim)
- **Threshold calibrado** por custo de negócio (FN=R$500, FP=R$50)

## Documentação

| Documento | Descrição |
|-----------|-----------|
| [ML Canvas](docs/ml_canvas.md) | Problema de negócio, stakeholders, métricas, SLOs |
| [Model Card](docs/model_card.md) | Performance, limitações, vieses, cenários de falha |
| [Arquitetura de Deploy](docs/deploy_architecture.md) | Real-time vs batch, diagrama, containerização |
| [Plano de Monitoramento](docs/monitoring_plan.md) | Métricas, alertas, drift detection, playbook |

## Docker

```bash
# Build da imagem
make docker
# ou: docker build -t churn-api .

# Executar o container
docker run -p 8000:8000 churn-api

# A API estará disponível em http://localhost:8000
```

## Dados

O dataset utilizado é o [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), que contém informações de clientes de uma empresa de telecomunicações.

### Download automático

O download é feito automaticamente ao executar o notebook ou usar a função `load_telco_churn()`. Para baixar manualmente:

```bash
python -c "from src.data_acquisition import download_telco_churn; download_telco_churn()"
```

### Credenciais do Kaggle

Para baixar o dataset, é necessário configurar as credenciais da API do Kaggle:

1. Crie uma conta em [kaggle.com](https://www.kaggle.com)
2. Vá em **Account** > **API** > **Create New Token**
3. Salve o arquivo `kaggle.json` em:
   - **Windows:** `C:\Users\<usuario>\.kaggle\kaggle.json`
   - **Linux/Mac:** `~/.kaggle/kaggle.json`

Alternativamente, defina as variáveis de ambiente `KAGGLE_USERNAME` e `KAGGLE_KEY`.

## Estrutura do Projeto

```
├── data/
│   ├── raw/            # Dados brutos
│   └── processed/      # Dados processados
├── docs/
│   ├── ml_canvas.md          # ML Canvas (stakeholders, métricas, SLOs)
│   ├── model_card.md         # Model Card (performance, limitações, vieses)
│   ├── deploy_architecture.md # Arquitetura de deploy (real-time vs batch)
│   └── monitoring_plan.md    # Plano de monitoramento (alertas, drift, playbook)
├── models/             # Modelos treinados (.pt, .joblib)
├── notebooks/
│   ├── 01_eda_baseline.ipynb  # EDA + baselines
│   └── 02_modeling.ipynb      # MLP PyTorch + comparação
├── src/
│   ├── api.py          # API FastAPI
│   ├── data_acquisition.py  # Download do dataset
│   ├── data_loader.py  # Carregamento de dados
│   ├── model.py        # Arquitetura MLP
│   ├── schemas.py      # Schemas Pydantic
│   └── training.py     # Loop de treinamento
├── tests/              # Testes automatizados
├── Makefile            # Comandos de desenvolvimento
├── pyproject.toml      # Dependências e configuração
└── README.md
```
