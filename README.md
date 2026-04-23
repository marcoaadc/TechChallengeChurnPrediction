# Tech Challenge - Churn Prediction

Projeto de previsão de Churn (cancelamento de clientes) utilizando uma Rede Neural do tipo MLP (Multi-Layer Perceptron) desenvolvida em PyTorch.

## Problema

Churn é a taxa de cancelamento de clientes em um determinado período. Prever quais clientes têm maior probabilidade de cancelar permite que a empresa tome ações preventivas de retenção, reduzindo custos e aumentando a receita.

Este projeto implementa uma API de classificação binária que recebe os dados de um cliente e retorna a probabilidade de churn.

## Stack

- **Modelo:** PyTorch (MLP)
- **Rastreamento de experimentos:** MLflow
- **API:** FastAPI + Uvicorn
- **Gerenciamento de dependências:** Poetry

## Instalação

```bash
# Instalar o Poetry (caso ainda não tenha)
pip install poetry

# Instalar as dependências do projeto
poetry install

# Ativar o ambiente virtual
poetry shell
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
├── docs/               # Documentação
├── models/             # Modelos treinados
├── notebooks/          # Notebooks de análise e experimentação
├── src/                # Código-fonte
├── tests/              # Testes automatizados
├── pyproject.toml      # Dependências (Poetry)
└── README.md
```
