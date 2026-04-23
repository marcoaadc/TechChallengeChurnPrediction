# ML Canvas — Previsão de Churn em Telecomunicações

## 1. Problema de Negócio

Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado. A diretoria precisa de um modelo preditivo que identifique clientes com alto risco de cancelamento (churn), permitindo ações proativas de retenção.

## 2. Stakeholders

| Stakeholder | Papel | Interesse |
|-------------|-------|-----------|
| Diretoria Comercial | Patrocinador | Reduzir taxa de churn e aumentar receita recorrente |
| Equipe de Retenção | Usuário final | Receber lista priorizada de clientes em risco para ação |
| Equipe de Dados | Desenvolvedor | Construir e manter o modelo em produção |
| Atendimento ao Cliente | Consumidor indireto | Abordagem personalizada para clientes em risco |

## 3. Dados

- **Fonte:** Dataset Telco Customer Churn (IBM/Kaggle)
- **Volume:** ~7.000 registros, 20 features
- **Granularidade:** 1 registro por cliente
- **Features:** dados demográficos (gender, SeniorCitizen, Partner, Dependents), serviços contratados (PhoneService, InternetService, etc.), dados financeiros (tenure, MonthlyCharges, TotalCharges)
- **Variável alvo:** Churn (Yes/No) — classificação binária
- **Desbalanceamento:** ~26% positivo (churn), ~74% negativo

## 4. Métricas

### Métricas Técnicas
| Métrica | Justificativa |
|---------|---------------|
| AUC-ROC | Avaliação geral da capacidade de ranking do modelo |
| PR-AUC | Foco na classe minoritária (churn) — mais informativo que AUC-ROC em datasets desbalanceados |
| F1-Score | Equilíbrio entre precision e recall |
| Recall | Maximizar detecção de churners — FN é mais custoso que FP |

### Métrica de Negócio
- **Custo de churn evitado:** Economia = (churners detectados × receita média retida) − (falsos positivos × custo da ação de retenção)
- Premissas: FN custa ~R$ 500 (receita perdida), FP custa ~R$ 50 (oferta de retenção)

## 5. Abordagem

- **Modelo central:** MLP (Multi-Layer Perceptron) com PyTorch
- **Baselines:** DummyClassifier (most_frequent), Regressão Logística
- **Pré-processamento:** StandardScaler (numéricas) + OneHotEncoder (categóricas) via sklearn Pipeline
- **Validação:** Stratified K-Fold (5 folds), seed=42
- **Tracking:** MLflow para parâmetros, métricas e artefatos

## 6. SLOs (Service Level Objectives)

| SLO | Target | Justificativa |
|-----|--------|---------------|
| Latência da API (p95) | < 200ms | Consulta em tempo real para atendimento |
| Disponibilidade | 99.5% | Operação contínua para equipe de retenção |
| AUC-ROC mínimo | > 0.80 | Baseline da Regressão Logística como piso |
| Recall mínimo | > 0.70 | Capturar pelo menos 70% dos churners |
| Tempo de re-treinamento | Mensal | Adaptar a mudanças no comportamento dos clientes |

## 7. Riscos e Limitações

- **Data drift:** Comportamento de churn pode mudar com novas ofertas ou concorrência
- **Viés demográfico:** Modelo pode ter performance diferente para subgrupos (seniors, gênero)
- **Tamanho do dataset:** ~7.000 registros pode limitar a generalização de modelos complexos
- **Features estáticas:** Dataset não captura dinâmica temporal (ex.: mudanças recentes de uso)

## 8. Cenários de Falha

| Cenário | Impacto | Mitigação |
|---------|---------|-----------|
| Modelo prediz alto churn para todos | Custo excessivo de retenção | Monitorar taxa de FP, threshold calibrado |
| Modelo não detecta churners reais | Perda de clientes sem ação | Alertar se recall cair abaixo de 0.60 |
| API fora do ar | Equipe de retenção sem dados | Health check + fallback para último batch |
| Data drift silencioso | Degradação gradual | Monitorar distribuição de features e métricas semanalmente |
