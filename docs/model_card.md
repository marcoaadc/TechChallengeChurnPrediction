# Model Card — Churn Prediction MLP

## 1. Informações do Modelo

| Campo | Valor |
|-------|-------|
| **Nome** | ChurnMLP |
| **Tipo** | Multi-Layer Perceptron (classificação binária) |
| **Framework** | PyTorch 2.x |
| **Arquitetura** | Input(44) → Linear(64) → BatchNorm → ReLU → Dropout(0.3) → Linear(32) → BatchNorm → ReLU → Dropout(0.3) → Linear(1) |
| **Loss** | BCEWithLogitsLoss com pos_weight=2.76 |
| **Otimizador** | Adam (lr=1e-3, weight_decay=1e-4) |
| **Regularização** | Dropout(0.3), L2 (weight_decay), BatchNorm, Early Stopping (patience=15) |
| **LR Scheduler** | ReduceLROnPlateau (factor=0.5, patience=5) |
| **Threshold de classificação** | 0.24 (calibrado por custo de negócio) |
| **Versão** | 1.0.0 |

## 2. Uso Pretendido

### Uso primário
Identificar clientes de telecomunicações com alto risco de cancelamento (churn), permitindo ações proativas de retenção pela equipe comercial.

### Usuários pretendidos
- **Equipe de Retenção:** recebe lista priorizada de clientes em risco
- **Atendimento ao Cliente:** abordagem personalizada durante interações
- **Diretoria Comercial:** monitoramento de taxa de churn e ROI das ações de retenção

### Usos fora do escopo
- Decisões automatizadas de encerramento de conta
- Precificação discriminatória baseada em risco de churn
- Aplicação em setores fora de telecomunicações sem re-treinamento

## 3. Dados de Treinamento

| Campo | Valor |
|-------|-------|
| **Dataset** | Telco Customer Churn (IBM/Kaggle) |
| **Registros** | 7.032 (após limpeza de 11 registros com TotalCharges inválido) |
| **Split** | 80% treino (5.625) / 20% teste (1.407) |
| **Validação** | 15% do treino separado para early stopping |
| **Desbalanceamento** | 26.58% positivo (churn), 73.42% negativo |
| **Features originais** | 19 (3 numéricas + 16 categóricas) |
| **Features engineered** | 4 (total_services_count, tenure_to_charges_ratio, has_no_support, is_new_customer) |
| **Features após encoding** | 44 (StandardScaler numéricas + OneHotEncoder categóricas) |

### Pré-processamento
- `TotalCharges`: conversão de string para numérico, remoção de NaN
- Numéricas: StandardScaler (média 0, desvio padrão 1)
- Categóricas: OneHotEncoder (drop='if_binary', handle_unknown='ignore')

## 4. Resultados de Avaliação

### Comparação de modelos (conjunto de teste, n=1.407)

| Modelo | Accuracy | F1 | Precision | Recall | AUC-ROC | PR-AUC |
|--------|----------|----|-----------|--------|---------|--------|
| DummyClassifier | 0.7342 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2658 |
| Logistic Regression | 0.7356 | 0.6165 | 0.5017 | 0.7995 | 0.8378 | 0.6410 |
| Random Forest | 0.7875 | 0.5504 | 0.6289 | 0.4893 | 0.8151 | 0.6050 |
| Gradient Boosting | 0.7846 | 0.5590 | 0.6134 | 0.5134 | 0.8357 | 0.6501 |
| **MLP (t=0.50)** | 0.7342 | 0.6215 | 0.5000 | 0.8209 | **0.8382** | 0.6381 |
| **MLP (t=0.24)** | 0.6134 | 0.5662 | 0.4034 | **0.9492** | **0.8382** | 0.6381 |

### Métricas de negócio (FN=R$500, FP=R$50)

| Modelo | Custo Total (R$) | Economia vs. sem modelo (R$) |
|--------|------------------|------------------------------|
| MLP (t=0.24) | 35.750 | Melhor custo-benefício |
| MLP (t=0.50) | 48.850 | — |
| Logistic Regression | ~49.000 | — |

### SLOs atingidos

| SLO | Target | Resultado | Status |
|-----|--------|-----------|--------|
| AUC-ROC | > 0.80 | 0.8382 | Atingido |
| Recall | > 0.70 | 0.9492 (t=0.24) | Atingido |
| Latência p95 | < 200ms | ~30ms (local) | Atingido |

## 5. Análise de Viés e Equidade

### Subgrupos avaliados

O dataset contém variáveis demográficas sensíveis que podem levar a tratamento desigual:

| Subgrupo | Proporção no dataset | Risco identificado |
|----------|---------------------|--------------------|
| **SeniorCitizen=1** | ~16% | Idosos têm taxa de churn mais alta (~41% vs ~24%). O modelo pode ser mais agressivo em classificá-los como churn, levando a mais ofertas de retenção direcionadas a esse grupo. |
| **gender** | ~50/50 | Distribuição equilibrada. Não há evidência de viés significativo por gênero nas métricas de churn. |
| **Partner=No** | ~52% | Clientes sem parceiro têm maior propensão a churn. O modelo reflete esse padrão dos dados. |

### Recomendações de equidade
- Monitorar recall e precision por subgrupo demográfico em produção
- Avaliar se ações de retenção baseadas no modelo não criam tratamento discriminatório
- Não usar o modelo para decisões que afetem negativamente o cliente (cancelamento forçado, aumento de preço)

## 6. Limitações

1. **Tamanho do dataset:** ~7.000 registros limita a capacidade de generalização, especialmente para subgrupos pequenos (ex: SeniorCitizen=1 com churn)
2. **Features estáticas:** o dataset não captura dinâmica temporal (mudanças recentes de uso, chamadas ao SAC, reclamações)
3. **Snapshot único:** dados representam um momento específico; mudanças de mercado, concorrência ou planos podem invalidar os padrões
4. **Precision baixa (0.40 com t=0.24):** ~60% dos clientes flagrados como churn não cancelariam de fato — gera custo operacional de retenção desnecessária
5. **Sem validação cruzada no MLP:** métricas baseadas em um único split train/test, sem intervalo de confiança para a rede neural
6. **Domínio restrito:** treinado exclusivamente em dados de telecomunicações; não transferível para outros setores

## 7. Cenários de Falha

| Cenário | Probabilidade | Impacto | Mitigação |
|---------|---------------|---------|-----------|
| Modelo prediz alto churn para quase todos (threshold muito baixo) | Média | Custo excessivo de retenção, fadiga da equipe | Monitorar taxa de FP; ajustar threshold se FP > 60% |
| Modelo não detecta churners reais (drift silencioso) | Alta (longo prazo) | Perda de clientes sem ação | Monitorar recall semanalmente; alertar se < 0.70 |
| Novo plano/oferta muda padrão de churn | Alta | Modelo desatualizado | Re-treinar mensalmente; monitorar distribuição de features |
| API indisponível | Baixa | Equipe sem predições | Health check, fallback para último batch |
| Dados de entrada inválidos | Baixa | Erro 422 | Validação Pydantic no endpoint |

## 8. Recomendações

- **Re-treinamento mensal** com dados atualizados para adaptar a mudanças de comportamento
- **Monitorar recall por subgrupo** (SeniorCitizen, Contract type) para detectar degradação localizada
- **Coletar feedback** da equipe de retenção sobre a qualidade das predições
- **Avaliar modelos complementares** (Gradient Boosting com mais features) para comparação contínua
- **Não automatizar decisões** baseadas apenas no modelo; usar como ferramenta de apoio à decisão humana
