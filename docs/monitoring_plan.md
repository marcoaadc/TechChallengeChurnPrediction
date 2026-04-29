# Plano de Monitoramento — Churn Prediction

## 1. Métricas Monitoradas

### Métricas Operacionais (API)

| Métrica | Descrição | Fonte | Frequência |
|---------|-----------|-------|------------|
| **Latência p50/p95/p99** | Tempo de resposta do endpoint `/predict` | Middleware de latência (logs JSON) | Contínuo |
| **Error rate** | % de requests com status >= 400 | Logs HTTP | Contínuo |
| **Throughput** | Requests por segundo em `/predict` | Logs HTTP | Contínuo |
| **Disponibilidade** | % de uptime do endpoint `/health` | Health check probe | A cada 30s |
| **Uso de memória/CPU** | Recursos consumidos pelo container | Container runtime metrics | Contínuo |

### Métricas de Modelo

| Métrica | Descrição | Fonte | Frequência |
|---------|-----------|-------|------------|
| **Distribuição de predições** | Histograma de `churn_probability` retornadas | Logs de predição | Diária |
| **Taxa de positivos** | % de predições com `churn_prediction=True` | Logs de predição | Diária |
| **Feature drift** | Mudança na distribuição das features de entrada | Dados de input logados | Semanal |
| **Recall real** | Recall medido contra churns confirmados | Dados de feedback (CRM) | Mensal |
| **AUC-ROC real** | AUC-ROC contra dados reais de churn | Dados de feedback (CRM) | Mensal |

## 2. Alertas

### Thresholds baseados nos SLOs

| Alerta | Condição | Severidade | Ação |
|--------|----------|------------|------|
| **Latência alta** | p95 > 200ms por 5 minutos | Warning | Investigar carga; considerar scaling |
| **Latência crítica** | p95 > 500ms por 2 minutos | Critical | Escalar; verificar recursos do container |
| **Error rate alto** | > 5% de requests com erro em 10 minutos | Warning | Verificar logs de erro; checar preprocessor |
| **Error rate crítico** | > 20% de requests com erro em 5 minutos | Critical | Incidente; possível falha no modelo/preprocessor |
| **API indisponível** | `/health` retorna `model_not_loaded` ou timeout | Critical | Reiniciar container; verificar artefatos |
| **Drift de predições** | Taxa de positivos muda > 10pp vs. baseline (26%) | Warning | Investigar dados de entrada; agendar re-treinamento |
| **Recall degradado** | Recall mensal < 0.70 (SLO mínimo) | Critical | Re-treinar modelo com dados recentes |

## 3. Detecção de Data Drift

### Estratégia

Monitorar a distribuição das features de entrada para detectar mudanças silenciosas nos dados que podem degradar o modelo.

### Métodos

| Método | Aplicação | Threshold |
|--------|-----------|-----------|
| **PSI (Population Stability Index)** | Features numéricas (tenure, MonthlyCharges, TotalCharges) | PSI > 0.2 → drift significativo |
| **Teste de Kolmogorov-Smirnov** | Features numéricas — teste estatístico de mudança de distribuição | p-value < 0.05 → drift |
| **Frequência de categorias** | Features categóricas — monitorar proporção de cada categoria | Variação > 5pp vs. treino → investigar |

### Frequência
- **Semanal:** calcular PSI e KS para as 3 features numéricas principais
- **Mensal:** comparar distribuição completa de todas as features vs. dados de treino

## 4. Triggers de Re-treinamento

| Trigger | Condição | Ação |
|---------|----------|------|
| **Agendado** | Mensalmente (conforme SLO) | Re-treinar com dados do último mês |
| **Drift detectado** | PSI > 0.2 em qualquer feature numérica | Agendar re-treinamento urgente |
| **Performance degradada** | Recall < 0.70 ou AUC-ROC < 0.80 em dados reais | Re-treinar e investigar causa raiz |
| **Mudança de negócio** | Novo plano, mudança de preços, campanha grande | Avaliar impacto e re-treinar se necessário |

### Processo de re-treinamento
1. Coletar dados atualizados do CRM/data warehouse
2. Executar pipeline de treino (notebook `02_modeling.ipynb` ou script equivalente)
3. Comparar métricas do novo modelo vs. modelo em produção
4. Se novo modelo for superior: atualizar artefatos (.pt, .joblib) e reiniciar API
5. Registrar experimento no MLflow para rastreabilidade

## 5. Playbook de Resposta a Incidentes

### Incidente: API indisponível

```
1. Verificar status do container (docker ps / kubectl get pods)
2. Checar logs do container para erros de startup
3. Verificar se arquivos .pt e .joblib existem no diretório models/
4. Se artefatos corrompidos: re-deploy com última versão válida
5. Se problema de infraestrutura: escalar para time de infra
6. Comunicar equipe de retenção sobre indisponibilidade
```

### Incidente: Error rate alto (> 20%)

```
1. Verificar logs de erro (grep por status 422, 500, 503)
2. Se 422 (pré-processamento): checar se dados de entrada mudaram de formato
3. Se 503 (modelo não carregado): reiniciar container
4. Se 500 (inferência): verificar integridade do modelo .pt
5. Testar manualmente com curl no endpoint /predict
6. Se problema persistir: rollback para versão anterior do modelo
```

### Incidente: Drift de predições detectado

```
1. Calcular PSI das features numéricas vs. dados de treino
2. Verificar se houve mudança de negócio (novo plano, campanha, etc.)
3. Comparar distribuição de predições: últimos 7 dias vs. baseline
4. Se drift confirmado: agendar re-treinamento com dados recentes
5. Comunicar stakeholders sobre possível degradação temporária
6. Após re-treinamento: validar métricas e atualizar modelo em produção
```

### Incidente: Recall abaixo do SLO (< 0.70)

```
1. Coletar dados de churns confirmados do último período
2. Avaliar predições do modelo contra dados reais
3. Calcular métricas atualizadas (recall, precision, AUC-ROC)
4. Identificar padrões de falha (quais clientes não foram detectados?)
5. Re-treinar modelo com dados atualizados
6. Ajustar threshold se necessário (find_optimal_threshold)
7. Deploy do novo modelo após validação
```

## 6. Dashboard de Monitoramento (Proposta)

### Painéis sugeridos

| Painel | Métricas | Visualização |
|--------|----------|--------------|
| **Operacional** | Latência p50/p95, throughput, error rate, uptime | Time series + indicadores |
| **Predições** | Distribuição de probabilidades, taxa de positivos, volume diário | Histograma + time series |
| **Drift** | PSI por feature, KS p-values | Gauge + time series |
| **Performance** | Recall, AUC-ROC, custo total (quando dados reais disponíveis) | Scorecard mensal |

### Ferramentas sugeridas
- **Logs:** ELK Stack (Elasticsearch + Logstash + Kibana) ou CloudWatch
- **Métricas:** Prometheus + Grafana
- **Alertas:** PagerDuty ou Opsgenie integrado com Grafana
