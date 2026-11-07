---
name: nslkdd-base
description: Fatos verificados sobre a base da competição (treino.csv / teste_sem_rotulo.csv, derivada do NSL-KDD) e a API do pacote local nslkdd para carregá-la. Use sempre que for ler os dados, citar números da base, discutir atributos (protocol_type, service, flag, colunas novas, ausentes, valores impossíveis, duplicatas, rótulos contraditórios) ou preparar a submissão.
---

# Base da competição — o que é preciso saber

A base foi trocada em 31/10/2026 e o enunciado foi atualizado em 07/11/2026. O problema mudou: agora é **binário**, com sujeira injetada e colunas de vazamento. Nada do NSL-KDD original (5 classes, `attack`, `difficulty`, CV por grupo de ataque, 17 ataques inéditos) vale aqui.

## Carregar (nunca reimplementar a leitura)
```python
from nslkdd import load_train, load_train_frame, config
X, y, groups = load_train()   # X: 41 atributos; y: normal/anomaly; groups: grupo de duplicatas para a CV
df = load_train_frame()       # DataFrame completo, inclui colunas novas (só para EDA/auditoria)
# Só no notebook 04:
from nslkdd import load_test, write_submission
X_te, ids = load_test(final_evaluation=True)
write_submission(ids, y_pred)  # valida contra modelo_de_submissao.csv e grava results/submissao_<TEAM_NUMBER>.csv
check_submission()             # relê, valida o formato da Seção 3.1 e devolve o SHA-256
```
`config` expõe `FEATURES`, `CATEGORICAL`, `LOW_CARDINALITY`, `NUMERIC`, `BINARY`, `RATES`, `HEAVY_TAILED`, `DOMAIN_MAX`, `LEAKY_COLUMNS`, `DERIVED_COLUMNS`, `CLASSES`, `POSITIVE`, `RANDOM_STATE`.
`load_*` padronizam o texto (caixa e espaços) e não removem linhas.

## Arquivos (`data/new-database/`)
| Arquivo | Conteúdo |
|---|---|
| `treino.csv` | 69.833 linhas, 49 colunas + `class` |
| `teste_sem_rotulo.csv` | 24.333 linhas, `id` + as mesmas 49 colunas |
| `modelo_de_submissao.csv` | `id,class_prevista` (a submissão precisa dos mesmos ids, na mesma ordem) |
| `treino.csv.pdf` | impressão do próprio CSV (1.205 págs); não usar |

## Números verificados (use-os no texto; não invente outros)
| Classe | Linhas brutas | % |
|---|---|---|
| normal | 65.754 | 94,2 |
| anomaly | 4.079 | 5,8 |
"Tudo normal": 94,2% de acurácia, F1-macro 0,485. Razão 16:1.

- **Colunas (49):** 41 de rede (`config.FEATURES`) + 5 derivadas + 3 de vazamento.
- **Vazamento (fora do modelo):** `score_ids_legado`, `tipo_alerta_soc`, `tempo_resposta_analista_min` são cópias ruidosas do rótulo, posteriores ao evento. Evidência (HGB, CV agrupada): 41 de rede F1 0,970; + as três 0,995; **só as três, sem rede, 0,992**. Entre conexões `normal`, score alto ou baixo não muda em nada a probabilidade vista pelo tráfego. Fora dos conflitos, score>0,5 & ALERTA dá 2.082 anomaly contra 27 normal. Estão no teste também: usá-las subiria o ranking e derrubaria a nota metodológica.
- **Derivadas (fora do modelo):** `src_bytes_kb`, `dst_bytes_kb`, `total_bytes`, `same_srv_pct`, `protocolo_cod` são cópias exatas, com os mesmos ausentes. Não recuperam nada.
- **Grafias:** `protocol_type` 6→3 valores, `flag` 19→11, `service` 96→67 após padronizar.
- **Ausentes:** 24 dos 41 atributos, 93,8% das linhas têm ao menos um. Maiores: `src_bytes` 40,0%, `dst_host_srv_count` 39,9%, `dst_host_serror_rate` 30,6%, `dst_host_same_src_port_rate` 28,0%. A maioria é compatível com MCAR. **`dst_host_serror_rate` é MNAR**: anomaly 8,5% se ausente vs 4,6% se presente; ausência sobe de ~30% para 52,7% quando `dst_host_srv_serror_rate` = 1.
- **Valores impossíveis (8,1% das linhas):** taxas 1,5/2/5/10 em 5 colunas de taxa, contagens multiplicadas por 10/100/1000 (`count`, `srv_count`, `dst_host_count`, `duration`), negativos em `duration` (358) e `src_bytes` (189). Taxa de anomalia igual à da base (5,9% vs 5,8%): erro de medição, não sinal. `su_attempted` = 2 existe no NSL-KDD original: manter.
- **Duplicatas:** 13.067 cópias (18,7% das linhas), 56.766 linhas distintas.
- **Rótulos contraditórios:** 1.702 grupos (3.730 linhas) com a mesma linha rotulada com as duas classes, 50,8% `anomaly` contra 3,3% no resto. Um modelo só de tráfego decide 98,5% deles com confiança; ~57% parecem ataques reais (968, contra 1.817 anomalias limpas).
- **Outliers (IQR×3 em log):** 14,6% das linhas, 11,8% das anomalias, com taxa de anomalia *menor* que a do resto (4,7% vs 6,0%). Diferente do NSL-KDD original: não concentram ataques. Ainda assim não removê-los: são tráfego real, e a remoção perderia positivos raros.
- **Teto do F1 "bruto":** com os conflitos na conta, o F1-macro fica preso em ~0,88 para qualquer modelo. Isso é aritmética do ruído, não falta de qualidade.
- **Esquema do teste:** categorias contidas nas do treino e sem grafias variantes (`tests/test_data.py`).

## Armadilhas específicas
- Relação falsa: `dst_host_srv_count` **não** é `dst_host_same_srv_rate × dst_host_count` (correlação 0,55); as janelas contam coisas diferentes. Testado e descartado como forma de imputar.
- `data/raw/` contém as linhas do teste com rótulos. **Nunca cruzar** com o teste, nem usar como treino extra (regra 8 do `CLAUDE.md`).
- Testar `id` como atributo não é possível (o treino não tem), e nem faria sentido.

## Dicionário de atributos
`references/atributos.md`.
