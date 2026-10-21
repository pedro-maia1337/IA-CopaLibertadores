---
name: nslkdd-base
description: Fatos verificados sobre a base NSL-KDD (KDDTrain+/KDDTest+) e a API do pacote local nslkdd para carregá-la. Use sempre que for ler os dados, mapear ataques para as 5 classes, citar números da base, discutir atributos (protocol_type, service, flag, difficulty, num_outbound_cmds), ataques inéditos do teste ou mudança de distribuição treino→teste.
---

# NSL-KDD — o que é preciso saber

## Carregar (nunca reimplementar a leitura)
```python
from nslkdd import load_train, load_train_frame, config
X, y, attack = load_train()            # X: 41 atributos; y: 5 classes; attack: nome original
df = load_train_frame()                # DataFrame completo, inclui 'difficulty' e 'category'
# Só no notebook 04:
from nslkdd import load_test
X_te, y_te, attack_te = load_test(final_evaluation=True)
```
`config` expõe `FEATURES`, `CATEGORICAL`, `NUMERIC`, `BINARY`, `HEAVY_TAILED`, `CLASSES`, `ATTACK_TO_CATEGORY`, `RANDOM_STATE`.

## Números verificados (use-os no texto; não invente outros)
| Classe | KDDTrain+ | % | KDDTest+ | % |
|---|---|---|---|---|
| normal | 67.343 | 53,5 | 9.711 | 43,1 |
| DoS | 45.927 | 36,5 | 7.458 | 33,1 |
| Probe | 11.656 | 9,3 | 2.421 | 10,7 |
| R2L | 995 | 0,8 | 2.754 | 12,2 |
| U2R | 52 | 0,04 | 200 | 0,9 |
| **Total** | **125.973** | | **22.544** | |

- 41 atributos + `attack` + `difficulty`. Sem valores ausentes (verificar e registrar mesmo assim).
- Categóricos: `protocol_type` (3), `flag` (11), `service` (70 no treino, 64 no teste, nenhum inédito no teste).
- `num_outbound_cmds` é constante (0) no treino → remover. `is_host_login` tem 1 único valor 1; `su_attempted` tem valor 2 (59 linhas) num atributo que deveria ser binário — inconsistência a discutir.
- Cauda longa: `src_bytes` mediana 44, máximo 1,38 bi → `log1p` antes de escalar.
- Redundância no treino (ignorando `difficulty`): 16 linhas com atributos duplicados, 15 grupos **inconsistentes** (mesmos 41 atributos, rótulos diferentes). 658 linhas do treino têm cópia exata no teste (contradiz parcialmente o enunciado; apenas relatar — não mexer no teste).
- `difficulty` (0–21) = quantos de 21 classificadores de referência acertaram o registro. É derivado do processo de rotulagem e indisponível em produção → **vazamento, descartar** (média 20,3 em normal vs 18,6 em ataques; eta com a classe ≈ 0,62).

## Mudança de distribuição treino → teste (o desafio real da competição)
- 17 ataques existem **só no teste**: apache2, mailbomb, processtable, udpstorm (DoS); mscan, saint (Probe); named, sendmail, snmpgetattack, snmpguess, worm, xlock, xsnoop (R2L); httptunnel, ps, sqlattack, xterm (U2R). `spy` e `warezclient` existem só no treino.
- R2L no treino é 89% `warezclient` (890/995); no teste é dominado por `guess_passwd` (1.231) e `warezmaster` (944), que no treino têm 53 e 20 exemplos. U2R no teste: 133 de 200 são `httptunnel` (inédito).
- 54 linhas de `snmpgetattack` no teste são idênticas a linhas `normal` do teste → teto de desempenho inevitável para R2L.
- Consequência: CV estratificada no treino dá F1-macro ≈ 0,91–0,94; no teste, modelos típicos ficam bem abaixo (literatura: ~0,5–0,65). Quem vence é quem melhora **recall de R2L/U2R sem destruir a precisão**, não quem otimiza a 4ª casa decimal da CV.
- Diagnóstico honesto disponível no treino: `make_cv("attack_group", attack=attack)` (cada tipo de ataque inteiro em um fold). Ver skill `etapa3-modelagem`.

## Mapeamento ataque → classe
Fica em `config.ATTACK_TO_CATEGORY` (cobre os 40 nomes do treino e do teste; o carregador falha se surgir nome desconhecido). `httptunnel` em U2R e `worm`, `snmp*` em R2L seguem o mapeamento mais usado na literatura (gera 200 U2R / 2.754 R2L no teste). Detalhes e dicionário de atributos: `references/atributos.md`.

## Pontos para confirmar com a professora (registrar a resposta em CLAUDE.md)
- Alvo = 5 categorias (assumido; o enunciado fala em U2R com "poucas dezenas" de instâncias, o que só vale para categorias).
