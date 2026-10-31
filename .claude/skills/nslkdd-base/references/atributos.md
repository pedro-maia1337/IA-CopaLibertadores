# Dicionário de atributos NSL-KDD (Tavallaee et al., 2009; KDD Cup 99)

Os 41 atributos abaixo são os únicos preditores da base atual. Ela acrescenta 5 colunas derivadas e 3 de vazamento, fora do modelo (ver `SKILL.md`). Os valores abaixo descrevem o NSL-KDD original; `service` tem 67 valores no `treino.csv`.

## Básicos da conexão TCP (1–9)
| # | Atributo | Tipo | Significado |
|---|---|---|---|
| 1 | duration | contínuo | duração da conexão (s) |
| 2 | protocol_type | categórico (3) | tcp, udp, icmp |
| 3 | service | categórico (70) | serviço de destino (http, ftp_data, private, ...) |
| 4 | flag | categórico (11) | status da conexão (SF, S0, REJ, RSTR, ...) |
| 5 | src_bytes | contínuo | bytes origem→destino |
| 6 | dst_bytes | contínuo | bytes destino→origem |
| 7 | land | binário | 1 se origem e destino (host/porta) iguais |
| 8 | wrong_fragment | contínuo | nº de fragmentos errados |
| 9 | urgent | contínuo | nº de pacotes urgentes |

## Conteúdo (10–22) — relevantes para ataques de uma conexão só
hot, num_failed_logins, logged_in (bin), num_compromised, root_shell (bin), su_attempted (deveria ser bin; tem valor 2), num_root, num_file_creations, num_shells, num_access_files, num_outbound_cmds (constante 0), is_host_login (bin), is_guest_login (bin).

## Tráfego por janela de 2 segundos (23–31) — relevantes para DoS/Probe
count, srv_count, serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate.

## Tráfego por host de destino, últimas 100 conexões (32–41)
dst_host_count, dst_host_srv_count, dst_host_same_srv_rate, dst_host_diff_srv_rate, dst_host_same_src_port_rate, dst_host_srv_diff_host_rate, dst_host_serror_rate, dst_host_srv_serror_rate, dst_host_rerror_rate, dst_host_srv_rerror_rate.

## Redundâncias entre atributos (treino.csv, verificado)
- Spearman > 0,95: `rerror_rate` ~ `srv_rerror_rate` (0,951), `same_srv_rate` ~ `diff_srv_rate` (−0,959)
- grupo SYN-error: `serror_rate` ~ `srv_serror_rate` (Pearson 0,97); `srv_serror_rate` não tem ausentes e cobre boa parte da informação de `serror_rate`
- grupo REJ-error: `rerror_rate` ~ `srv_rerror_rate` (Pearson 0,99)

São atributos redundantes no sentido do slide "Dados inconsistentes e redundantes" (valor deduzível de outro). Candidatos a remoção/redução; árvores lidam bem com eles.

## Por que cada família importa
- Ataques volumétricos e varreduras: muitas conexões em pouco tempo → atributos de janela (23–41) dominam.
- Ataques de uma conexão só: atributos de conteúdo (10–22), `service`, `src_bytes`/`dst_bytes`, `duration`.

## Referência
Tavallaee, M.; Bagheri, E.; Lu, W.; Ghorbani, A. "A Detailed Analysis of the KDD CUP 99 Data Set". IEEE CISDA, 2009.
