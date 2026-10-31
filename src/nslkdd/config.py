"""Constantes do projeto: caminhos, semente, esquema da base e regras de domínio."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "new-database"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = RESULTS / "models"
EXPERIMENT_LOG = RESULTS / "experiments.csv"

TRAIN_FILE = DATA_DIR / "treino.csv"
TEST_FILE = DATA_DIR / "teste_sem_rotulo.csv"
SUBMISSION_TEMPLATE = DATA_DIR / "modelo_de_submissao.csv"
SUBMISSION_FILE = RESULTS / "submissao.csv"

RANDOM_STATE = 42

# Os 41 atributos de tráfego do NSL-KDD: únicos permitidos como preditores.
FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# Cópias exatas de outros atributos (mesmos ausentes): não trazem informação nova.
DERIVED_COLUMNS = {
    "src_bytes_kb": "src_bytes / 1024",
    "dst_bytes_kb": "dst_bytes / 1024",
    "total_bytes": "src_bytes + dst_bytes",
    "same_srv_pct": "same_srv_rate * 100",
    "protocolo_cod": "protocol_type (0=tcp, 1=udp, 2=icmp)",
}
# Cópias ruidosas do rótulo, geradas depois do evento: nunca usar como atributo.
LEAKY_COLUMNS = ["score_ids_legado", "tipo_alerta_soc", "tempo_resposta_analista_min"]

TARGET = "class"
ID = "id"
SUBMISSION_TARGET = "class_prevista"
CLASSES = ["normal", "anomaly"]
POSITIVE = "anomaly"

CATEGORICAL = ["protocol_type", "service", "flag"]
LOW_CARDINALITY = ["protocol_type", "flag"]
BINARY = ["land", "logged_in", "root_shell", "is_host_login", "is_guest_login"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
RATES = [c for c in FEATURES if c.endswith("_rate")]
# Contagens/volumes com cauda longa: candidatos a log1p.
HEAVY_TAILED = [
    "duration", "src_bytes", "dst_bytes", "hot", "num_failed_logins",
    "num_compromised", "num_root", "num_file_creations", "num_access_files",
    "count", "srv_count", "dst_host_count", "dst_host_srv_count",
]
CONSTANT_IN_TRAIN = ["num_outbound_cmds"]

# Limites vêm da definição dos atributos no KDD (janelas de contagem e taxas),
# não de estatísticas da base: taxas em [0, 1], dst_host_* satura em 255,
# count/srv_count em 511 e duration no máximo documentado do NSL-KDD.
DOMAIN_MAX = {
    **{c: 1.0 for c in RATES},
    "count": 511, "srv_count": 511,
    "dst_host_count": 255, "dst_host_srv_count": 255,
    "duration": 58329,
}
