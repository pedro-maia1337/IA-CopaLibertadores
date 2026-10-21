"""Constantes do projeto: caminhos, semente, esquema da base e mapeamento de ataques."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = RESULTS / "models"
EXPERIMENT_LOG = RESULTS / "experiments.csv"

TRAIN_FILE = DATA_RAW / "KDDTrain+.txt"
TEST_FILE = DATA_RAW / "KDDTest+.txt"

RANDOM_STATE = 42

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
COLUMNS = FEATURES + ["attack", "difficulty"]

CATEGORICAL = ["protocol_type", "service", "flag"]
BINARY = ["land", "logged_in", "root_shell", "is_host_login", "is_guest_login"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
# Contagens/volumes com cauda longa: candidatos a log1p.
HEAVY_TAILED = [
    "duration", "src_bytes", "dst_bytes", "hot", "num_failed_logins",
    "num_compromised", "num_root", "num_file_creations", "num_access_files",
    "count", "srv_count", "dst_host_count", "dst_host_srv_count",
]

# Nunca usar como atributo preditivo: 'difficulty' é metadado derivado do
# rótulo (quantos classificadores de referência acertaram o registro).
LEAKY_COLUMNS = ["difficulty"]
CONSTANT_IN_TRAIN = ["num_outbound_cmds"]

CLASSES = ["normal", "DoS", "Probe", "R2L", "U2R"]

ATTACK_TO_CATEGORY = {
    "normal": "normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "mailbomb": "DoS", "processtable": "DoS",
    "udpstorm": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "named": "R2L", "sendmail": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "worm": "R2L", "xlock": "R2L", "xsnoop": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "httptunnel": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}
