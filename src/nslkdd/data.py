"""Leitura do NSL-KDD com proteção contra uso prematuro do KDDTest+."""
import pandas as pd

from . import config

_TARGETS = {"category", "binary", "attack"}


def _read(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não encontrado. Baixe KDDTrain+.txt e KDDTest+.txt "
            "(https://www.kaggle.com/datasets/hassan06/nslkdd) para data/raw/."
        )
    df = pd.read_csv(path, names=config.COLUMNS)
    unknown = set(df["attack"]) - set(config.ATTACK_TO_CATEGORY)
    if unknown:
        raise ValueError(f"Ataques sem categoria em ATTACK_TO_CATEGORY: {sorted(unknown)}")
    df["category"] = df["attack"].map(config.ATTACK_TO_CATEGORY)
    return df


def load_train_frame() -> pd.DataFrame:
    """KDDTrain+ completo (atributos + attack + difficulty + category)."""
    return _read(config.TRAIN_FILE)


def load_test_frame(final_evaluation: bool = False) -> pd.DataFrame:
    """KDDTest+. Exige final_evaluation=True: o teste só entra na Etapa 4."""
    if not final_evaluation:
        raise PermissionError(
            "KDDTest+ é reservado à avaliação final. Toda decisão deve ser "
            "validada no treino (CV). Use load_test_frame(final_evaluation=True) "
            "apenas no notebook 04, depois de congelar o pipeline."
        )
    return _read(config.TEST_FILE)


def split_xy(df: pd.DataFrame, target: str = "category"):
    """Separa X (41 atributos, sem difficulty) e y. Devolve também attack (grupos)."""
    if target not in _TARGETS:
        raise ValueError(f"target deve ser um de {_TARGETS}")
    y = df[target] if target != "binary" else (df["category"] != "normal").astype(int)
    return df[config.FEATURES].copy(), y.copy(), df["attack"].copy()


def load_train(target: str = "category"):
    return split_xy(load_train_frame(), target)


def load_test(target: str = "category", final_evaluation: bool = False):
    return split_xy(load_test_frame(final_evaluation), target)
