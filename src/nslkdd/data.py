"""Leitura da base da competição, com proteção contra uso prematuro do teste."""
import numpy as np
import pandas as pd

from . import config

_EXPECTED = config.FEATURES + list(config.DERIVED_COLUMNS) + config.LEAKY_COLUMNS


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza caixa e espaços dos categóricos ('TCP', 'sf', 'http '). Não remove linhas."""
    df = df.copy()
    df["protocol_type"] = df["protocol_type"].str.strip().str.lower()
    df["flag"] = df["flag"].str.strip().str.upper()
    df["service"] = df["service"].str.strip()
    return df


def _read(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} não encontrado: coloque a base em {config.DATA_DIR}.")
    df = pd.read_csv(path)
    missing = set(_EXPECTED) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {path.name}: {sorted(missing)}")
    return normalize_text(df)


def load_train_frame() -> pd.DataFrame:
    """treino.csv completo, com texto padronizado (inclui colunas derivadas e de vazamento)."""
    df = _read(config.TRAIN_FILE)
    unknown = set(df[config.TARGET]) - set(config.CLASSES)
    if unknown:
        raise ValueError(f"Rótulos desconhecidos: {sorted(unknown)}")
    return df


def load_test_frame(final_evaluation: bool = False) -> pd.DataFrame:
    """teste_sem_rotulo.csv. Exige final_evaluation=True: o teste só entra na Etapa 4."""
    if not final_evaluation:
        raise PermissionError(
            "O teste é reservado à submissão final. Toda decisão deve ser validada no "
            "treino (CV). Use load_test_frame(final_evaluation=True) apenas no notebook "
            "04, depois de congelar o pipeline."
        )
    return _read(config.TEST_FILE)


def duplicate_groups(X: pd.DataFrame) -> np.ndarray:
    """Identificador de cada conjunto de linhas com os 41 atributos idênticos (NaN == NaN)."""
    hashes = pd.util.hash_pandas_object(X[config.FEATURES], index=False)
    return pd.factorize(hashes)[0]


def conflict_mask(groups, y) -> np.ndarray:
    """True nas linhas cujo grupo de duplicatas tem mais de um rótulo."""
    return pd.Series(np.asarray(y)).groupby(np.asarray(groups)).transform("nunique").to_numpy() > 1


def evaluation_mask(groups, y) -> np.ndarray:
    """Linhas que medem generalização: uma por grupo, sem rótulo contraditório.

    Conflitos são cara ou coroa (mesma linha com os dois rótulos): nenhum modelo acerta,
    e mantê-los esconde as diferenças entre técnicas. Cópias pesariam a mesma linha
    várias vezes, algo que não existe no teste.
    """
    first = ~pd.Series(np.asarray(groups)).duplicated().to_numpy()
    return first & ~conflict_mask(groups, y)


def split_xy(df: pd.DataFrame):
    """Separa X (só os 41 atributos), y e os grupos de duplicatas usados na CV."""
    X = df[config.FEATURES].copy()
    return X, df[config.TARGET].copy(), duplicate_groups(X)


def load_train():
    return split_xy(load_train_frame())


def load_test(final_evaluation: bool = False):
    """Devolve (X, ids) do teste, sem remover nem reordenar linhas."""
    df = load_test_frame(final_evaluation)
    return df[config.FEATURES].copy(), df[config.ID].copy()


def write_submission(ids, y_pred, path=config.SUBMISSION_FILE) -> pd.DataFrame:
    """Valida contra modelo_de_submissao.csv (mesmos ids, mesma ordem) e grava o CSV."""
    template = pd.read_csv(config.SUBMISSION_TEMPLATE)
    sub = pd.DataFrame({config.ID: np.asarray(ids), config.SUBMISSION_TARGET: np.asarray(y_pred)})
    if len(sub) != len(template) or (sub[config.ID].to_numpy() != template[config.ID].to_numpy()).any():
        raise ValueError("Os ids não coincidem com modelo_de_submissao.csv (conjunto ou ordem).")
    invalid = set(sub[config.SUBMISSION_TARGET].unique()) - set(config.CLASSES)
    if invalid:
        raise ValueError(f"Previsões fora de {config.CLASSES}: {sorted(map(str, invalid))}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return sub
