import numpy as np
import pandas as pd
import pytest

from nslkdd import config, data


def test_train_shape_and_classes():
    df = data.load_train_frame()
    assert df.shape == (69_833, 50)
    assert df[config.TARGET].value_counts().to_dict() == {"normal": 65_754, "anomaly": 4_079}


def test_text_is_normalized_without_losing_rows():
    df = data.load_train_frame()
    assert set(df["protocol_type"]) == {"tcp", "udp", "icmp"}
    assert df["flag"].nunique() == 11
    assert df["service"].nunique() == 67
    assert df["service"].dropna().eq(df["service"].dropna().str.strip()).all()


def test_test_requires_explicit_flag():
    with pytest.raises(PermissionError):
        data.load_test()
    with pytest.raises(PermissionError):
        data.load_test_frame()


def test_test_schema_is_compatible_with_train():
    train = data.load_train_frame()
    X, ids = data.load_test(final_evaluation=True)
    assert X.shape == (24_333, 41) and ids.is_unique
    for col in config.CATEGORICAL:
        assert set(X[col].dropna()) <= set(train[col].dropna())


def test_only_traffic_attributes_reach_the_model():
    X, y, groups = data.load_train()
    assert list(X.columns) == config.FEATURES and X.shape[1] == 41
    assert not set(X.columns) & (set(config.LEAKY_COLUMNS) | set(config.DERIVED_COLUMNS))
    assert config.TARGET not in X.columns
    assert len(groups) == len(X) == len(y)


def test_duplicate_groups_and_conflicts():
    X, y, groups = data.load_train()
    conflict = data.conflict_mask(groups, y)
    assert len(np.unique(groups)) == 56_766
    assert conflict.sum() == 3_730
    mask = data.evaluation_mask(groups, y)
    assert not conflict[mask].any()
    assert not pd.Series(groups[mask]).duplicated().any()


def test_derived_columns_are_exact_copies():
    df = data.load_train_frame()
    assert np.nanmax(np.abs(df["src_bytes_kb"] - df["src_bytes"] / 1024)) < 1e-9
    assert np.nanmax(np.abs(df["dst_bytes_kb"] - df["dst_bytes"] / 1024)) < 1e-9
    assert np.nanmax(np.abs(df["total_bytes"] - df["src_bytes"] - df["dst_bytes"])) < 1e-9
    assert np.nanmax(np.abs(df["same_srv_pct"] - df["same_srv_rate"] * 100)) < 1e-9
    assert df.groupby("protocol_type")["protocolo_cod"].nunique().eq(1).all()


def test_submission_rejects_wrong_ids_and_labels(tmp_path):
    template = pd.read_csv(config.SUBMISSION_TEMPLATE)
    ids, n = template[config.ID].to_numpy(), len(template)
    path = tmp_path / "sub.csv"
    with pytest.raises(ValueError):
        data.write_submission(ids[::-1], ["normal"] * n, path)
    with pytest.raises(ValueError):
        data.write_submission(ids, ["DoS"] * n, path)
    with pytest.raises(ValueError):
        data.write_submission(ids, ["Normal"] * n, path)  # exige minúsculas
    with pytest.raises(ValueError):
        data.write_submission(ids, [None] * n, path)  # sem valores vazios
    sub = data.write_submission(ids, ["normal"] * n, path)
    assert list(pd.read_csv(path).columns) == [config.ID, config.SUBMISSION_TARGET]
    assert len(sub) == n
    assert b"\r\n" not in path.read_bytes()
    first = data.check_submission(path)
    data.write_submission(ids, ["normal"] * n, path)
    assert data.check_submission(path) == first  # mesma entrada, mesmos bytes


def test_submission_file_follows_team_number_rule():
    if config.TEAM_NUMBER is None:
        assert config.SUBMISSION_FILE is None
        with pytest.raises(ValueError):
            data.write_submission([], [])
    else:
        assert config.SUBMISSION_FILE.name == f"submissao_{config.TEAM_NUMBER}.csv"
