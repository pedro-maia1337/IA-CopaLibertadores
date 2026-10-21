import pytest

from nslkdd import config, data


def test_train_shape_and_mapping():
    df = data.load_train_frame()
    assert df.shape[0] == 125_973
    assert df["category"].notna().all()
    assert set(df["category"]) == set(config.CLASSES)


def test_test_requires_explicit_flag():
    with pytest.raises(PermissionError):
        data.load_test()


def test_test_mapping_complete():
    df = data.load_test_frame(final_evaluation=True)
    assert df.shape[0] == 22_544
    assert df["category"].value_counts().to_dict() == {
        "normal": 9711, "DoS": 7458, "R2L": 2754, "Probe": 2421, "U2R": 200}


def test_difficulty_never_in_features():
    X, y, _ = data.load_train()
    assert "difficulty" not in X.columns and "attack" not in X.columns
    assert X.shape[1] == 41
