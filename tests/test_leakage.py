import json

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from nslkdd import audit, data
from nslkdd.preprocessing import build_pipeline


def test_samplers_only_touch_fit_data():
    X, y, _ = data.load_train()
    X, y = X.iloc[:3000], y.iloc[:3000]
    pipe = build_pipeline(DecisionTreeClassifier(random_state=0), redundancy="drop_duplicates",
                          balancing="ros")
    pipe.fit(X, y)
    # predict não reamostra: uma previsão por linha recebida
    assert len(pipe.predict(X.iloc[:500])) == 500


def test_every_service_encoding_builds():
    X, y, _ = data.load_train()
    X, y = X.iloc[:3000], y.iloc[:3000]
    for enc in ["onehot", "onehot_rare", "frequency", "target", "ordinal"]:
        p = build_pipeline(DecisionTreeClassifier(random_state=0), service_encoding=enc)
        assert np.isfinite(p.fit(X, y).predict_proba(X.iloc[:10])).all()


def test_scanner_flags_bad_patterns(tmp_path):
    nb = {"cells": [{"cell_type": "code", "source": [
        "scaler.fit(X_test)\n", "X = pd.concat([train, test])\n"]}],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    p = tmp_path / "02_bad.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    levels = [f["nivel"] for f in audit.scan_notebook(p)]
    assert levels.count("ERRO") == 2
