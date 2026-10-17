# =============================================================
# WHAT THIS FILE DOES:
# Trains the three tiny models on fake rows and checks that a winner
# is saved. This does not prove the real model is accurate. It proves
# the training path runs and writes the files the endpoint needs.
#
# HOW IT CONNECTS TO OTHER FILES:
# - train.py does the work
# - evaluate.py's winner rule is checked directly too
# =============================================================

import json

from evaluate import choose_winner
from tests.tiny_data import tiny_bank_frame
from train import sqlite_tracking_uri, train_and_select


def test_choose_winner_uses_the_highest_practice_score():
    assert choose_winner({"logistic_regression": 0.2, "random_forest": 0.4}) == "random_forest"


def test_train_and_select_writes_a_model_and_a_table(tmp_path):
    summary = train_and_select(
        tiny_bank_frame(),
        tracking_uri=sqlite_tracking_uri(tmp_path / "mlflow.db"),
        experiment_name="unit-test-bank",
        small=True,
        cv_folds=3,
        output_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
    )

    assert summary["winner"] in {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    }
    assert (tmp_path / "models" / "pipeline.joblib").exists()
    card = json.loads((tmp_path / "models" / "model_card.json").read_text(encoding="utf-8"))
    assert card["winner"] == summary["winner"]
    assert "duration" in card["excluded_because_leakage"]

    results = (tmp_path / "reports" / "results.md").read_text(encoding="utf-8")
    assert "logistic_regression" in results
    assert "random_forest" in results
    assert "gradient_boosting" in results
    assert "(selected)" in results
