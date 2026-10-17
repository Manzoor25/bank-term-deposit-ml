# =============================================================
# WHAT THIS FILE DOES:
# Checks the scoring contract: required columns, ignored duration,
# and the Azure entry point returning the same shape.
#
# HOW IT CONNECTS TO OTHER FILES:
# - predict.py scores rows
# - scoring/score.py is what Azure calls
# - train.py produces the tiny model used here
# =============================================================

import json

import pytest

from predict import score_records
from scoring.score import run
from tests.tiny_data import one_customer, tiny_bank_frame
from train import sqlite_tracking_uri, train_and_select


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    # scope="module" means we train the tiny model once for this file,
    # not once per test. The real training is a different command.
    folder = tmp_path_factory.mktemp("trained")
    summary = train_and_select(
        tiny_bank_frame(),
        tracking_uri=sqlite_tracking_uri(folder / "mlflow.db"),
        experiment_name="unit-test-predict",
        small=True,
        output_dir=folder,
        reports_dir=folder,
    )
    from predict import load_pipeline

    return load_pipeline(summary["model_path"])


def test_missing_column_is_a_loud_error(pipeline):
    customer = one_customer()
    del customer["job"]
    with pytest.raises(ValueError, match="job"):
        score_records(pipeline, [customer])


def test_duration_does_not_change_the_score(pipeline):
    short_call = score_records(pipeline, [one_customer(duration=5)])
    long_call = score_records(pipeline, [one_customer(duration=5000)])
    assert short_call == long_call


def test_score_script_returns_a_probability(pipeline, monkeypatch):
    # The Azure script keeps the model in a module-level variable.
    # The test plugs our tiny pipeline in, so we do not need Azure.
    import scoring.score as score_module

    monkeypatch.setattr(score_module, "pipeline", pipeline)
    monkeypatch.setattr(score_module, "model_card", {"winner": "tiny"})

    response = run(json.dumps({"inputs": [one_customer()]}))
    assert response["model_name"] == "tiny"
    assert len(response["predictions"]) == 1
    probability = response["predictions"][0]["probability"]
    assert 0.0 <= probability <= 1.0
