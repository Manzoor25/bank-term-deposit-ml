# =============================================================
# WHAT THIS FILE DOES:
# Loads the saved model and scores one or more customer rows.
# This is the same function the Azure endpoint uses, so a bug
# found by a local test is a bug the endpoint will not have.
#
# HOW IT CONNECTS TO OTHER FILES:
# - features.py prepares the row (drops duration, fixes pdays)
# - train.py writes models/pipeline.joblib, which we load here
# - scoring/score.py is the Azure wrapper around score_records()
# - tests/test_predict.py checks the contract
# =============================================================

import json
from pathlib import Path

import joblib
import pandas as pd

from config import DECISION_THRESHOLD, MODEL_DIR, REQUIRED_RAW_COLUMNS
from features import add_engineered_columns


# --------------------------------------------------------------
# FUNCTION: load_pipeline
# WHAT IT DOES: Reads the saved assembly line from disk
# WHY WE NEED IT: Training is slow. Scoring should not retrain.
# INPUTS: model_path, optional. Defaults to models/pipeline.joblib
# OUTPUTS: The fitted sklearn Pipeline
# --------------------------------------------------------------
def load_pipeline(model_path=None):
    path = Path(model_path) if model_path else MODEL_DIR / "pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model at {path}. Train one first with: python main.py train"
        )
    return joblib.load(path)


# --------------------------------------------------------------
# FUNCTION: load_model_card
# WHAT IT DOES: Reads the small JSON note saved next to the model
# WHY WE NEED IT: The response can name the winning algorithm
#                 without opening the model object
# INPUTS: card_path, optional
# OUTPUTS: A dictionary. Empty if the note is missing.
# --------------------------------------------------------------
def load_model_card(card_path=None):
    path = Path(card_path) if card_path else MODEL_DIR / "model_card.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------
# FUNCTION: score_records
# WHAT IT DOES: Scores a list of customer dictionaries
# WHY WE NEED IT: Both the local command and the Azure endpoint
#                 need one definition of "a valid request"
# INPUTS:
#   pipeline: a fitted Pipeline from train.py
#   records: a list of dictionaries, one customer each
#   threshold: probability cutoff for the yes/no label
# OUTPUTS: A list of dictionaries with probability and a yes/no call
#
# LEARNING NOTE: We raise ValueError when a required column is missing.
# The alternative is to guess. Guessing would make the endpoint look
# healthy while returning nonsense. A loud error is the kinder behavior.
# --------------------------------------------------------------
def score_records(pipeline, records, threshold=DECISION_THRESHOLD):
    if not isinstance(records, list) or not records:
        raise ValueError("Send a non-empty list of customers under the key 'inputs'.")

    frame = pd.DataFrame(records)
    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{missing}. duration and day are optional and are ignored."
        )

    prepared = add_engineered_columns(frame)
    # Column 1 of predict_proba is the probability of class 1 ("yes").
    probabilities = pipeline.predict_proba(prepared)[:, 1]

    scored = []
    for probability in probabilities:
        probability = float(probability)
        scored.append(
            {
                "will_subscribe": bool(probability >= threshold),
                "probability": round(probability, 4),
                "threshold": threshold,
            }
        )
    return scored


if __name__ == "__main__":
    print("predict.py is a library. Use: python main.py predict")
