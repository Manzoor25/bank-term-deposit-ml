# =============================================================
# WHAT THIS FILE DOES:
# This is the file Azure ML calls when a request hits the endpoint.
# Azure looks for two function names specifically: init and run.
# You do not call them yourself in the cloud. Azure does.
#
# HOW IT CONNECTS TO OTHER FILES:
# - predict.py holds the real scoring rules. This file only adapts
#   them to Azure's expected shape.
# - features.py is used indirectly, through predict.score_records
# - The model file is NOT in this folder. Azure places the registered
#   model in a folder and tells us where via AZUREML_MODEL_DIR.
#
# WHY THIS FILE IS SO SHORT:
# If scoring rules lived only here, local tests could pass while the
# endpoint did something else. One function, two doors.
# =============================================================

import json
import os

# Azure adds the uploaded code folder to the import path, so the
# project-root files (predict.py, features.py, config.py) import
# as if we were running from that folder.
from predict import load_model_card, load_pipeline, score_records

# The model is loaded once, in init, and reused for every request.
# Loading it inside run() would repeat disk work on every call.
pipeline = None
model_card = {}


# --------------------------------------------------------------
# FUNCTION: _find_file
# WHAT IT DOES: Walks a folder until it finds a file by name
# WHY WE NEED IT: Azure sometimes nests the registered model inside
#                 an extra folder. Searching is more reliable than
#                 guessing that path.
# INPUTS: root folder, file name
# OUTPUTS: The full path, or None if it is not there
# --------------------------------------------------------------
def _find_file(root, file_name):
    for directory, _subdirs, file_names in os.walk(root):
        if file_name in file_names:
            return os.path.join(directory, file_name)
    return None


# --------------------------------------------------------------
# FUNCTION: init
# WHAT IT DOES: Loads the model once, when the endpoint starts
# WHY WE NEED IT: Azure ML requires a function with this exact name
# INPUTS: None. The model location comes from an environment variable.
# OUTPUTS: None. It stores the model in the module-level variables above.
# --------------------------------------------------------------
def init():
    global pipeline, model_card
    model_root = os.environ.get("AZUREML_MODEL_DIR", "models")
    pipeline_path = _find_file(model_root, "pipeline.joblib")
    if pipeline_path is None:
        raise FileNotFoundError(
            f"pipeline.joblib was not inside AZUREML_MODEL_DIR ({model_root})."
        )
    pipeline = load_pipeline(pipeline_path)

    card_path = _find_file(model_root, "model_card.json")
    model_card = load_model_card(card_path) if card_path else {}


# --------------------------------------------------------------
# FUNCTION: run
# WHAT IT DOES: Scores one HTTP request body
# WHY WE NEED IT: Azure ML requires a function with this exact name
# INPUTS: raw_data, the request body as text (or a dict in local tests)
# OUTPUTS: A dictionary Azure turns into JSON
#
# Expected body:
#   {"inputs": [ {customer fields...}, {another customer...} ]}
# --------------------------------------------------------------
def run(raw_data):
    if pipeline is None:
        init()

    if isinstance(raw_data, str):
        payload = json.loads(raw_data)
    else:
        payload = raw_data

    if not isinstance(payload, dict) or "inputs" not in payload:
        raise ValueError("Request JSON must be an object with an 'inputs' list.")

    predictions = score_records(pipeline, payload["inputs"])
    return {
        "model_name": model_card.get("winner", "unknown"),
        "predictions": predictions,
    }


if __name__ == "__main__":
    print("Azure ML calls init() and run(). Use python main.py predict locally.")
