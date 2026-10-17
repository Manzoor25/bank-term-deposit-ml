# =============================================================
# WHAT THIS FILE DOES:
# Builds a tiny fake bank table for tests. Tests must not download
# the real file, and they must not depend on the internet.
#
# HOW IT CONNECTS TO OTHER FILES:
# - tests/test_features.py, test_train.py, and test_predict.py import
#   tiny_bank_frame() from here
# =============================================================

import pandas as pd


# --------------------------------------------------------------
# FUNCTION: tiny_bank_frame
# WHAT IT DOES: Returns 24 fake customers, half yes and half no
# WHY WE NEED IT: Training tests need both answers, or the metrics
#                 that measure ranking cannot be computed
# INPUTS: None
# OUTPUTS: A pandas DataFrame with the same columns as the real CSV
# --------------------------------------------------------------
def tiny_bank_frame():
    jobs = ["technician", "management", "admin.", "blue-collar"]
    rows = []
    for index in range(24):
        subscribed = index % 2 == 0
        rows.append(
            {
                "age": 30 + index,
                "job": jobs[index % len(jobs)],
                "marital": "married" if subscribed else "single",
                "education": "secondary" if subscribed else "tertiary",
                "default": "no",
                "balance": 500 + index * 25,
                "housing": "yes" if index % 3 == 0 else "no",
                "loan": "no" if index % 5 else "yes",
                "contact": "cellular" if subscribed else "unknown",
                "day": (index % 28) + 1,
                "month": "may" if subscribed else "aug",
                "duration": 50 + index,
                "campaign": 1 + (index % 4),
                "pdays": -1 if index % 4 == 0 else 15 + index,
                "previous": 0 if index % 4 == 0 else 1,
                "poutcome": "unknown" if index % 4 == 0 else "failure",
                "y": "yes" if subscribed else "no",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------
# FUNCTION: one_customer
# WHAT IT DOES: Returns one request dictionary, like sample_request.json
# WHY WE NEED IT: Scoring tests need a valid customer
# INPUTS: duration, so a test can prove that column is ignored
# OUTPUTS: A dictionary
# --------------------------------------------------------------
def one_customer(duration=100):
    return {
        "age": 42,
        "job": "technician",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "balance": 1500,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "duration": duration,
        "campaign": 2,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown",
    }
