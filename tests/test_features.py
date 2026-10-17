# =============================================================
# WHAT THIS FILE DOES:
# Checks the column rewrites: no cheating column, no -1 sentinel,
# and the campaign count does not include the call we have not made.
#
# HOW IT CONNECTS TO OTHER FILES:
# - Calls add_engineered_columns from features.py
# - Uses the tiny table from tests/tiny_data.py
# =============================================================

from features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    add_engineered_columns,
)
from tests.tiny_data import tiny_bank_frame


def test_duration_and_target_and_day_are_removed():
    prepared = add_engineered_columns(tiny_bank_frame())
    assert "duration" not in prepared.columns
    assert "y" not in prepared.columns
    assert "day" not in prepared.columns


def test_never_contacted_flag_replaces_negative_pdays():
    raw = tiny_bank_frame()
    prepared = add_engineered_columns(raw)

    never_rows = raw["pdays"] < 0
    assert (prepared.loc[never_rows, "never_contacted_before"] == 1).all()
    assert (prepared.loc[never_rows, "pdays"] == 0).all()


def test_campaign_count_excludes_the_call_being_scored():
    raw = tiny_bank_frame()
    prepared = add_engineered_columns(raw)
    assert "campaign" not in prepared.columns
    assert (prepared["contacts_already_made"] == (raw["campaign"] - 1).clip(lower=0)).all()


def test_model_columns_are_all_present():
    prepared = add_engineered_columns(tiny_bank_frame())
    for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        assert column in prepared.columns
