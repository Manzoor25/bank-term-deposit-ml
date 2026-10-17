# =============================================================
# WHAT THIS FILE DOES:
# Checks that the project settings still refuse the cheating column.
# A one-line "cleanup" that removes duration from LEAKAGE_COLUMNS
# would make the model look better and be wrong. This test catches that.
#
# HOW IT CONNECTS TO OTHER FILES:
# - Imports config.py only
# =============================================================

from settings import LEAKAGE_COLUMNS, TARGET_COLUMN, UNUSED_COLUMNS


def test_duration_is_treated_as_leakage():
    assert "duration" in LEAKAGE_COLUMNS


def test_day_is_not_used():
    assert "day" in UNUSED_COLUMNS


def test_target_column_is_y():
    assert TARGET_COLUMN == "y"
