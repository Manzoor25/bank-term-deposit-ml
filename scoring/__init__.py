# =============================================================
# WHAT THIS FILE DOES:
# Marks the scoring folder as a Python package so tests can import
# scoring.score. Azure itself loads score.py by file path, so this
# file is for our tests, not for Azure.
#
# HOW IT CONNECTS TO OTHER FILES:
# - tests import scoring.score
# - score.py next to this file is the Azure entry point
# =============================================================
