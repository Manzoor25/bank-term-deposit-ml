# =============================================================
# WHAT THIS FILE DOES:
# Checks that azure_ml.py is valid Python without importing the
# Azure packages or logging into Azure. GitHub Actions runs this.
#
# HOW IT CONNECTS TO OTHER FILES:
# - Reads azure_ml.py as text and parses it
# =============================================================

import ast
from pathlib import Path


def test_azure_module_has_no_syntax_errors():
    source = Path("azure_ml.py").read_text(encoding="utf-8")
    ast.parse(source)
