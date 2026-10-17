# =============================================================
# WHAT THIS FILE DOES:
# This file is the project's settings notebook. It used to be named
# config.py. That name collides with a module inside the Azure ML
# scoring server, so the endpoint crashed on import. settings.py
# does not collide. It names the
# dataset, the folders, the column we are trying to predict,
# and the columns we refuse to use.
#
# HOW IT CONNECTS TO OTHER FILES:
# - ingest.py uses the download URL and the raw-data folder
# - features.py uses the target column and the columns we drop
# - train.py uses the random seed and the experiment name
# - azure_ml.py uses the Azure resource names when we deploy
#
# WHY ONE FILE FOR SETTINGS:
# If the same number is copied into five files, changing it
# means hunting through five files. Here you change it once.
# =============================================================

import os
from pathlib import Path

# python-dotenv reads a local file named .env, if you have one.
# WHY: your laptop can have different Azure names than a classmate's
# laptop, without either of you editing this Python file.
# The Azure scoring image does not install python-dotenv. The try/except
# lets that image import this file without an extra package.
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

# __file__ is the path of THIS file. .parent is the folder it sits in.
# That folder is the project root, no matter which folder you ran
# the command from.
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"

# Public dataset. No Kaggle login. The zip has another zip inside it.
# ingest.py knows how to unpack both layers.
DATA_PAGE_URL = "https://archive.ics.uci.edu/dataset/222/bank+marketing"
DATA_ZIP_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
RAW_CSV_NAME = "bank-full.csv"

# Citation for the dataset (required by its CC BY 4.0 license):
# Moro, Rita, and Cortez. Bank Marketing. UCI Machine Learning Repository, 2012.
# https://doi.org/10.24432/C5K306

# The question the model answers: did the client subscribe ("yes" / "no")?
TARGET_COLUMN = "y"
POSITIVE_LABEL = "yes"

# duration is how many seconds the phone call already lasted.
# A banker would not know that BEFORE deciding to make the call.
# Keeping it would be like grading a student with the answer key open.
LEAKAGE_COLUMNS = ["duration"]

# day-of-month is not cheating, but it is a brittle calendar quirk.
# We drop it on purpose. The reason is written up in DECISIONS.md.
UNUSED_COLUMNS = ["day"]

# Every column the published CSV is supposed to have.
# ingest.py checks this so a bad download fails loudly, not quietly.
EXPECTED_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "y",
]

# What a caller must send to the endpoint. duration and day are optional
# and ignored. y is the answer, so the caller must not need to send it.
REQUIRED_RAW_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "month",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]

# 42 is an arbitrary fixed seed. Fixed means "run this twice, get the
# same split". It does not mean the number 42 is magical.
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
DECISION_THRESHOLD = 0.5

EXPERIMENT_NAME = "bank-term-deposit"
REGISTERED_MODEL_NAME = "bank-term-deposit-classifier"
ENDPOINT_NAME = "bank-td-mm25"
DEPLOYMENT_NAME = "blue"

# Smallest general-purpose CPU size that Azure ML online endpoints accept.
# One machine, always on. Cost notes are in README.md.
INSTANCE_TYPE = "Standard_F2s_v2"
INSTANCE_COUNT = 1

# These are names, not passwords. Override them in .env if you deploy
# into a different resource group.
# AZUREML_ARM_* is set automatically inside an Azure ML job. The laptop
# does not have those variables, so it falls back to the names below
# and asks `az account show` for the subscription id.
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID") or os.getenv(
    "AZUREML_ARM_SUBSCRIPTION", ""
)
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP") or os.getenv(
    "AZUREML_ARM_RESOURCEGROUP", "rg-bank-deposit-ml"
)
AZURE_ML_WORKSPACE = os.getenv("AZURE_ML_WORKSPACE") or os.getenv(
    "AZUREML_ARM_WORKSPACE_NAME", "mlw-bank-deposit"
)
AZURE_ML_LOCATION = os.getenv("AZURE_ML_LOCATION", "eastus")
