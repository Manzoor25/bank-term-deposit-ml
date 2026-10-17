# =============================================================
# WHAT THIS FILE DOES:
# Turns the raw spreadsheet into the columns the model is
# allowed to see, and builds the sklearn "assembly line"
# that encodes those columns the same way every time.
#
# HOW IT CONNECTS TO OTHER FILES:
# - It reads column rules from config.py
# - train.py calls add_engineered_columns() before fitting
# - predict.py and scoring/score.py call the same function
#   so a live request is prepared exactly like a training row
#
# THE ASSEMBLY-LINE IDEA:
# A sklearn Pipeline is a fixed sequence of steps. Fit it on
# training rows, save it, and the saved object remembers the
# encodings. Serving code must not rebuild those encodings by hand.
# =============================================================

# pandas is the table library. We nickname it pd, which is the
# spelling you will see in almost every Python data tutorial.
import pandas as pd

# ColumnTransformer, Pipeline, and the encoders all come from
# scikit-learn, the standard classical-ML library. We are not
# using a neural network. This dataset is a table of a few dozen
# thousand rows, which is what these models are built for.
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import LEAKAGE_COLUMNS, TARGET_COLUMN, UNUSED_COLUMNS

# Numbers the model can do arithmetic on, AFTER engineering.
# contacts_already_made and never_contacted_before are created below.
# They are not columns in the original CSV.
NUMERIC_FEATURES = [
    "age",
    "balance",
    "contacts_already_made",
    "pdays",
    "never_contacted_before",
    "previous",
]

# Categories: labels like "married" or "may", not amounts.
# We will turn each label into its own yes/no column (one-hot encoding).
CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]


# --------------------------------------------------------------
# FUNCTION: add_engineered_columns
# WHAT IT DOES: Copies a table and rewrites it into model-ready columns
# WHY WE NEED IT: The raw file contains a cheating column (duration),
#                 a sentinel value (-1), and a count that includes
#                 the call we have not made yet.
# INPUTS: raw_frame, a pandas DataFrame. It may include extra columns.
# OUTPUTS: A new DataFrame. The original is not modified.
#
# LEARNING NOTE: We could hide this rewrite inside the Pipeline with
# a custom class. We do not, because custom classes are awkward to
# save and load (the serving computer must import the exact same class).
# Plain sklearn steps save cleanly. The small rewrite stays in this
# function, which both training and scoring import.
# --------------------------------------------------------------
def add_engineered_columns(raw_frame):
    frame = raw_frame.copy()  # copy so we never surprise the caller by editing their table

    columns_to_drop = [
        column
        for column in [TARGET_COLUMN, *LEAKAGE_COLUMNS, *UNUSED_COLUMNS]
        if column in frame.columns
    ]
    if columns_to_drop:
        frame = frame.drop(columns=columns_to_drop)

    # to_numeric turns "42" into 42. errors="coerce" turns garbage into
    # missing (NaN) instead of crashing. The imputer fills missing later.
    for column in ["age", "balance", "campaign", "pdays", "previous"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "campaign" in frame.columns:
        # The published "campaign" count includes the call that produced
        # the outcome. Before that call, the honest count is one less.
        # .clip(lower=0) stops a 0 from becoming -1.
        frame["contacts_already_made"] = (frame["campaign"] - 1).clip(lower=0)
        frame = frame.drop(columns=["campaign"])

    if "pdays" in frame.columns:
        # -1 in this file is a code for "never contacted", not a real
        # number of days. A model that treats -1 as a number will think
        # those people were contacted in the future.
        never_contacted = frame["pdays"].isna() | (frame["pdays"] < 0)
        frame["never_contacted_before"] = never_contacted.astype(int)
        frame.loc[never_contacted, "pdays"] = 0

    for column in CATEGORICAL_FEATURES:
        if column in frame.columns:
            # "May" and "may" would otherwise become two different categories.
            # map (instead of a plain .str call) keeps real missing values as
            # missing, so the imputer can fill them later.
            frame[column] = frame[column].map(_clean_label)

    for column in NUMERIC_FEATURES:
        if column in frame.columns:
            # Integers cannot store "missing". Floats can. MLflow's schema
            # check warns if we claim a column is an integer and later send
            # a blank. Float avoids that trap.
            frame[column] = frame[column].astype(float)

    return frame


# --------------------------------------------------------------
# FUNCTION: _clean_label
# WHAT IT DOES: Lowercases one category label and trims spaces
# WHY WE NEED IT: The loop above stays readable, and missing values
#                 stay missing instead of becoming the word "nan"
# INPUTS: value, one cell
# OUTPUTS: A cleaned string, or the original missing value
# --------------------------------------------------------------
def _clean_label(value):
    if pd.isna(value):
        return value
    return str(value).strip().lower()


# --------------------------------------------------------------
# FUNCTION: build_preprocessor
# WHAT IT DOES: Builds the encode-and-scale steps, with no model yet
# WHY WE NEED IT: Logistic regression needs scaled numbers. All three
#                 models need categories turned into numbers. Doing that
#                 in one shared object means the models are compared
#                 fairly, on the same columns.
# INPUTS: None. The column lists at the top of this file are the inputs.
# OUTPUTS: A ColumnTransformer (a sklearn step that applies different
#          tools to different columns)
#
# LEARNING NOTE: Trees do not need StandardScaler. We scale anyway so
# one assembly line can feed all three model types. The alternative is
# two assembly lines. Simpler wins here.
# --------------------------------------------------------------
def build_preprocessor():
    numeric_steps = Pipeline(
        steps=[
            # median is the middle value. It is less yanked around by one
            # huge bank balance than the average (mean) would be.
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_steps = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # handle_unknown="ignore" means a new job title at serving time
            # becomes all zeros instead of crashing the endpoint.
            # sparse_output=False returns a normal table. Easier to debug,
            # and some models reject the compressed "sparse" form.
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numbers", numeric_steps, NUMERIC_FEATURES),
            ("labels", categorical_steps, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # extra columns are ignored, not silently used
    )


# --------------------------------------------------------------
# FUNCTION: build_pipeline
# WHAT IT DOES: Glues the preprocessor and one model into one object
# WHY WE NEED IT: Saving one object means serving cannot forget a step
# INPUTS: estimator, an unfitted sklearn model
# OUTPUTS: An unfitted Pipeline. train.py fits it.
#
# LEARNING NOTE: The alternative is to transform the table, save the
# encoder, save the model, and remember to apply them in order yourself.
# People forget the order. The Pipeline does not.
# --------------------------------------------------------------
def build_pipeline(estimator):
    return Pipeline(
        steps=[
            ("prepare_columns", build_preprocessor()),
            ("model", estimator),
        ]
    )
