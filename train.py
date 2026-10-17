# =============================================================
# WHAT THIS FILE DOES:
# Trains three different model types on the same rows, logs each
# attempt to MLflow, and saves the winner to models/.
#
# HOW IT CONNECTS TO OTHER FILES:
# - ingest.py supplies the table (main.py usually does that call)
# - features.py builds the assembly line around each model
# - evaluate.py computes the scores and the Markdown table
# - azure_ml.py can point MLflow at Azure before this file runs
# - predict.py loads the file this script writes
#
# THE FLOW:
# 1. Split the rows into a practice set (train) and a final exam (test).
# 2. On the practice set only, cross-validate three models.
# 3. Pick the winner from those practice scores.
# 4. Fit each model on the full practice set and grade the exam once.
# 5. Save the winner. Write reports/results.md so the numbers are not retyped.
# =============================================================

import json
import os
from pathlib import Path

# joblib saves Python objects to disk. sklearn recommends it over pickle
# for models because it handles large numeric arrays more efficiently.
import joblib

# MLflow is a lab notebook for experiments. Each run stores the settings
# and the scores so you can compare them later, including in Azure ML.
import mlflow
import mlflow.sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from config import (
    CV_FOLDS,
    DECISION_THRESHOLD,
    EXPERIMENT_NAME,
    MODEL_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from evaluate import choose_winner, compute_metrics, results_markdown
from features import add_engineered_columns, build_pipeline


# --------------------------------------------------------------
# FUNCTION: encode_target
# WHAT IT DOES: Turns the "yes"/"no" text into 1 and 0
# WHY WE NEED IT: sklearn models want numbers, not words, as the answer
# INPUTS: frame, the raw table, which must still contain column y
# OUTPUTS: A column of 1s and 0s
#
# LEARNING NOTE: We do this BEFORE add_engineered_columns, because that
# function drops y on purpose so the answer cannot sneak into the inputs.
# --------------------------------------------------------------
# --------------------------------------------------------------
# FUNCTION: sqlite_tracking_uri
# WHAT IT DOES: Builds a local MLflow address that points at a SQLite file
# WHY WE NEED IT: A SQLite file is one notebook on your disk.
#                 The older alternative is a folder named mlruns with
#                 many small files. One file is easier to find and delete.
#                 Azure ML is a different address, used only with --azure.
# INPUTS: database_path, where the .db file should live
# OUTPUTS: A string like sqlite:///C:/project/mlflow.db
#
# LEARNING NOTE: You could also pass "file:./mlruns". We do not.
# One database file is easier to explain than a tree of folders.
# --------------------------------------------------------------
def sqlite_tracking_uri(database_path):
    database_path = Path(database_path).resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database_path.as_posix()}"


# --------------------------------------------------------------
# FUNCTION: encode_target
# WHAT IT DOES: Turns the "yes"/"no" text into 1 and 0
# WHY WE NEED IT: sklearn models want numbers, not words, as the answer
# INPUTS: frame, the raw table, which must still contain column y
# OUTPUTS: A column of 1s and 0s
#
# LEARNING NOTE: We do this BEFORE add_engineered_columns, because that
# function drops y on purpose so the answer cannot sneak into the inputs.
# --------------------------------------------------------------
def encode_target(frame):
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(
            f"The table has no '{TARGET_COLUMN}' column, so there is nothing to learn."
        )
    labels = frame[TARGET_COLUMN].astype(str).str.strip().str.lower()
    unexpected = sorted(set(labels.unique()) - {"yes", "no"})
    if unexpected:
        raise ValueError(f"Target column has unexpected labels: {unexpected}")
    return (labels == "yes").astype(int)


# --------------------------------------------------------------
# FUNCTION: build_models
# WHAT IT DOES: Creates the three unfitted models we compare
# WHY WE NEED IT: One place lists the model types. Adding a fourth
#                 model means adding one entry here, not rewriting the loop.
# INPUTS: small (True uses tiny settings so tests finish in seconds)
# OUTPUTS: A dictionary of name -> unfitted model
#
# LEARNING NOTE: These three families answer the same yes/no question
# in different ways.
# - Logistic regression draws one straight boundary. Easy to explain.
# - A random forest lets many decision trees vote.
# - Gradient boosting builds trees one after another, each fixing
#   the mistakes of the previous tree.
# A neural network is a fourth option. It is unnecessary on this
# small, already-tabular problem, and harder to explain in an interview.
# --------------------------------------------------------------
def build_models(small=False):
    if small:
        return {
            "logistic_regression": LogisticRegression(
                max_iter=200,
                class_weight="balanced",
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=10,
                max_depth=3,
                random_state=RANDOM_STATE,
                class_weight="balanced_subsample",
            ),
            "gradient_boosting": HistGradientBoostingClassifier(
                max_iter=10,
                max_depth=3,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
        }

    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=1,  # 1 keeps the laptop responsive. -1 uses every core.
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_depth=6,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }


# --------------------------------------------------------------
# FUNCTION: train_and_select
# WHAT IT DOES: Runs the full comparison and saves the winner
# WHY WE NEED IT: This is the experiment. Everything else is plumbing.
# INPUTS:
#   frame: the raw table from ingest.py
#   tracking_uri: where MLflow should write. None uses MLflow's default.
#   experiment_name: the notebook tab these runs appear under
#   small: tiny models for tests
#   cv_folds: how many practice-exam slices to use
#   output_dir: where to write pipeline.joblib
#   reports_dir: where to write results.md
# OUTPUTS: A dictionary with the winner name, scores, and file paths
# --------------------------------------------------------------
def train_and_select(
    frame,
    tracking_uri=None,
    experiment_name=EXPERIMENT_NAME,
    register_name=None,
    small=False,
    cv_folds=CV_FOLDS,
    output_dir=MODEL_DIR,
    reports_dir=REPORTS_DIR,
):
    if not tracking_uri:
        # Inside an Azure ML job the platform sets this variable for us.
        # We must not replace it with a local SQLite file, or the run
        # never shows up in the workspace.
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        tracking_uri = sqlite_tracking_uri(PROJECT_ROOT / "mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    if str(tracking_uri).startswith("azureml:"):
        print("Logging this experiment to Azure Machine Learning.")
    else:
        print("Logging this experiment to a local SQLite file.")

    mlflow.set_experiment(experiment_name)

    answers = encode_target(frame)
    prepared = add_engineered_columns(frame)

    # stratify=answers keeps the same yes/no mix in both splits.
    # Without it, the exam could accidentally contain almost no "yes" rows.
    features_train, features_test, answers_train, answers_test = train_test_split(
        prepared,
        answers,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=answers,
    )

    folder = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    models = build_models(small=small)

    cv_scores = {}
    fitted_models = {}
    test_metrics = {}
    run_ids = {}
    rows_for_table = []

    for model_name, estimator in models.items():
        pipeline = build_pipeline(estimator)

        # Cross-validation retrains several times on the PRACTICE set only.
        # average_precision is the score we trust on this imbalanced problem.
        practice_scores = cross_val_score(
            pipeline,
            features_train,
            answers_train,
            cv=folder,
            scoring="average_precision",
        )
        cv_mean = float(practice_scores.mean())
        cv_scores[model_name] = cv_mean

        pipeline.fit(features_train, answers_train)
        fitted_models[model_name] = pipeline

        # The exam is graded here, but it does not choose the winner.
        # We still log it so the results table can show the final exam.
        probability = pipeline.predict_proba(features_test)[:, 1]
        metrics = compute_metrics(answers_test, probability, DECISION_THRESHOLD)
        test_metrics[model_name] = metrics

        with mlflow.start_run(run_name=model_name) as run:
            run_ids[model_name] = run.info.run_id
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("small", small)
            mlflow.log_param("cv_folds", cv_folds)
            mlflow.log_param("train_rows", int(len(features_train)))
            mlflow.log_param("test_rows", int(len(features_test)))
            mlflow.log_metric("cv_average_precision", cv_mean)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)
            # input_example lets MLflow record what a row looks like.
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                input_example=features_train.head(2),
            )

        rows_for_table.append(
            {
                "model_name": model_name,
                "cv_average_precision": cv_mean,
                "test_average_precision": metrics["average_precision"],
                "test_roc_auc": metrics["roc_auc"],
                "test_f1": metrics["f1"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_accuracy": metrics["accuracy"],
            }
        )

    winner_name = choose_winner(cv_scores)
    winner = fitted_models[winner_name]

    if register_name:
        # This uses the tracking server, not the Azure CLI. An Azure ML
        # job can do it. A laptop without `az` cannot.
        mlflow.register_model(
            model_uri=f"runs:/{run_ids[winner_name]}/model",
            name=register_name,
        )
        print(f"Registered {register_name} from run {run_ids[winner_name]}")

    # Re-open the winning run and tag it, so the Azure ML screen shows
    # which attempt we actually shipped.
    with mlflow.start_run(run_id=run_ids[winner_name]):
        mlflow.set_tag("selected_best", "true")
        mlflow.set_tag("selection_metric", "cv_average_precision")

    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "pipeline.joblib"
    joblib.dump(winner, model_path)

    # Keep a copy of the MLflow model next to the joblib file.
    # A cloud job can copy this folder to its output mount. The laptop
    # cannot upload it itself on this subscription.
    if not str(mlflow.get_tracking_uri()).startswith("azureml:"):
        from mlflow.artifacts import download_artifacts

        download_artifacts(
            run_id=run_ids[winner_name],
            artifact_path="model",
            dst_path=str(output_dir / "mlflow-model"),
        )

    model_card = {
        "winner": winner_name,
        "selection_metric": "cv_average_precision",
        "selection_score": cv_scores[winner_name],
        "threshold": DECISION_THRESHOLD,
        "mlflow_run_id": run_ids[winner_name],
        "experiment_name": experiment_name,
        "train_rows": int(len(features_train)),
        "test_rows": int(len(features_test)),
        "positive_rate_train": float(answers_train.mean()),
        "excluded_because_leakage": ["duration"],
        "excluded_as_brittle": ["day"],
        "note": (
            "Threshold 0.5 is a teaching default, not a business decision. "
            "The endpoint returns a probability so a caller can choose their own cutoff."
        ),
    }
    card_path = output_dir / "model_card.json"
    card_path.write_text(json.dumps(model_card, indent=2), encoding="utf-8")

    table = results_markdown(rows_for_table, winner_name, "average precision")
    results_path = reports_dir / "results.md"
    results_path.write_text(
        "# Model comparison\n\n" + table + "\nNumbers were written by train.py.\n",
        encoding="utf-8",
    )
    (reports_dir / "metrics.json").write_text(
        json.dumps(
            {
                "winner": winner_name,
                "cv_scores": cv_scores,
                "test_metrics": test_metrics,
                "run_ids": run_ids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "winner": winner_name,
        "cv_scores": cv_scores,
        "test_metrics": test_metrics,
        "run_ids": run_ids,
        "model_path": str(model_path),
        "model_card_path": str(card_path),
        "results_path": str(results_path),
        "results_markdown": table,
    }


if __name__ == "__main__":
    # Running this file directly uses whatever CSV ingest.py can find.
    # main.py is the supported entry point. This block is here so the
    # file is not a dead end if you open it and press Run.
    from ingest import ingest

    summary = train_and_select(ingest())
    print(f"Winner: {summary['winner']}")
    print(summary["results_markdown"])
