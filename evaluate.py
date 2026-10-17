# =============================================================
# WHAT THIS FILE DOES:
# Scores a model's predictions and formats the comparison table.
# It does not train anything. Training is train.py's job.
#
# HOW IT CONNECTS TO OTHER FILES:
# - train.py calls compute_metrics() once per model
# - train.py calls results_markdown() to write reports/results.md
#
# WHY METRICS LIVE ALONE:
# It is tempting to print one accuracy number and stop. This file
# exists so we always compute the same set of scores, including the
# ones that tell the truth on an imbalanced yes/no problem.
# =============================================================

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# --------------------------------------------------------------
# FUNCTION: compute_metrics
# WHAT IT DOES: Turns probabilities into a small dictionary of scores
# WHY WE NEED IT: Accuracy alone is misleading here. About 88% of
#                 clients said "no". A model that always says "no"
#                 would be 88% accurate and completely useless.
# INPUTS:
#   y_true: 1 for subscribed, 0 otherwise
#   y_probability: the model's estimated probability of "yes"
#   threshold: probability at or above this becomes a "yes" prediction
# OUTPUTS: A dictionary of floats between 0 and 1 (higher is better)
#
# LEARNING NOTE: ROC-AUC and average precision use the ranking of the
# probabilities, not the threshold. F1, precision, and recall use the
# threshold. We select the winning model with average precision, which
# cares about finding the rare "yes" rows.
# --------------------------------------------------------------
def compute_metrics(y_true, y_probability, threshold):
    predicted = (y_probability >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "average_precision": float(average_precision_score(y_true, y_probability)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predicted)),
    }


# --------------------------------------------------------------
# FUNCTION: choose_winner
# WHAT IT DOES: Picks the model with the highest cross-validated score
# WHY WE NEED IT: The choice must be made BEFORE we look at the test
#                 set. If we pick the winner using the test set, the
#                 test set is no longer an honest final exam.
# INPUTS: cv_scores, a dictionary like {"random_forest": 0.41, ...}
# OUTPUTS: The winning model name (a string)
#
# LEARNING NOTE: max(scores, key=scores.get) means "give me the name
# whose value is largest". A manual if/else ladder would also work.
# --------------------------------------------------------------
def choose_winner(cv_scores):
    if not cv_scores:
        raise ValueError("choose_winner received no scores.")
    return max(cv_scores, key=cv_scores.get)


# --------------------------------------------------------------
# FUNCTION: results_markdown
# WHAT IT DOES: Builds a Markdown table a human can read in the README
# WHY WE NEED IT: Hand-typed numbers drift away from the real run.
#                 The training script writes this text itself.
# INPUTS:
#   rows: list of dictionaries, one per model
#   winner_name: which model was selected
#   selection_metric: the name of the score used to pick the winner
# OUTPUTS: A Markdown string
# --------------------------------------------------------------
def results_markdown(rows, winner_name, selection_metric):
    header = (
        "| Model | CV average precision | Test average precision | "
        "Test ROC-AUC | Test F1 | Test precision | Test recall | Test accuracy |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    lines = [header]
    for row in rows:
        marker = " (selected)" if row["model_name"] == winner_name else ""
        lines.append(
            "| {name}{marker} | {cv:.3f} | {ap:.3f} | {roc:.3f} | {f1:.3f} | "
            "{precision:.3f} | {recall:.3f} | {accuracy:.3f} |\n".format(
                name=row["model_name"],
                marker=marker,
                cv=row["cv_average_precision"],
                ap=row["test_average_precision"],
                roc=row["test_roc_auc"],
                f1=row["test_f1"],
                precision=row["test_precision"],
                recall=row["test_recall"],
                accuracy=row["test_accuracy"],
            )
        )
    note = (
        "\nThe selected model was chosen by **cross-validated "
        f"{selection_metric}** on the training split only. "
        "Test columns are the final exam. They were not used to pick the winner.\n"
    )
    return "".join(lines) + note


if __name__ == "__main__":
    print("evaluate.py holds scoring helpers. Run python main.py train to use them.")
