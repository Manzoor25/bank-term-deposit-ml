# Progress

## Done

- Chose the dataset and the modeling rules. Reasons are in [DECISIONS.md](DECISIONS.md).
- Ingestion, feature engineering, three-model training, scoring, and pytest are in place.
- `ruff check` passed and `pytest` passed: 15 tests.
- Trained on the real 45,211-row file. Gradient boosting won on cross-validated average precision (0.428). The table is in [reports/results.md](reports/results.md).
- Registered `bank-term-deposit-classifier` version 2 in workspace `mlw-bank-deposit`.
- Deployed managed online endpoint `bank-td-mm25` on `Standard_F2s_v2`. A live invoke returned probability 0.4229, the same number as the laptop.
- Published the repo and the blog page.

## Next

- Read [LEARNING_GUIDE.md](LEARNING_GUIDE.md) before treating this as interview material.
- Delete the endpoint when you are done looking. It bills about $0.085 an hour and does not scale to zero: `python main.py teardown`.

## Decisions and blockers

- Decisions are in [DECISIONS.md](DECISIONS.md).
- This subscription disables public storage access. The laptop could not upload the model. A compute cluster with its own identity, inside the managed virtual network, did the upload and the deploy. That is decision 15.
- The endpoint is the thing that is billing now. The training cluster is set to scale to zero.
