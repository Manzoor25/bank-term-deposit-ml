# Progress

## Done

- Chose the dataset and the modeling rules. Reasons are in [DECISIONS.md](DECISIONS.md).
- Created Azure resource group `rg-bank-deposit-ml` and workspace `mlw-bank-deposit` in East US.
- Ingestion, feature engineering, three-model training, scoring, and pytest are in place.
- `ruff check` passed and `pytest` passed: 15 tests, no Azure login required.
- Trained on the real 45,211-row file. Gradient boosting won on cross-validated average precision (0.428). The table is in [reports/results.md](reports/results.md).

## Next

- Wait for the managed virtual network private endpoints to become active.
- Push this repo, then run [azure/train_job.yml](azure/train_job.yml) so training, registration, and deployment happen on compute that can reach storage.
- Invoke the endpoint and write its URL into the README.
- Publish the blog with GitHub Pages.

## Decisions and blockers

- Decisions are in [DECISIONS.md](DECISIONS.md).
- Blocker, with a path around it: this subscription forces storage public access off. A laptop cannot upload the model. Decision 15 records that, and the cloud job is the workaround.
- The endpoint is not deployed yet, so the virtual machine is not billing yet. The workspace storage, key vault, and Application Insights already exist.
