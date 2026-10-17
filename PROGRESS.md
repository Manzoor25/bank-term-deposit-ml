# Progress

## Done

- Chose the dataset and the modeling rules. Reasons are in [DECISIONS.md](DECISIONS.md).
- Created Azure resource group `rg-bank-deposit-ml` and workspace `mlw-bank-deposit` in East US.
- Ingestion, feature engineering, three-model training, scoring, and pytest are in place.
- `ruff check` passed and `pytest` passed: 15 tests, no Azure login required.
- Trained on the real 45,211-row file. Gradient boosting won on cross-validated average precision (0.428). The table is in [reports/results.md](reports/results.md).

## Next

- Log the same experiment to Azure ML (`python main.py train --azure`).
- Register the winner and deploy the managed online endpoint.
- Fill the README, the learning guide, and the blog post.
- Publish the GitHub repo and the blog page.

## Decisions and blockers

- Decisions are in [DECISIONS.md](DECISIONS.md).
- No blocker yet. The endpoint is not deployed, so the virtual machine is not billing yet.
- The workspace already created a storage account, a key vault, and Application Insights. Those are the cheap part.
