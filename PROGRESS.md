# Progress

## Done

- Chose the dataset and the modeling rules. Reasons are in [DECISIONS.md](DECISIONS.md).
- Created Azure resource group `rg-bank-deposit-ml` and workspace `mlw-bank-deposit` in East US.
- Scaffolded the project: settings, ingestion, features, training, scoring, tests, and the GitHub Actions workflow.

## Next

- Run the unit tests and fix whatever they catch.
- Train on the real CSV and write the results table.
- Log that run to Azure ML, register the winner, and deploy the managed online endpoint.
- Fill the README, the learning guide, and the blog post with the real numbers.
- Publish the GitHub repo and the blog page.

## Decisions and blockers

- Decisions so far are in [DECISIONS.md](DECISIONS.md).
- No blocker yet. The workspace exists. The endpoint is not deployed, so it is not billing compute yet.
- The workspace itself created a storage account, a key vault, and Application Insights. Those are the cheap part. The expensive part is the endpoint virtual machine, which does not exist yet.
