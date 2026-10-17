# Decisions

When two reasonable choices existed, this file records the one I made and the one I rejected.
Read this with [LEARNING_GUIDE.md](LEARNING_GUIDE.md). The guide tells you which question to ask at each decision.

## 1. Which dataset?

**Chose:** UCI Bank Marketing, file `bank-full.csv` (45,211 phone-call rows, 17 columns).
**Rejected:** A Kaggle churn download, and the older `bank-additional-full.csv`.

Why:
- The UCI file has a stable public URL. No Kaggle account.
- `bank-additional-full.csv` adds economy-wide numbers (interest rate, employment). Those numbers are the same for every customer in a month, and they can dominate the model. A bank deciding whether to call *this* person does not get to change the national interest rate. The newer file is the cleaner teaching problem.
- The famous trap in this file is the `duration` column. That trap is the point.

License: CC BY 4.0. Citation is in [README.md](README.md) and in [config.py](config.py).

## 2. What is the question the model answers?

**Chose:** "If we place this call, how likely is a term-deposit subscription?"
**Rejected:** "Given that we already know how long the call lasted, did they subscribe?"

The second question is easier and useless. The call length is only known after the call. Using it is leakage: the model peeks at information it will not have when you actually need the prediction.

## 3. Which columns are refused?

**Chose:** Drop `duration` (leakage) and `day` (brittle calendar quirk).
**Rejected:** Keep every column, because "more features are always better."

`day` is the day of the month of a historical call. It is not cheating, but a model can memorize "the 5th was a good day in this old campaign." Next month's calendar will not repeat that. `month` stays, because seasonality (May is busy, December is quiet) is a real planning signal.

## 4. What did we do with `pdays = -1`?

**Chose:** Treat `-1` as a code, not a number. Add `never_contacted_before` (1 or 0) and set `pdays` to 0 for those rows.
**Rejected:** Leave `-1` in the numeric column.

`-1` means "never contacted before," which the file's own notes say. A linear model that sees `-1` thinks those people were contacted in the future. The extra yes/no column tells the truth. Filling with 0 is safe *because* the flag separates "never contacted" from "contacted today."

## 5. What did we do with `campaign`?

**Chose:** Subtract 1, and call the result `contacts_already_made`.
**Rejected:** Feed the raw `campaign` number straight in.

The published definition says `campaign` includes the last contact. That last contact is the call whose outcome we are predicting. Before the call, the honest count is one less. A value of 0 means "this would be the first attempt in this campaign."

## 6. How are categories turned into numbers?

**Chose:** A scikit-learn `Pipeline` with one-hot encoding and `handle_unknown="ignore"`.
**Rejected:** Hand-written `if job == "technician"` columns, or label encoding (technician=3, management=7).

Label encoding pretends that management is "more than" technician because 7 > 3. That is nonsense. One-hot encoding makes a separate yes/no column per job. `handle_unknown="ignore"` means a job title the model has never seen becomes all zeros instead of crashing the endpoint.

The engineering steps (`duration` drop, `pdays` flag) live in [features.py](features.py) instead of inside the saved Pipeline. A custom class inside a Pipeline has to be importable on the serving computer under the exact same name. Plain sklearn steps do not have that problem. Training and scoring both call the same function, and a test checks they stay in sync.

## 7. Which three models, and why not a neural net?

**Chose:**
1. Logistic regression (one straight boundary, easy to explain)
2. Random forest (many trees vote)
3. Histogram gradient boosting (trees built in sequence, each fixing the last tree's mistakes)

**Rejected:** A deep neural network.

This is a table of about 45,000 rows and a handful of columns. Classical models are the right tool, they train in minutes on a laptop, and you can explain them in an interview without a GPU story.

`class_weight="balanced"` is on because about 88% of people said no. Without it, a model can look accurate by always saying no.

## 8. How is the winner chosen?

**Chose:** Highest cross-validated average precision on the training split. The test split is graded once and does not vote.
**Rejected:** Pick whichever model has the highest test score. Also rejected: pick by accuracy.

Accuracy is a trap here. Always saying "no" is about 88% accurate and finds zero subscribers. Average precision cares about ranking the rare "yes" rows near the top.

Picking the winner on the test set would turn the final exam into practice. Then you would not have an honest number left to report.

The 0.5 probability cutoff is a teaching default, not a business rule. The endpoint returns the probability so a bank can choose "only call people above 0.7" if false alarms are expensive.

## 9. Where do experiment logs live?

**Chose:** A local SQLite file (`mlflow.db`) by default. `--azure` switches the same training function to the workspace MLflow address.
**Rejected:** The old `./mlruns` folder, and training on an Azure compute cluster.

We pin MLflow 2.22, not MLflow 3. MLflow 3's saver refuses to store a random forest unless you whitelist internal scikit-learn types, one error at a time. Version 2 is also what the Azure ML tracking examples still match. MLflow 2.22 requires pandas older than 3, so pandas is pinned to 2.2.3 even though 3.0 exists. Local logs go to one SQLite file (`mlflow.db`) instead of a folder of text files, so there is a single notebook to delete. The Azure address is only used when you pass `--azure`, so tests and GitHub Actions never need a cloud login.

Training itself stays on the laptop. The table is small. A compute cluster would cost money and add nothing to the lesson. The cloud's job is the experiment log, the model registry, and the live endpoint.

## 10. Where does training run?

**Chose:** Train on this laptop. Log the MLflow experiment to Azure Machine Learning when you pass `--azure`.
**Rejected:** An Azure ML compute cluster for training.

The table is small. A cluster would cost money and add nothing to the lesson. The cloud's job in this project is the experiment log, the model registry, and the live endpoint. That split is a normal production pattern: train where it is cheap, register and serve where other people can call it.

## 11. Which Azure resources?

**Chose:** A new resource group `rg-bank-deposit-ml` and workspace `mlw-bank-deposit` in East US.
**Rejected:** Reusing the existing workspace `mlw-e2e-fdpo`.

That other workspace belongs to a different lab. Mixing experiments would make both projects harder to explain and harder to delete. East US already had Machine Learning quota on this subscription.

GitHub repo account: `Manzoor25`, because that is the GitHub account already logged in on this machine. The Azure login is a different identity (`mamoh@microsoft.com`). The repo does not contain the Azure subscription id.

## 12. How is the endpoint shaped?

**Chose:** A managed online endpoint, one instance, smallest CPU size that works (`Standard_F2s_v2`, falling back to `Standard_DS2_v2` or `Standard_DS3_v2`), custom `score.py`, key authentication.
**Rejected:** Azure Container Apps or a laptop Flask server as the "production" URL. Also rejected: no-code MLflow scoring.

A managed online endpoint is what the task asked for, and it is the Azure ML product meant for real-time model calls. The custom scoring script gives a JSON contract a person can read (`inputs` in, probability out) and lets us reject a missing column loudly. No-code MLflow scoring works, but the request body is a dataframe layout that is harder to explain.

Key authentication is the simple switch for a demo. A locked-down production system would use Microsoft Entra ID instead of a key, and would not print the key. This project never writes the key into Git. Calls go through `az` / the Azure SDK, which keep the key in Azure.

The endpoint does **not** scale to zero. Leaving it up bills every hour. `python main.py teardown` deletes it. See the cost notes in the README.

## 13. What do tests and GitHub Actions cover?

**Chose:** pytest on a 24-row fake table, plus ruff, on every push. No Azure login in CI.
**Rejected:** Downloading the real CSV inside the test run, and deploying from GitHub Actions.

Tests should pass on a plane. A workflow that deploys on every push would create a bill and would need a secret. The real training run is a separate command you run on purpose.

## 15. How does Azure storage authentication work?

**Chose:** Workspace datastore auth mode `identity`, plus a managed virtual network with a private endpoint to storage.
**Rejected:** Turning shared keys or public network access back on. This subscription will not allow it. A brand-new storage account came back with public access disabled and shared keys disabled, and a patch to enable public access was ignored.

Two errors showed up in order:
1. `KeyBasedAuthenticationNotPermitted` — the workspace was trying to use an account key.
2. `AuthorizationFailure` — after switching to identity, the laptop still could not reach the blob endpoint, because public network access is off and there is no private path from this machine.

The managed network gives Azure ML compute a private door. [azure/train_job.yml](azure/train_job.yml) runs training and deployment from that compute. It downloads the code from GitHub instead of uploading a local folder, because the upload is the step the laptop cannot do.

## 16. Why is the settings file not named config.py?

**Chose:** [settings.py](settings.py).
**Rejected:** Keeping the name `config.py`.

The Azure ML scoring server already has a module named `config`. Our file shadowed it. The container downloaded the model, started, and died on import with `attempted relative import with no known parent package`. Renaming the file fixed the crash. The inference server package `azureml-inference-server-http` also had to be in the container, or it never got as far as the import.

## 14. Is the model binary committed?

**Chose:** No. Git ignores `models/*.joblib`. The registry copy is the one the endpoint serves. `reports/results.md` is committed so the numbers are still visible.
**Rejected:** Committing the model file so the repo "just works" offline.

Model files get large, and a committed model drifts away from the code that claims to have trained it. Regenerating it is `python main.py train`.
