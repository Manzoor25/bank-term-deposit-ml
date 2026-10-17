# How to review this project

The point of this file is not to restate the code. It is a practice interview.

Do not read the answers first. Each section gives you a question. Answer it out loud, then open the file it names and check yourself. If you cannot answer without looking, you are not ready to claim that part in an interview. That is a useful result, not a failure.

Budget: 45 minutes. A timer helps.

## 0. The sentence you should be able to say cold

> I trained a yes/no model on the UCI bank marketing calls, refused the call-duration column because it is only known after the call, compared logistic regression, a random forest, and gradient boosting, picked the winner with cross-validated average precision, and deployed that pipeline to an Azure ML managed online endpoint.

If any clause in that sentence is fuzzy, that clause is your study target. The sections below match the clauses.

## 1. Ten minutes: walk one prediction

Open [sample_request.json](sample_request.json). One customer. Notice `duration` is 999. Then follow the call, and say what each stop does before you read the function.

1. [scoring/score.py](scoring/score.py) `run` receives the JSON. Why is the model loaded in `init` instead of here?
2. [predict.py](predict.py) `score_records` checks required columns. What happens if `job` is missing, and why is a crash better than a guess?
3. [features.py](features.py) `add_engineered_columns` drops `duration`. Confirm the sample's 999 cannot change the score. The test that locks this in is `test_duration_does_not_change_the_score`.
4. The saved object in `models/pipeline.joblib` is a sklearn Pipeline: impute, scale, one-hot, then the model. Why is that one object, not four files?

Check yourself: `python main.py predict` printed probability 0.4229 for that sample. Changing `duration` in the JSON and running it again should print the same number. Do that once. It takes ten seconds and it makes the leakage story physical.

## 2. Twenty minutes: interrogate DECISIONS.md

For each decision, ask "why not the other way?" Answer before you read the paragraph.

| Decision | The question to ask out loud | Where the code proves it |
| --- | --- | --- |
| 1. Dataset | Why not a Kaggle churn file? | [ingest.py](ingest.py) downloads a public URL. No token. |
| 2. The question | What would go wrong if we kept `duration`? | [config.py](config.py) `LEAKAGE_COLUMNS` |
| 3. Dropping `day` | Is `day` leakage, or just a bad feature? | It is in `UNUSED_COLUMNS`, not `LEAKAGE_COLUMNS`. Be ready to say the difference. |
| 4. `pdays = -1` | Why is -1 not a small number of days? | [features.py](features.py) builds `never_contacted_before` |
| 5. `campaign` minus 1 | What does the published column include that we should not? | Same function, `contacts_already_made` |
| 6. One-hot, not label encoding | Why is management not "more than" technician? | `OneHotEncoder(handle_unknown="ignore")` |
| 7. Three classical models | Why not a neural net on 45,000 rows? | [train.py](train.py) `build_models` |
| 8. Winner rule | Why not pick the model with the best test score? | `choose_winner` uses only the CV scores |
| 9. MLflow 2 and SQLite | Why not the newest MLflow? | [requirements.txt](requirements.txt) pins 2.22.1 |
| 10. Train on the laptop | Why is the cloud not doing the training? | `python main.py train` never starts a compute cluster |
| 12. Custom `score.py` | Why not the no-code MLflow scorer? | The JSON contract is `inputs` in, probability out |
| 13. CI does not deploy | Why is that a feature, not a missing feature? | [.github/workflows/ci.yml](.github/workflows/ci.yml) |
| 15. Identity storage auth | Why did the first upload fail? | README troubleshooting section |

Decision 8 is the one interviewers push on. Practice this sentence:

> The test set is a final exam. If I use it to choose the model, I no longer have an exam. I chose gradient boosting because its cross-validated average precision on the training split was 0.428, a hair above the forest at 0.426. Then I reported the test numbers once.

## 3. The accuracy trap, in numbers you should remember

- About 11.7% of people said yes.
- Always saying no is about 88% accurate and finds zero customers.
- The selected model is only 81% accurate. That is worse than always saying no.
- It finds about 62% of the subscribers (recall 0.616) and is right about a third of the time it says yes (precision 0.335).

If an interviewer says "81% accuracy is not great," the right response is: "Accuracy is the wrong score on this dataset. I optimized average precision and I would set the cutoff with the bank, not at 0.5."

The cutoff is a business decision. The project does not pretend 0.5 is sacred. Say that.

## 4. Eight interview questions and a solid answer

Use your own words. These are the beats, not a script to memorize.

**Why this dataset?**
Public, no login, and it contains a famous leakage column. The lesson is the column, not the bank.

**What did you exclude, and why?**
`duration`, because it is known only after the call. `day`, because it is a leftover calendar day, not because it is cheating. I can say which list each column is in.

**How did you avoid train/serve skew?**
Training and the endpoint both call `add_engineered_columns`, then the same saved Pipeline. A test scores the same row with two different durations and requires the same probability.

**How did you handle imbalance?**
`class_weight="balanced"`, and I did not select on accuracy. I selected on average precision.

**Why these three models?**
A linear baseline I can explain, a bagged tree model, and a boosted tree model. No neural net, because the table is small and the story would be worse.

**Why was the winner so close?**
Gradient boosting beat the forest by 0.002 on the practice score. I should not overclaim. A close race is a result. The forest even has a slightly higher test ROC-AUC (0.793 vs 0.789). I still shipped boosting because the selection rule was written down before I looked at the exam.

**What does the endpoint cost?**
About $0.085 an hour for one `Standard_F2s_v2`, and it does not scale to zero. Training was local. I can delete the endpoint with `python main.py teardown`.

**What would you do next if this were real?**
Calibrate the probabilities, pick the cutoff from a cost of a wasted call versus a missed subscriber, and split by time if we had years. We do not have a clean year column, so a random split was the honest available option. I would also turn on Entra ID auth instead of a key.

## 5. What not to claim

- Do not say the model is production-ready for a bank. It is a teaching deployment of a public research dataset.
- Do not say 81% accuracy means it is good, or that it is bad, without the always-no comparison.
- Do not say you used the call duration. You refused it. If you cannot show the test that proves that, do not claim it.
- Do not quote the Azure subscription id. It is not in the repo on purpose.
- Do not describe the endpoint as free. It bills while it exists.

## 6. A short rerun, if you want the story in your hands

```powershell
.\venv\Scripts\Activate.ps1
pytest
python main.py predict
```

Then change `duration` in [sample_request.json](sample_request.json) and predict again. Then put it back.

Read [reports/results.md](reports/results.md) and point at the row that says `(selected)`. Say which column chose it, and which columns did not.

That is the review. The code is the evidence. This file is the questioning.
