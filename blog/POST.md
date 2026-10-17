# The column that makes a model look smart

A bank called 45,211 people and asked them to open a term deposit. The public dataset records who said yes. It also records how long the call lasted.

That second number is a trap.

If you are deciding whether to dial someone, you do not know how long the conversation will be. You only learn that after you have already paid for the call. A model trained on call length learns "long calls were successful calls," which is true and useless. It will look excellent in a notebook and fail the moment you try to use it.

This project refuses that column on purpose.

## What we built

A small production-style pipeline, not a one-off notebook:

1. Download the UCI Bank Marketing file. The zip contains another zip. That surprise is in the code, not in a README you have to remember.
2. Rewrite the columns a model is allowed to see. `-1` in "days since last contact" is a code for "never contacted," not a number of days. The campaign count includes the call we have not made yet, so we subtract one.
3. Train three models on the same rows: logistic regression, a random forest, and gradient boosting.
4. Pick the winner using cross-validated average precision on the training split only. The test split is a final exam. It does not vote.
5. Log the experiment to Azure Machine Learning, register the winner, and serve it from a managed online endpoint.

## The score that matters

About 12% of people said yes. A model that always says no is about 88% accurate and finds nobody. Accuracy is the wrong trophy.

Gradient boosting won, barely. Cross-validated average precision was 0.428, against 0.426 for the forest and 0.398 for logistic regression. On the held-out exam it finds about 62% of the subscribers, and about one in three of its "yes" calls is right. The endpoint returns the probability. A bank can raise the cutoff if wasted calls are expensive. 0.5 is a teaching default, not a business rule.

The selected model's accuracy is 81%. That is worse than always saying no. That sentence is the one to remember.

## Why the cloud is only half the story

Training happened on a laptop. The table is small. A compute cluster would have been a bill, not a lesson.

This subscription also disables public access on storage accounts, and it will not let you turn that access back on. A laptop cannot upload a model file straight to the workspace storage account. The fix is a managed virtual network: Azure ML gets a private door to storage, and the register-and-deploy step runs from compute that can open that door. The scoring URL is still a normal HTTPS endpoint.

It costs about $0.085 an hour for the smallest CPU, and it does not turn itself off. Delete it when you are done looking.

## What I would say in an interview

I would not lead with the algorithm. I would lead with the column I refused, the metric I refused, and the fact that the test set did not choose the winner. The three models are the easy part. The discipline is the project.

The review worksheet is [LEARNING_GUIDE.md](../LEARNING_GUIDE.md). Read the decisions before you read the answers.
