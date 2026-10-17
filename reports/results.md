# Model comparison

| Model | CV average precision | Test average precision | Test ROC-AUC | Test F1 | Test precision | Test recall | Test accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | 0.398 | 0.409 | 0.772 | 0.375 | 0.268 | 0.627 | 0.756 |
| random_forest | 0.426 | 0.437 | 0.793 | 0.436 | 0.347 | 0.587 | 0.822 |
| gradient_boosting (selected) | 0.428 | 0.455 | 0.789 | 0.434 | 0.335 | 0.616 | 0.812 |

The selected model was chosen by **cross-validated average precision** on the training split only. Test columns are the final exam. They were not used to pick the winner.

Numbers were written by train.py.
