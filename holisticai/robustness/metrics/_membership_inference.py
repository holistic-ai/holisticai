import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)


def evaluate_membership_inference(target, inferred):
    cm = confusion_matrix(target.reshape(-1), inferred.reshape(-1), normalize="true")
    acc = accuracy_score(target, inferred)
    precision = precision_score(inferred, target)
    recall = recall_score(inferred, target)

    test_acc = cm[0, 0]
    train_acc = cm[1, 1]

    return pd.DataFrame.from_dict(
        {
            "Members Accuracy": train_acc,
            "Non Members Accuracy": test_acc,
            "Attack Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
        },
        orient="index",
    )
