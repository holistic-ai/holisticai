import numpy as np
import pandas as pd

# def classification_metrics(y_pred, y_true, positive_value):
#     train_acc = np.sum(np.around(y_pred, decimals=6) == np.around(y_true, decimals=6).reshape(1,-1)) / len(y_pred)
#
#     precision, recall = calc_precision_recall(y_pred, np.around(y_true, decimals=8), positive_value=positive_value)
#
#     df = pd.DataFrame()
#     df = df.append({'Accuracy':train_acc, 'Precision':precision, 'Recall':recall}, ignore_index=True)
#     df = df.append({'Accuracy':1, 'Precision':1, 'Recall':1}, ignore_index=True).T
#     df.columns = ['Values', 'Reference']
#     df['Reference'] = df['Reference'].astype('int32')
#     return df


def classification_metrics(y_pred, y_true, positive_value):
    train_acc = np.sum(
        np.around(y_pred, decimals=6) == np.around(y_true, decimals=6).reshape(1, -1)
    ) / len(y_pred)

    precision, recall = calc_precision_recall(
        y_pred, np.around(y_true, decimals=8), positive_value=positive_value
    )

    df = pd.DataFrame()
    new_row = pd.DataFrame(
        {"Accuracy": [train_acc], "Precision": [precision], "Recall": [recall]}
    )
    df = pd.concat([df, new_row], ignore_index=True)

    new_row = pd.DataFrame({"Accuracy": [1], "Precision": [1], "Recall": [1]})
    df = pd.concat([df, new_row], ignore_index=True).T
    df.columns = ["Values", "Reference"]
    df["Reference"] = df["Reference"].astype("int32")
    return df


def calc_precision_recall(predicted, actual, positive_value=1):
    score = 0  # both predicted and actual are positive
    num_positive_predicted = 0  # predicted positive
    num_positive_actual = 0  # actual positive
    for i in range(len(predicted)):
        if predicted[i] == positive_value:
            num_positive_predicted += 1
        if actual[i] == positive_value:
            num_positive_actual += 1
        if predicted[i] == actual[i]:
            if predicted[i] == positive_value:
                score += 1

    if num_positive_predicted == 0:
        precision = 1
    else:
        precision = (
            score / num_positive_predicted
        )  # the fraction of predicted “Yes” responses that are correct
    if num_positive_actual == 0:
        recall = 1
    else:
        recall = (
            score / num_positive_actual
        )  # the fraction of “Yes” responses that are predicted correctly

    return precision, recall
