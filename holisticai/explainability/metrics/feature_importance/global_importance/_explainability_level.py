import numpy as np
import pandas as pd

from ..utils import partial_dependence_creator


def compute_feature_scores(data, threshold):
    scores = [
        {
            "feature": feat,
            "scores": sum([1 if rr > threshold else 0 for rr in r]),
            "few_points": flag,
        }
        for feat, (r, flag) in data.items()
    ]
    scores = pd.DataFrame(scores)[["few_points", "feature", "scores"]]
    scores = scores.sort_values("scores", ascending=False)
    return scores


def compute_similarity(df, num_chunks):
    chunk_size = len(df) // num_chunks
    few_points = False

    if chunk_size == 0:
        few_points = True
        return (1, 1), few_points

    y_data = df["score"].values
    y_data = y_data[: num_chunks * chunk_size]
    c = np.stack(np.split(y_data, num_chunks), axis=0)
    dc = c[:, 1:] - c[:, :-1]

    sims = []
    for i in [2, 1]:
        norm = np.linalg.norm(dc[i])
        if norm > 0:
            sim = np.matmul(dc[i], dc[i - 1]) / (
                np.linalg.norm(dc[i]) * np.linalg.norm(dc[i - 1])
            )
        else:
            sim = 1
        sims.append(sim)

    return sims, few_points


def compute_partial_dependence(model, feature_importance, x, target=None):
    num_chunks = 3
    threshold = 0
    grid_resolution = 50
    index_to_class = {0: "Hard", 1: "Medium", 2: "Easy"}
    class_to_score = {"Hard": 0, "Medium": 0.5, "Easy": 1}
    class_to_index = {c: i for i, c in index_to_class.items()}
    categories = class_to_index.keys()

    feature_importance_indexes = list(feature_importance["Variable"].index)

    partial_dependence = partial_dependence_creator(
        model,
        grid_resolution=grid_resolution,
        x=x,
        feature_ids=feature_importance_indexes,
        target=target,
    )
    data = {
        feat: compute_similarity(df, num_chunks)
        for feat, df in partial_dependence.items()
    }
    score_data = compute_feature_scores(data, threshold)
    score_data["scores"] = score_data.apply(
        lambda x: index_to_class[x["scores"]], axis=1
    )
    score = score_data.groupby("scores").count()["feature"]
    score_dict = pd.DataFrame(score / score.sum()).to_dict()["feature"]

    values = []
    full_score = {c: 0 for c in categories}
    for c, v in score_dict.items():
        full_score[c] = v
        values.append(class_to_score[c] * full_score[c])

    score = sum(values)

    # for debbug
    # easy = int(full_score['Easy']*100)
    # medium = int(full_score['Medium']*100)
    # hard = int(full_score['Hard']*100)

    # metric = {'score':score, 'easy':easy, 'medium':medium, 'hard':hard}

    return score


def explainability_ease_score(model_type, model, x, target, feature_importance):
    if model_type == "binary_classification":
        targets = sorted(set(model.classes_) - {0})
        result = pd.DataFrame.from_dict(
            {
                "Global Explainability Ease Score": compute_partial_dependence(
                    model, feature_importance, x, target
                )
                for target in targets
            },
            orient="index",
        )

    elif model_type == "regression":
        result = pd.DataFrame.from_dict(
            {
                "Global Explainability Ease Score": compute_partial_dependence(
                    model, feature_importance, x
                )
            },
            orient="index",
        )

    return result.rename(columns={0: "Value"})
