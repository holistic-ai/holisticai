from __future__ import annotations

import numpy as np
import pandas as pd
from holisticai.utils import Importances, PartialDependence


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
    return scores.sort_values("scores", ascending=False)


def calculate_discrete_derivative(y_values):
    """Calculate the discrete derivative for a sequence of y values."""
    dy = np.diff(y_values)
    dx = np.ones_like(dy)  # Assuming x values are equally spaced with a difference of 1
    return dy / dx


def cosine_similarity(v1, v2):
    """Calculate the cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


def compare_tangents(points):
    num_sections = 3
    few_points = False
    n = len(points)
    tol = 1e-5
    if n < num_sections:
        few_points = True
        return (1, 1), few_points

    cut1 = n // 3
    cut2 = 2 * n // 3

    section1 = points[: cut1 + 1]
    section2 = points[cut1 : cut2 + 1]
    section3 = points[cut2:]

    sections = [section1, section2, section3]
    slopes = []
    for section in sections:
        if len(section) > 1:
            section_slopes = calculate_discrete_derivative(section)
            average_slope = np.mean(section_slopes)
        else:
            average_slope = 0
        slopes.append(average_slope + tol)

    similarities = []
    for i in range(len(slopes) - 1):
        similarity = cosine_similarity([slopes[i]], [slopes[i + 1]])
        similarities.append(similarity)
    return similarities, few_points


class XAIEaseAnnotator:
    threshold: float = 0
    levels: list[str] = ["Hard", "Medium", "Easy"]

    def compute_xai_ease_score_data(self, partial_dependence, ranked_feature_importance):
        """
        Computes the XAI Ease Score data for a given partial dependence plot.

        Args:
            partial_dependence (dict): A dictionary containing the partial dependence plots for each feature.

        Returns:
            score_data (DataFrame): The computed score data.
        """
        partial_dependence_formatted = {
            f: partial_dependence[i]["average"][0] for i, f in enumerate(ranked_feature_importance.feature_names)
        }
        data = {feat: compare_tangents(df) for feat, df in partial_dependence_formatted.items()}
        score_data = compute_feature_scores(data, self.threshold)
        score_data["scores"] = score_data.apply(lambda x: self.levels[x["scores"]], axis=1)
        return score_data


class XAIEaseScore:
    """
    Class for computing the XAI Ease Score.

    The XAI Ease Score measures the ease of interpretability of a model's explanations.
    It takes into account the similarity between partial dependence plots of different features
    and assigns scores based on the similarity values.

    Attributes:
        num_chunks (int): The number of chunks to divide the partial dependence plots into.
        threshold (float): The threshold value for computing feature scores.
        levels (list): The levels of ease scores, in descending order of difficulty.

    Methods:
        __compute_xai_ease_score: Computes the XAI Ease Score for a given score data.
        __call__: Computes the XAI Ease Score for a set of partial dependence plots.
        __xai_feature_ease_score: Computes the XAI Ease Score for a single partial dependence plot.
    """

    reference: float = 1.0
    name: str = "XAI Ease Score"
    annotator: XAIEaseAnnotator = XAIEaseAnnotator()

    def __init__(self, detailed: bool = False):
        self.detailed = detailed

    def compute_xai_ease_score(self, score_data):
        """
        Computes the XAI Ease Score for a given score data.

        Args:
            score_data (DataFrame): The score data.

        Returns:
            xai_ease_score (float): The computed XAI Ease Score.
        """
        max_score = 2
        score_dict = pd.DataFrame(
            score_data.groupby("scores").count()["feature"] / score_data.groupby("scores").count()["feature"].sum()
        ).to_dict()["feature"]

        values = []
        full_score = {c: 0 for c in self.annotator.levels}
        for c, v in score_dict.items():
            full_score[c] = v
            values.append(self.annotator.levels.index(c) * full_score[c])

        return sum(values) / max_score

    def __call__(
        self,
        partial_dependence: PartialDependence,
        ranked_feature_importance: Importances,
    ):
        """
        Computes the XAI Ease Score for a set of partial dependence plots.

        Args:
            partial_dependence (list): A list of dictionaries containing the partial dependence plots for each feature.
            features (list): A list of feature names.

        Returns:
            float: The computed XAI Ease Score.
        """

        def compute_metric(pdep, rfi):
            score_data = self.annotator.compute_xai_ease_score_data(pdep, rfi)
            return float(self.compute_xai_ease_score(score_data))

        scores = [compute_metric(pdep, ranked_feature_importance) for pdep in partial_dependence.values]
        if self.detailed:
            return scores
        return float(np.mean(scores))


def xai_ease_score(partial_dependence: PartialDependence, ranked_feature_importance: Importances):
    """
    This metric, ranging from 0 to 1, measures the ease of explaining a model's predictions for the top feature importances (>80%) \
    using partial dependence plots. The XAI Ease Score considers the similarity between regions of the partial dependence plots \
    for different features and assigns scores based on these similarity values.

    Parameters
    ----------
    partial_dependence: PartialDependence
        The partial dependence values for each feature.
    ranked_feature_importance: Importances
        The ranked feature importance values.

    Returns
    -------
        float: The XAI Ease Score.

    Examples
    --------
    >>> from holisticai.explainability.commons import PartialDependence, Importances
    >>> from holisticai.explainability.metrics.global_importance import xai_ease_score
    >>> partial_dependence = [
    ...     {
    ...         "average": [[0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3]],
    ...         "grid_values": [[1, 2, 3, 4, 5, 6, 7, 8, 9]],
    ...     },
    ...     {
    ...         "average": [[0.4, 0.5, 0.6, 0.6, 0.5, 0.4, 0.4, 0.5, 0.6]],
    ...         "grid_values": [[1, 2, 3, 4, 5, 6, 7, 8, 9]],
    ...     },
    ... ]
    >>> partial_dependence = PartialDependence(values=partial_dependence)
    >>> feature_importance = Importances(values=np.array([0.5, 0.5]),
    >>> feature_names=['feature1', 'feature2'])
    >>> xai_ease_score(partial_dependence, feature_importance)
    0.5
    """
    metric = XAIEaseScore()
    return metric(partial_dependence, ranked_feature_importance)
