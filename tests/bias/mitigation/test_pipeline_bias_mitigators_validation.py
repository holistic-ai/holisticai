import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from holisticai.pipeline import Pipeline
from holisticai.utils.transformers.bias import BIAS_TAGS


def test_incorrect_pipeline_mitigators_tags():
    from holisticai.mitigation.bias import EqualizedOdds, Reweighing

    def incorrect_post_processing_pipeline_tag_definition():
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("estimator", LogisticRegression()),
                (f"{BIAS_TAGS.PRE}processing", EqualizedOdds()),
            ]
        )

    error_message = "Mitigator objects and name doesn't match"
    pytest.raises(
        TypeError,
        incorrect_post_processing_pipeline_tag_definition,
        match=error_message,
    )

    def incorrect_post_processing_pipeline_mitigator_definition():
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("estimator", LogisticRegression()),
                (f"{BIAS_TAGS.POST}processing", Reweighing()),
            ]
        )

    error_message = "Mitigator objects and name doesn't match"
    pytest.raises(
        TypeError,
        incorrect_post_processing_pipeline_mitigator_definition,
        match=error_message,
    )

    def incorrect_inp_processing_pipeline_mitigator_definition():
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (f"{BIAS_TAGS.PRE}processing", EqualizedOdds()),
                ("estimator", LogisticRegression()),
            ]
        )

    error_message = "Mitigator objects and name doesn't match"
    pytest.raises(
        TypeError,
        incorrect_inp_processing_pipeline_mitigator_definition,
        match=error_message,
    )
