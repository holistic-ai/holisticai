import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from holisticai.pipeline import Pipeline
from holisticai.utils.transformers.bias import BIAS_TAGS
from tests.testing_utils._tests_data_utils import load_preprocessed_adult
from tests.testing_utils._tests_utils import data_info, evaluate_pipeline, fit

seed = 42
train_data, test_data = load_preprocessed_adult()


def test_incorrect_preprocessing_tag_in_postprocessing():
    from holisticai.bias.mitigation import EqualizedOdds

    def get_pipeline():
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("estimator", LogisticRegression()),
                (f"{BIAS_TAGS.PRE}processing", EqualizedOdds()),
            ]
        )

    error_message = "Mitigator objects and name doesn't match"
    pytest.raises(TypeError, get_pipeline, match=error_message)


def test_incorrect_preprocessing_object_in_postprocessing():
    from holisticai.bias.mitigation import Reweighing

    def get_pipeline():
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("estimator", LogisticRegression()),
                (f"{BIAS_TAGS.POST}processing", Reweighing()),
            ]
        )

    error_message = "Mitigator objects and name doesn't match"
    pytest.raises(TypeError, get_pipeline, match=error_message)


def test_incorrect_postprocessing_object_in_preprocessing():
    from holisticai.bias.mitigation import EqualizedOdds

    def get_pipeline():
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (f"{BIAS_TAGS.PRE}processing", EqualizedOdds()),
                ("estimator", LogisticRegression()),
            ]
        )

    error_message = "Mitigator objects and name doesn't match"
    pytest.raises(TypeError, get_pipeline, match=error_message)
