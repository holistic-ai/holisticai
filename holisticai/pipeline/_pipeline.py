import numpy as np
from sklearn.pipeline import Pipeline as SKLPipeline
from sklearn.utils.metaestimators import available_if

from holisticai.pipeline._pipeline_helper import PipelineHelper


def _fulfill_conditions(fn_name: str):
    def check(self):
        if fn_name == "predict_proba":
            return (
                hasattr(self._final_estimator, "predict_proba")
                and not self.post_estimator_transformers
            )
        if fn_name == "predict_score":
            return hasattr(self, "predict_proba") and self.post_estimator_transformers
        if fn_name == "predictions":
            return self.post_estimator_transformers

    return check


class Pipeline(SKLPipeline, PipelineHelper):
    """
    Holistic AI Pipeline

    Description
    -----------
    Holisticai pipeline wrap the sklearn pipeline to support unconventional transformers.
    Unconventional transformers (u-transformers) are transformers that doesn't follow the typically
    sklearn workflow. For example, Bias Mitigator needs update inputs, outputs, and other parameters during the
    fit and transform process. The current version of this pipeline supports only binary
    classification.
    """

    def __init__(self, steps, *, memory=None, verbose=False):
        """
        Initialize Holistic AI Pipeline

        Description
        -----------
        Preprocess the steps before pass to sklearn pipeline. The preprocessing map the bias mitigators and
        and wrap the estimator so we can share paramters during the fit ans transform function.

        Parameters
        ----------
        steps: list
            A list of transformers/u-transformers and estimator

        memory : str or object with the joblib.Memory interface, default=None
            Used to cache the fitted transformers of the pipeline. By default,
            no caching is performed. If a string is given, it is the path to
            the caching directory. Enabling caching triggers a clone of
            the transformers before fitting. Therefore, the transformer
            instance given to the pipeline cannot be inspected
            directly. Use the attribute ``named_steps`` or ``steps`` to
            inspect estimators within the pipeline. Caching the
            transformers is advantageous when fitting is time consuming.

        verbose : bool, default=False
            If True, the time elapsed while fitting each step will be printed as it
            is completed.
        """
        steps = self.preprocessing_steps(steps)
        super(Pipeline, self).__init__(steps=steps, memory=memory, verbose=verbose)

    def fit(self, X, y=None, **fit_params):
        """Fit the model.

        Fit all the transformers/u-transformers one after the other and transform the
        data and parameters. Then, fit the transformed data using the final estimator.
        Finally, fit the u-transformers for postprocessing.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : object
            Pipeline with fitted steps.
        """
        super().fit(X, y, **fit_params)
        Xt = self._transform_without_final(X)
        self.fit_post_estimator_transformers(Xt, y)
        return self

    @available_if(_fulfill_conditions("predict_proba"))
    def predict_proba(self, X, **predict_proba_params):
        """Update avaiable conditions for predict_proba"""
        return super().predict_proba(X, **predict_proba_params)

    @available_if(_fulfill_conditions("predict_score"))
    def predict_score(self, X, **predict_score_params):
        """
        Return probability vector

        Description
        -----------

        Transform the data, and the postprocessor u-transformer compute the predictions for that model.
        Only available with postprocessors bias mitigator.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        np.ndarray
            probability value for each example
        """
        return self.predictions(X, **predict_score_params)["y_score"]

    @available_if(_fulfill_conditions("predictions"))
    def predictions(self, X, **params):
        """
        Post-processor prediction

        Description
        -----------

        Transform the data, and the postprocessor u-transformer compute the predictions for that model.
        If the pipeline doesn't have postprocessors, sklearn pipeline functions are used
        for prediction.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the pipeline. Note that while this may be
            used to return uncertainties from some models with return_std
            or return_cov, uncertainties that are generated by the
            transformations in the pipeline are not propagated to the
            final estimator.

            .. versionadded:: 0.20

        Returns
        -------
        dict
            dictionary with postprocessor outputs
        """
        Xt = self._transform_without_final(X)
        return self._transform_post_estimator_transformers(Xt, **params)
