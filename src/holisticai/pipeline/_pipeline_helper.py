from sklearn.pipeline import Pipeline as SKLPipeline

from holisticai.pipeline.handlers._estimator import EstimatorHandler
from holisticai.pipeline.handlers._pipeline_params import PipelineParametersHandler
from holisticai.pipeline.handlers._utransformers import UTransformersHandler

SUPPORTED_FUNCTIONS = [
    "fit",
    "predict",
    "predict_proba",
    "predict_score",
    "predictions",
]
POST_PREDICTION = {
    "predict": "y_pred",
    "predict_score": "y_score",
    "predict_proba": "y_proba",
}
POST_SCORE = "score"


class PipelineHelper:
    def __getattribute__(self, name):
        """
        This function return the attribute using the following rules:
        - for all functions listed in SUPPORTED_FUNCTIONS return a wrapped function that update the input arguments with
          the cache before call fit function.
        - for any other function we check if the pipeline has the attribute else try the attribute from the main class.
        """
        if name in SUPPORTED_FUNCTIONS:
            return object.__getattribute__(self, "handle_pipeline_methods")(name)
        else:  # noqa: RET505
            return object.__getattribute__(self, name)

    def handle_pipeline_methods(self, fn_name):
        def function(X, y=None, **kargs):
            params = self.preprocessing_parameters(kargs, y)

            if self.post_estimator_transformers:
                if fn_name in POST_PREDICTION:
                    # For predictions we must check supported methods
                    fn = object.__getattribute__(self, "predictions")
                    output = fn(X, **params)[POST_PREDICTION[fn_name]]

                elif fn_name == POST_SCORE:
                    # For score we must check supported methods
                    output = fn = object.__getattribute__(self, "score")

                else:
                    # For fitting model the method mus be called from the main class
                    output = object.__getattribute__(self, fn_name)(X, **params)

            else:
                # Other case the primitive method could be invoked
                output = getattr(SKLPipeline, fn_name)(self, X, **params)

            return output

        return function

    @property
    def post_estimator_transformers(self):
        """
        return list o post-estimator transformers
        """
        return self.utransformers_hdl.pos_processing_steps

    def preprocessing_steps(self, steps):
        """
        Preprocessing steps before calling sklearn.pipeline.Pipeline contructor

        Description
        ----------
        This function collects all bias mitigation steps, remove steps after estimator and wrap the estimator

        Parameters
        ----------
        steps : list
            pipeline steps

        Returns
        -------
        list
            sklearn pipeline steps

        """
        self.params_hdl = PipelineParametersHandler()
        self.estimator_hdl = EstimatorHandler(self.params_hdl)
        self.utransformers_hdl = UTransformersHandler(steps, self.params_hdl, self.estimator_hdl)

        steps = self.utransformers_hdl.drop_post_processing_steps(steps)
        return self.estimator_hdl.wrap_and_link_estimator_step(steps)

    def preprocessing_parameters(self, params, y=None):
        """
        Preprocessing parameters.

        Description
        ----------
        The function filter, drop and feed parameters for unconventional transformers.

        Parameters
        ----------
        params : dict
            dictionary of parameters

        Returns
        -------
        dict
            estimator parameters

        """
        params = self.params_hdl.bias_mitigator.feed(params, return_dropped=True)
        self.params_hdl.estimator.feed(params)

        if y is not None:
            params.update({"y": y})
            self.params_hdl.estimator["y"] = y

        if "sample_weight" in params:
            self.params_hdl.estimator["sample_weight"] = params["sample_weight"]

        return params

    def fit_post_estimator_transformers(self, Xt, y):
        """
        fit post-estimator transformers.

        Description
        ----------
        Complete the pipeline fit procedure for transformers defined after the estimator in the pipeline.
        Prediction is necessary to avoid conflicts with preprocessing approaches during fitting.

        Parameters
        ----------
        Xt : numpy array
            Transformed input matrix

        y : numpy array
            Target vector

        Returns
        -------
        None
        """
        output_kargs = self.estimator_hdl.get_fit_params(Xt)
        self.utransformers_hdl.fit_postprocessing(Xt, y=y, **output_kargs)

    def _transform_post_estimator_transformers(self, Xt, **params):
        """
        call transform for post-estimator transformers.

        Description
        ----------
        Complete the pipeline transform procedure for transformers defined after the estimator in the pipeline.

        Parameters
        ----------
        Xt : numpy array
            Transformed input matrix

        Returns
        -------
        dict
            Post-processed predictions vectors
        """

        output_kargs = self.estimator_hdl.get_transform_params(Xt)
        transform_kargs = {k[4:]: v for k, v in params.items() if k.startswith("bm__")}
        transform_kargs["X"] = Xt
        output_kargs.update(transform_kargs)
        return self.utransformers_hdl.transform_postprocessing(**output_kargs)

    def _transform_without_final(self, X):
        """Transform the data. Not apply `transform` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `transform` method. Only valid if the final estimator
        implements `transform`.

        This also works where final estimator is `None` in which case all prior
        transformations are applied.

        Parameters
        ----------
        X : matrix-like
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed data.
        """
        Xt = X
        for _, _, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return Xt
