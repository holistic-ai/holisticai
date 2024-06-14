class WEstimator:
    """
    This wrap the estimator class and helps link parameters with updates done by unconventional transformers during pipeline execution.
    """

    def __init__(self, obj, params_hdl):
        """
        Paramters
        ---------
        obj : object
            model object

        params_hdl : UTransformersHandler
            Pipeline parameters handler during fit, fit_transform, transform function execution.
        """
        self.obj = obj
        self.params_hdl = params_hdl

    def __getattribute__(self, name):
        """
        This function return the attribute using the following the rules:
        - for `fit` function: return a wrapped function that update the input arguments with the cache and then call fit function.
        - for any other function we check:
            - if the estimator has the attribute -> invoke it
            - else -> try to invoke the function from the Wrapper class.
        """
        if name.startswith("fit"):

            def fitwrapper(X, y=None, **kargs):
                # kargs.update(getattr(object.__getattribute__(self, 'params_hdl'), 'get_estimator_paramters')())
                kargs.update(self.params_hdl.get_estimator_paramters())
                fit_params = {}
                if y is not None:
                    fit_params.update({"y": y})
                fit_params.update(kargs)
                return getattr(object.__getattribute__(self, "obj"), name)(X, **fit_params)

            output = fitwrapper
        elif hasattr(object.__getattribute__(self, "obj"), name):
            output = getattr(object.__getattribute__(self, "obj"), name)
        else:
            output = object.__getattribute__(self, name)
        return output


class EstimatorHandler:
    """
    This class handles actions realted to the estimator step (westimator).
    The class wraps the estimator and allows update the estimator parameters during a pipeline execution.
    """

    def __init__(self, params_hdl):
        self.params_hdl = params_hdl

    def wrap_and_link_estimator_step(self, steps):
        """
        Modify the estimator step and link step name and estimator.

        Description
        ----------
        This function wrap the estimator, save the step name and the westimator reference and
        return the updated steps list.

        Parameters
        ----------
        steps : list
            pipeline steps list

        Returns
        -------
        list

        """
        self.estimator_name = steps[-1][0]
        self.params_hdl.create_estimator_parameters(self.estimator_name, steps[-1][1])
        steps[-1] = (steps[-1][0], WEstimator(steps[-1][1], self.params_hdl))
        self.estimator = steps[-1][1]
        return steps

    def get_fit_params(self, Xt):
        kargs = {}

        y_pred = self.estimator.predict(Xt)
        kargs["y_pred"] = y_pred

        if hasattr(self.estimator, "predict_proba"):
            y_proba = self.estimator.predict_proba(Xt)
            kargs["y_proba"] = y_proba
        return kargs

    def get_transform_params(self, Xt):
        kargs = {}

        y_pred = self.estimator.predict(Xt)
        kargs["y_pred"] = y_pred

        if hasattr(self.estimator, "predict_proba"):
            y_proba = self.estimator.predict_proba(Xt)
            kargs["y_proba"] = y_proba
        return kargs
