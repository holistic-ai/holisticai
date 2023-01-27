from abc import ABC, abstractmethod

import numpy as np


class TransformerBase:
    """
    Base implementation for unconventional transformers.
    For each pipeline execution the transformer will be linked to the estimator and custom parameters.
    """

    def has_param_handler(self):
        return hasattr(self, "params_hdl")

    def link_parameters(self, params_hdl, estimator_hdl):
        self.params_hdl = params_hdl
        self.estimator_hdl = estimator_hdl

    def get_bias_param(self, param_name, default=None):
        if param_name in self.params_hdl.bias_mitigator:
            return self.params_hdl.bias_mitigator[param_name]
        else:
            return default

    def update_bias_param(self, param_name, param_value):
        if hasattr(self, "params_hdl"):
            self.params_hdl.bias_mitigator[param_name] = param_value
        else:
            self.bias_mitigator_params[param_name] = param_value

    def get_estimator_param(self, param_name, default=None):
        if param_name in self.params_hdl.estimator:
            return self.params_hdl.estimator[param_name]
        else:
            return default

    def update_estimator_param(self, param_name, param_value):
        if hasattr(self, "params_hdl"):
            self.params_hdl.estimator[param_name] = param_value
        else:
            self.estimator_params[param_name] = param_value


class BMTransformerBase(ABC, TransformerBase):
    """
    Bias Mitigation Transformer Base
    The class implement input preprocessing for all Bias Mitigators Types.
    """

    def _get_param(self, kargs, param_name):
        if hasattr(self, param_name):
            return getattr(self, param_name)
        elif param_name in kargs:
            return kargs[param_name]
        else:
            return None

    def _to_numpy(self, kargs, value_name, ravel=True):
        value = np.array(kargs[value_name])
        return value.ravel() if ravel else value

    def _load_data_pipeline(self):

        if not hasattr(self, "params_hdl"):
            return {}

        params = {}
        bm_param_names = ["group_a", "group_b"]
        for param_name in bm_param_names:
            if param_name in self.params_hdl.bias_mitigator:
                params[param_name] = self.params_hdl.bias_mitigator[param_name]
            elif hasattr(self, param_name):
                params[param_name] = getattr(self, param_name)

        es_param_names = ["sample_weight", ("y", "y_true")]
        for param_name in es_param_names:
            if isinstance(param_name, tuple):
                es_param_name, param_name = param_name
            else:
                es_param_name = param_name

            if es_param_name in self.params_hdl.estimator:
                params[param_name] = self.params_hdl.estimator[es_param_name]

        return params

    def reformat_function(self, func):
        def wrapped_func(*args, **kargs):
            self.estimator_params = {}
            self.bias_mitigator_params = {}
            fun_varnames = func.__code__.co_varnames[1:]
            params = dict(zip(fun_varnames, args))
            kargs.update(params)
            kargs.update(self._load_data_pipeline())
            params = {v: kargs[v] for v in fun_varnames if v in kargs}
            return func(**params)

        return wrapped_func

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)

        if hasattr(obj, "fit"):
            docstring = obj.fit.__doc__
            obj.fit = obj.reformat_function(obj.fit)
            obj.fit.__doc__ = docstring

        if hasattr(obj, "transform"):
            docstring = obj.transform.__doc__
            obj.transform = obj.reformat_function(obj.transform)
            obj.transform.__doc__ = docstring

        if hasattr(obj, "fit_transform"):
            docstring = obj.fit_transform.__doc__
            obj.fit_transform = obj.reformat_function(obj.fit_transform)
            obj.fit_transform.__doc__ = docstring

        if hasattr(obj, "predict"):
            docstring = obj.predict.__doc__
            obj.predict = obj.reformat_function(obj.predict)
            obj.predict.__doc__ = docstring

        if hasattr(obj, "predict_proba"):
            docstring = obj.predict_proba.__doc__
            obj.predict_proba = obj.reformat_function(obj.predict_proba)
            obj.predict_proba.__doc__ = docstring

        obj.estimator_params = {}
        obj.bias_mitigator_params = {}
        return obj
