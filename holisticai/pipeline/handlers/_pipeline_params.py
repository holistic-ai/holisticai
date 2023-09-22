class ParametersHandler:
    def __init__(self, param_names=None, step_name=None):
        self.param_names = param_names
        self.step_name = step_name
        self.clean_parameters()

    def clean_parameters(self):
        self.dict_params = {}

    def set_shared_parameters(self, dict_params):
        self.dict_params = dict_params

    def __contains__(self, param_name):
        return param_name in self.dict_params

    def __getitem__(self, param_name):
        return self.dict_params[param_name]

    def __setitem__(self, param_name, param_value):
        self.dict_params[param_name] = param_value

    def feed(self, params, return_dropped=False):
        dropped_params = {}
        self.clean_parameters()
        if self.step_name:
            for name, value in params.items():
                if name.startswith(self.step_name):
                    param_name = name.split("__", 1)[1]
                    self[param_name] = value
                else:
                    dropped_params[name] = value

        elif self.param_names:

            for name, value in params.items():
                if name in self.param_names:
                    param_name = name.split("__", 1)[1]
                    self[param_name] = value
                else:
                    dropped_params[name] = value

        if return_dropped:
            return dropped_params


class PipelineParametersHandler:
    def __init__(self):
        self.bias_mitigator = ParametersHandler(
            param_names=["bm__group_a", "bm__group_b"]
        )

    def create_estimator_parameters(self, estimator_name, estimator):
        self.estimator = ParametersHandler(step_name=estimator_name)
        self.estimator_model = estimator

    def get_estimator_paramters(self):
        return self.estimator.dict_params
