import sys
sys.path.append('./././')

from holisticai.benchmark.tasks import  get_task

task = get_task("regression")
task.benchmark(type='postprocessing')

import numpy as np
from holisticai.bias.mitigation.postprocessing.plugin_estimator_and_recalibration.algorithm import PluginEstimationAndCalibrationAlgorithm

class MyPostprocessingMitigator():
    """
    My Postprocessing Mitigator
    """
    def __init__(self):
        self.algorithm_ = PluginEstimationAndCalibrationAlgorithm()

    def fit(self, y_pred, group_a, group_b):
        sensitive_features = np.stack([group_a, group_b], axis=1)
        self.algorithm_.fit(y_pred, sensitive_features)
        return self

    def transform(self, y_pred, group_a, group_b):
        sensitive_features = np.stack([group_a, group_b], axis=1)
        new_y_pred = self.algorithm_.transform(y_pred, sensitive_features)
        return {"y_pred": new_y_pred}
    
my_mitigator = MyPostprocessingMitigator()

task.run_benchmark(custom_mitigator = my_mitigator, type = 'postprocessing')