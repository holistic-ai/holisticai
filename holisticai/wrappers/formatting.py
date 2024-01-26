# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Module providing convenience functions.
"""
# pylint: disable=C0302
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import os
import shutil
import sys
import tarfile
import warnings
import zipfile
from functools import wraps
from inspect import signature
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import numpy as np
import six
from scipy.special import gammainc  # pylint: disable=E0611
from tqdm.auto import tqdm

from holisticai.robustness.mitigation import config

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------- CONSTANTS AND TYPES


from holisticai.wrappers.certification.deep_z import PytorchDeepZ
from holisticai.wrappers.certification.derandomized_smoothing.derandomized_smoothing import (
    BlockAblator,
    ColumnAblator,
)
from holisticai.wrappers.classification.blackbox import BlackBoxClassifier
from holisticai.wrappers.classification.catboost import CatBoostARTClassifier
from holisticai.wrappers.classification.classifier import (
    Classifier,
    ClassifierClassLossGradients,
    ClassifierDecisionTree,
    ClassifierLossGradients,
    ClassifierNeuralNetwork,
)
from holisticai.wrappers.classification.detector_classifier import DetectorClassifier
from holisticai.wrappers.classification.ensemble import EnsembleClassifier
from holisticai.wrappers.classification.GPy import GPyGaussianProcessClassifier
from holisticai.wrappers.classification.keras import KerasClassifier

# from art.experimental.estimators.classification.jax import JaxClassifier
from holisticai.wrappers.classification.lightgbm import LightGBMClassifier
from holisticai.wrappers.classification.mxnet import MXClassifier
from holisticai.wrappers.classification.pytorch import PyTorchClassifier
from holisticai.wrappers.classification.query_efficient_bb import (
    QueryEfficientGradientEstimationClassifier,
)
from holisticai.wrappers.classification.scikitlearn import (
    ScikitlearnAdaBoostClassifier,
    ScikitlearnBaggingClassifier,
    ScikitlearnClassifier,
    ScikitlearnDecisionTreeClassifier,
    ScikitlearnDecisionTreeRegressor,
    ScikitlearnExtraTreeClassifier,
    ScikitlearnExtraTreesClassifier,
    ScikitlearnGradientBoostingClassifier,
    ScikitlearnLogisticRegression,
    ScikitlearnRandomForestClassifier,
    ScikitlearnSVC,
)
from holisticai.wrappers.classification.tensorflow import (
    TensorFlowClassifier,
    TensorFlowV2Classifier,
)
from holisticai.wrappers.classification.xgboost import XGBoostClassifier
from holisticai.wrappers.generation import TensorFlowGenerator
from holisticai.wrappers.generation.tensorflow import TensorFlowV2Generator

# from holisticai.wrappers.object_detection.object_detector import ObjectDetector
# from holisticai.wrappers.object_detection.pytorch_object_detector import PyTorchObjectDetector
# from holisticai.wrappers.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN
# from holisticai.wrappers.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN
from holisticai.wrappers.pytorch import PyTorchEstimator
from holisticai.wrappers.regression.scikitlearn import ScikitlearnRegressor

# from holisticai.wrappers.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech
# from holisticai.wrappers.speech_recognition.tensorflow_lingvo import TensorFlowLingvoASR
from holisticai.wrappers.tensorflow import TensorFlowV2Estimator

CLASSIFIER_LOSS_GRADIENTS_TYPE = Union[  # pylint: disable=C0103
    ClassifierLossGradients,
    EnsembleClassifier,
    GPyGaussianProcessClassifier,
    KerasClassifier,
    # JaxClassifier,
    MXClassifier,
    PyTorchClassifier,
    ScikitlearnLogisticRegression,
    ScikitlearnSVC,
    TensorFlowClassifier,
    TensorFlowV2Classifier,
    QueryEfficientGradientEstimationClassifier,
]

CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE = Union[  # pylint: disable=C0103
    ClassifierClassLossGradients,
    EnsembleClassifier,
    GPyGaussianProcessClassifier,
    KerasClassifier,
    MXClassifier,
    PyTorchClassifier,
    ScikitlearnLogisticRegression,
    ScikitlearnSVC,
    TensorFlowClassifier,
    TensorFlowV2Classifier,
]

CLASSIFIER_NEURALNETWORK_TYPE = Union[  # pylint: disable=C0103
    ClassifierNeuralNetwork,
    DetectorClassifier,
    EnsembleClassifier,
    KerasClassifier,
    MXClassifier,
    PyTorchClassifier,
    TensorFlowClassifier,
    TensorFlowV2Classifier,
]

CLASSIFIER_DECISION_TREE_TYPE = Union[  # pylint: disable=C0103
    ClassifierDecisionTree,
    LightGBMClassifier,
    ScikitlearnDecisionTreeClassifier,
    ScikitlearnExtraTreesClassifier,
    ScikitlearnGradientBoostingClassifier,
    ScikitlearnRandomForestClassifier,
    XGBoostClassifier,
]

CLASSIFIER_TYPE = Union[  # pylint: disable=C0103
    Classifier,
    BlackBoxClassifier,
    CatBoostARTClassifier,
    DetectorClassifier,
    EnsembleClassifier,
    GPyGaussianProcessClassifier,
    KerasClassifier,
    # JaxClassifier,
    LightGBMClassifier,
    MXClassifier,
    PyTorchClassifier,
    ScikitlearnClassifier,
    ScikitlearnDecisionTreeClassifier,
    ScikitlearnExtraTreeClassifier,
    ScikitlearnAdaBoostClassifier,
    ScikitlearnBaggingClassifier,
    ScikitlearnExtraTreesClassifier,
    ScikitlearnGradientBoostingClassifier,
    ScikitlearnRandomForestClassifier,
    ScikitlearnLogisticRegression,
    ScikitlearnSVC,
    TensorFlowClassifier,
    TensorFlowV2Classifier,
    XGBoostClassifier,
    CLASSIFIER_NEURALNETWORK_TYPE,
]

GENERATOR_TYPE = Union[
    TensorFlowGenerator, TensorFlowV2Generator
]  # pylint: disable=C0103

REGRESSOR_TYPE = Union[
    ScikitlearnRegressor, ScikitlearnDecisionTreeRegressor
]  # pylint: disable=C0103
"""
OBJECT_DETECTOR_TYPE = Union[  # pylint: disable=C0103
    ObjectDetector,
    PyTorchObjectDetector,
    PyTorchFasterRCNN,
    TensorFlowFasterRCNN,
]

SPEECH_RECOGNIZER_TYPE = Union[  # pylint: disable=C0103
    PyTorchDeepSpeech,
    TensorFlowLingvoASR,
]

PYTORCH_ESTIMATOR_TYPE = Union[  # pylint: disable=C0103
    PyTorchClassifier,
    PyTorchDeepSpeech,
    PyTorchEstimator,
    PyTorchObjectDetector,
    PyTorchFasterRCNN,
]

TENSORFLOWV2_ESTIMATOR_TYPE = Union[  # pylint: disable=C0103
    TensorFlowV2Classifier,
    TensorFlowV2Estimator,
]

ESTIMATOR_TYPE = Union[  # pylint: disable=C0103
    CLASSIFIER_TYPE, REGRESSOR_TYPE, OBJECT_DETECTOR_TYPE, SPEECH_RECOGNIZER_TYPE
]
"""
# ABLATOR_TYPE = Union[BlockAblator, ColumnAblator]  # pylint: disable=C0103

# CERTIFIER_TYPE = Union[PytorchDeepZ]  # pylint: disable=C0103
