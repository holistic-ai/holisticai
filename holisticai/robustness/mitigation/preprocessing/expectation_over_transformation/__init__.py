"""
Module providing expectation over transformations.
"""
from holisticai.robustness.preprocessing.expectation_over_transformation.image_center_crop.pytorch import (
    EoTImageCenterCropPyTorch,
)
from holisticai.robustness.preprocessing.expectation_over_transformation.image_rotation.pytorch import (
    EoTImageRotationPyTorch,
)
from holisticai.robustness.preprocessing.expectation_over_transformation.image_rotation.tensorflow import (
    EoTImageRotationTensorFlow,
)
from holisticai.robustness.preprocessing.expectation_over_transformation.natural_corruptions.brightness.pytorch import (
    EoTBrightnessPyTorch,
)
from holisticai.robustness.preprocessing.expectation_over_transformation.natural_corruptions.brightness.tensorflow import (
    EoTBrightnessTensorFlow,
)
from holisticai.robustness.preprocessing.expectation_over_transformation.natural_corruptions.contrast.pytorch import (
    EoTContrastPyTorch,
)
from holisticai.robustness.preprocessing.expectation_over_transformation.natural_corruptions.contrast.tensorflow import (
    EoTContrastTensorFlow,
)
from holisticai.robustness.preprocessing.expectation_over_transformation.natural_corruptions.gaussian_noise.pytorch import (
    EoTGaussianNoisePyTorch,
)
from holisticai.robustness.preprocessing.expectation_over_transformation.natural_corruptions.gaussian_noise.tensorflow import (
    EoTGaussianNoiseTensorFlow,
)
from holisticai.robustness.preprocessing.expectation_over_transformation.natural_corruptions.shot_noise.pytorch import (
    EoTShotNoisePyTorch,
)
from holisticai.robustness.preprocessing.expectation_over_transformation.natural_corruptions.shot_noise.tensorflow import (
    EoTShotNoiseTensorFlow,
)
from holisticai.robustness.preprocessing.expectation_over_transformation.natural_corruptions.zoom_blur.pytorch import (
    EoTZoomBlurPyTorch,
)
from holisticai.robustness.preprocessing.expectation_over_transformation.natural_corruptions.zoom_blur.tensorflow import (
    EoTZoomBlurTensorFlow,
)
