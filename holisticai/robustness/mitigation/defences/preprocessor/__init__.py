"""
Module implementing preprocessing defences against adversarial attacks.
"""
from holisticai.robustness.mitigation.defences.preprocessor.feature_squeezing import (
    FeatureSqueezing,
)
from holisticai.robustness.mitigation.defences.preprocessor.gaussian_augmentation import (
    GaussianAugmentation,
)
from holisticai.robustness.mitigation.defences.preprocessor.inverse_gan import (
    DefenseGAN,
    InverseGAN,
)
from holisticai.robustness.mitigation.defences.preprocessor.jpeg_compression import (
    JpegCompression,
)
from holisticai.robustness.mitigation.defences.preprocessor.label_smoothing import (
    LabelSmoothing,
)
from holisticai.robustness.mitigation.defences.preprocessor.mp3_compression import (
    Mp3Compression,
)
from holisticai.robustness.mitigation.defences.preprocessor.mp3_compression_pytorch import (
    Mp3CompressionPyTorch,
)
from holisticai.robustness.mitigation.defences.preprocessor.pixel_defend import (
    PixelDefend,
)
from holisticai.robustness.mitigation.defences.preprocessor.preprocessor import (
    Preprocessor,
)
from holisticai.robustness.mitigation.defences.preprocessor.resample import Resample
from holisticai.robustness.mitigation.defences.preprocessor.spatial_smoothing import (
    SpatialSmoothing,
)
from holisticai.robustness.mitigation.defences.preprocessor.spatial_smoothing_pytorch import (
    SpatialSmoothingPyTorch,
)
from holisticai.robustness.mitigation.defences.preprocessor.spatial_smoothing_tensorflow import (
    SpatialSmoothingTensorFlowV2,
)
from holisticai.robustness.mitigation.defences.preprocessor.thermometer_encoding import (
    ThermometerEncoding,
)
from holisticai.robustness.mitigation.defences.preprocessor.variance_minimization import (
    TotalVarMin,
)
from holisticai.robustness.mitigation.defences.preprocessor.video_compression import (
    VideoCompression,
)
from holisticai.robustness.mitigation.defences.preprocessor.video_compression_pytorch import (
    VideoCompressionPyTorch,
)
