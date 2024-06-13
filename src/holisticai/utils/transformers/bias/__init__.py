from collections import namedtuple
from holisticai.utils.transformers.bias._inprocessing import BMInprocessing
from holisticai.utils.transformers.bias._postprocessing import BMPostprocessing
from holisticai.utils.transformers.bias._preprocessing import BMPreprocessing
from holisticai.utils.transformers.bias._group_utils import SensitiveGroups

BiasMitigationTags = namedtuple("BiasMitigationTags", ["PRE", "INP", "POST"])
BIAS_TAGS = BiasMitigationTags(PRE="bm_pre", INP="bm_inp", POST="bm_pos")

__all__ = ["BMInprocessing", "BMPostprocessing", "BMPreprocessing", "SensitiveGroups"]
