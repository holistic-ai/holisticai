"""
The :mod:holisticai.privacy.metrics module includes attacks and privacy metrics
"""

from ._BlackBoxAttack import BlackBoxAttack, append_if_not_empty
from ._metrics import k_anonymity, l_diversity
