# # MIT License
# #
# # Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
# #
# # Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# # documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# # rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# # persons to whom the Software is furnished to do so, subject to the following conditions:
# #
# # The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# # Software.
# #
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# # WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# # TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.
# """
# Module providing convenience functions.
# """

from __future__ import annotations

from typing import Optional, Union

import numpy as np


def to_categorical(labels: Union[np.ndarray, list[float]], nb_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert an array of labels to binary class matrix.

    Parameters
    ----------
    labels : Union[np.ndarray, list[float]]
        An array of integer labels of shape `(nb_samples,)`.
    nb_classes : Optional[int]
        The number of classes (possible labels).

    Returns
    -------
    np.ndarray
        A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`.
    """
    labels = np.array(labels, dtype=int)
    if nb_classes is None:
        nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return categorical
