from typing import Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, list[float], pd.Series]
MatrixLike = Union[np.ndarray, list[list[float]], pd.DataFrame]
