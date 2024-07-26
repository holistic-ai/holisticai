import numpy as np
import pandas as pd


def cp_mat(y_true, y_pred, n_classes):
    """Returns the matrix of conditional probabilities y_pred | y_true"""
    tab = pd.crosstab(y_true, y_pred)
    cols = list({str(int(x)) for x in range(n_classes)} - {str(int(x)) for x in tab.columns})
    if cols != []:
        tab[cols] = 0
    tab = tab.values
    probs = tab.transpose() / tab.sum(axis=1)
    return probs.transpose()


def p_vec(y, flatten=True):
    """Returns the matrix of probabilities for the levels y"""
    tab = pd.crosstab(y, "count").values
    out = tab / tab.sum()
    if flatten:
        out = out.flatten()
    return out


def pars_to_cpmat(opt, n_groups=3, n_classes=3):
    """Reshapes the LP parameters as an n_group * n_class * n_class array"""
    shaped = np.reshape(opt.x, (n_groups, n_classes, n_classes))
    flipped = np.array([m.T for m in shaped])
    return flipped
