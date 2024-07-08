import numpy as np


def f_mincon(etaX, S, M, L, beta, lambda_):
    S = np.array(S).astype(np.int32)
    p = np.array([np.mean(S == 0), np.mean(S == 1)])
    group_values = np.unique(S)
    assert len(group_values) == 2

    def h_fn(E, S, i):
        return 2 * p[S] * i * M * E / L - p[S] * (i**2) * (M**2) / (L**2)

    def tot_fn(H, S, i):
        return np.exp(lambda_[i + L] * (2 * S - 1) / beta + H / beta)

    # Vectorized calculations
    indices = np.arange(-L, L + 1)
    Hs = h_fn(etaX, S, indices[:, None])
    TOTs = tot_fn(Hs, S, indices[:, None])
    TOT = np.sum(TOTs, axis=0)

    Hs = np.array(Hs)
    TOTs = np.array(TOTs)
    i_values = np.arange(2 * L + 1)
    H_i = Hs[i_values]
    tmp1 = TOTs[i_values] / TOT
    RISs = tmp1 * ((2 * S[None, :] - 1) * lambda_[i_values, None] + H_i - beta * np.log((2 * L + 1) * tmp1))
    RIS = np.sum(RISs, axis=0)

    ris = np.mean(RIS[S == 0]) + np.mean(RIS[S == 1])

    return ris


def f_lambda(Y, S, M, L, beta):
    from functools import partial

    from scipy.optimize import minimize

    lambda_ = 0.99 * np.ones(shape=(2 * L + 1,))
    fun = partial(f_mincon, Y, S, M, L, beta)
    lambda_ = minimize(fun, lambda_, bounds=[(0, 4 * M)] * len(lambda_)).x  # , method='L-BFGS-B').x

    return lambda_
