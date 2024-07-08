import math

import numpy as np
from joblib import Parallel, delayed
from numpy.linalg import inv


def calculate_popularity_model(ratings):
    propensity_score = [float(np.count_nonzero(ratings[:, i])) / ratings.shape[0] for i in range(ratings.shape[1])]
    temp = np.array(propensity_score)
    temp = temp.reshape((1, len(temp)))
    return temp


def erros(P_s, Q_s, R_s, W_s, lamda, beta):
    eR_s = np.sum(P_s * Q_s.T, axis=1)
    et = np.sum(pow(R_s - eR_s, 2))
    e_s = (lamda / 2) * (pow(P_s, 2) + pow(Q_s.T, 2)) + (beta / 2) * (
        pow((pow(P_s, 2) + pow(Q_s.T, 2)), 2) * W_s[:, None]
    )
    countt = len(e_s)
    e = et + np.sum(e_s)
    err = math.sqrt(e / countt)
    errt = math.sqrt(et / countt)
    return err, errt


def compute_mat(qr, ivp):
    qr_ = qr[:, :, None]
    qr_t = qr[:, None, :]
    ivp_ = ivp.reshape(-1, 1, 1)
    return np.sum(ivp_ * np.matmul(qr_, qr_t), axis=0)


def compute_vect(qr, ivp, r):
    temp = (ivp * r).reshape(-1, 1)
    return np.sum(temp * qr, axis=0)


def update_emb(b, R_, E_, invprop_):
    mat_temp = compute_mat(E_, invprop_)
    vect_temp = compute_vect(E_, invprop_, R_)
    return np.dot(inv(mat_temp + b), vect_temp)


def updateP(P, b, R, Q, u_rated_items, invprop):  # noqa: N802
    P = Parallel(n_jobs=-1, verbose=0)(
        delayed(update_emb)(b, R[u, rated_items], Q[:, rated_items].T, invprop[u, rated_items])
        for u, rated_items in enumerate(u_rated_items)
    )
    return np.stack(P, axis=0)


def updateQ(P, b, R, Q, i_rated_users, invprop):  # noqa: N802
    Q = Parallel(n_jobs=-1, verbose=0)(
        delayed(update_emb)(b, R[rated_users, i], P[rated_users, :], invprop[rated_users, i])
        for i, rated_users in enumerate(i_rated_users)
    )
    return np.stack(Q, axis=1)
