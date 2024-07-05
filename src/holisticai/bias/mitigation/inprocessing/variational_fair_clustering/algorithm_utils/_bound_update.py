import numpy as np


def normalize(S_in):
    maxcol = np.max(S_in, axis=1, keepdims=True)
    S_in = S_in - maxcol
    S_out = np.exp(S_in)
    S_out_sum = np.sum(S_out, axis=1, keepdims=True)
    S_out = S_out / S_out_sum

    return S_out


def normalize_2(S_in):
    S_in_sum = np.sum(S_in, axis=1, keepdims=True)
    S_in = S_in / S_in_sum
    return S_in


def bound_energy(S, S_in, a_term, b_term):
    E = np.nansum(S * np.log(np.maximum(S, 1e-15)) - S * np.log(np.maximum(S_in, 1e-15)) + a_term * S + b_term * S)
    return E


def get_S_discrete(L, N, K):  # noqa: N802
    x = range(N)
    temp = np.zeros((N, K), dtype=float)
    temp[(x, L)] = 1
    return temp


def compute_b(S, groups_ids, group_prob, F_a):
    R = (group_prob.values[:, None] / np.sum(S, axis=0, keepdims=True)).T

    V = groups_ids.values[:, None, :]

    F_b = np.maximum(np.sum(V * S[:, :, None], axis=0, keepdims=True), 1e-15)

    F_a_b = F_a[:, None, :] / F_b

    return np.sum(R[None, :, :] - F_a_b, axis=-1)


class BoundUpdate:
    def __init__(self, bound_lambda, L, bound_iteration):
        self.bound_lambda = bound_lambda
        self.L = L
        self.bound_iteration = bound_iteration

    def transform(self, a_p, group_prob, groups_ids):
        oldE = float("inf")
        S = np.exp(-a_p)
        S = normalize_2(S)
        a_term = -a_p.copy()

        F_a = group_prob.values[None, :] * groups_ids.values

        for i in range(self.bound_iteration):
            b = compute_b(S, groups_ids, group_prob, F_a)
            b_term = self.bound_lambda * b
            terms = (a_term - b_term) / self.L
            S_in_2 = normalize(terms)
            S_in = S.copy()
            S = S * S_in_2
            S = normalize_2(S)

            E = bound_energy(S, S_in, a_p, b_term)

            report_E = E

            if i > 1 and (abs(E - oldE) <= 1e-5 * abs(oldE)):
                break

            oldE = E
            report_E = E

        L = np.argmax(S, axis=1)

        return L, S, report_E
