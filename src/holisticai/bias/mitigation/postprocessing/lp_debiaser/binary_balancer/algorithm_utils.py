import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


class CLFRates:
    def __init__(self, y_true, y_pred, round=4):  # noqa: A002
        self.tab = confusion_matrix(y_true, y_pred)
        tn = self.tab[0, 0]
        fn = self.tab[1, 0]
        fp = self.tab[0, 1]
        tp = self.tab[1, 1]
        self.pr = np.round((tp + fp) / len(y_true), round)
        self.nr = np.round((tn + fn) / len(y_true), round)
        self.tnr = np.round(tn / (tn + fp), round)
        self.tpr = np.round(tp / (tp + fn), round)
        self.fnr = np.round(fn / (fn + tp), round)
        self.fpr = np.round(fp / (fp + tn), round)
        self.acc = (tn + tp) / len(y_true)


def from_top(roc_point, round=4):  # noqa: A002, ARG001
    d = np.sqrt(roc_point[0] ** 2 + (roc_point[1] - 1) ** 2)
    return d


def loss_from_roc(y, probs, roc):
    points = [(roc[0][i], roc[1][i]) for i in range(len(roc[0]))]
    guess_list = [threshold(probs, t) for t in roc[2]]
    accs = [accuracy_score(y, g) for g in guess_list]
    js = [p[1] - p[0] for p in points]
    tops = [from_top(point) for point in points]
    return {"guesses": guess_list, "accs": accs, "js": js, "tops": tops}


def pred_from_pya(y_pred, p_attr, pya, binom=False):
    # Getting the groups and making the initially all-zero predictor
    groups = np.unique(p_attr)
    out = y_pred.copy()

    for i, g in enumerate(groups):
        group_ids = p_attr == g

        # Pulling the fitted switch probabilities for the group
        p = pya[i]

        # Indices in the group from which to choose swaps
        pos = group_ids & (y_pred == 1)
        neg = group_ids & (y_pred == 0)

        if not binom:
            # Randomly picking the positive predictions
            pos_samp = np.random.choice(a=np.flatnonzero(pos), size=int(p[1] * pos.sum()), replace=False)
            neg_samp = np.random.choice(a=np.flatnonzero(neg), size=int(p[0] * neg.sum()), replace=False)
            samp = np.concatenate((pos_samp, neg_samp)).flatten()
            out[samp] = 1
            out[group_ids & ~np.isin(np.arange(len(y_pred)), samp)] = 0

    return out.astype(np.uint8)


# Quick function for thresholding probabilities
def threshold(probs, cutoff=0.5):
    return np.array(probs >= cutoff).astype(np.uint8)
