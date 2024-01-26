"""
Module implementing varying metrics for assessing model robustness. These fall mainly under two categories:
attack-dependent and attack-independent.
"""
import logging

# from __future__ import absolute_import, division, print_function, unicode_literals
# from __future__ import annotations
from functools import reduce
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import numpy.linalg as la
import pandas as pd
from scipy.optimize import fmin as scipy_optimizer
from scipy.stats import weibull_min
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

from holisticai.pipeline import Pipeline
from holisticai.robustness.metrics.verification_decisions_trees import (
    RobustnessVerificationTreeModelsCliqueMethod,
)
from holisticai.robustness.mitigation.attacks.attack import EvasionAttack
from holisticai.robustness.mitigation.attacks.evasion.auto_attack import AutoAttack
from holisticai.robustness.mitigation.attacks.evasion.fast_gradient import (
    FastGradientMethod,
)
from holisticai.robustness.mitigation.attacks.evasion.hop_skip_jump import HopSkipJump
from holisticai.robustness.mitigation.config import ART_NUMPY_DTYPE
from holisticai.robustness.mitigation.utils.formatting import random_sphere
from holisticai.robustness.utils import is_instance_valid
from holisticai.wrappers.classification.classifier import (
    ClassGradientsMixin,
    ClassifierMixin,
)

# if TYPE_CHECKING:
from holisticai.wrappers.formatting import (
    CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE,
    CLASSIFIER_LOSS_GRADIENTS_TYPE,
    CLASSIFIER_TYPE,
)

logger = logging.getLogger(__name__)

SUPPORTED_METHODS: Dict[str, Dict[str, Any]] = {
    "auto": {
        "class": AutoAttack,
        "params": {"eps_step": 0.1, "eps_max": 1.0},
    },
    "fgsm": {
        "class": FastGradientMethod,
        "params": {"eps_step": 0.1, "eps_max": 1.0, "clip_min": 0.0, "clip_max": 1.0},
    },
    "hsj": {
        "class": HopSkipJump,
        "params": {
            "max_iter": 50,
            "max_eval": 10000,
            "init_eval": 100,
            "init_size": 100,
        },
    },
}


def get_crafter(
    classifier: "CLASSIFIER_TYPE", attack: str, params: Optional[Dict[str, Any]] = None
) -> "EvasionAttack":
    """
    Create an attack instance to craft adversarial samples.

    :param classifier: A trained model.
    :param attack: adversarial attack name.
    :param params: Parameters specific to the adversarial attack.
    :return: An attack instance.
    """
    try:
        crafter = SUPPORTED_METHODS[attack]["class"](classifier)
    except Exception:  # pragma: no cover
        logger.warning(
            "Available attacks include %s.", ", ".join(SUPPORTED_METHODS.keys())
        )
        raise NotImplementedError(
            f"{attack} crafting method not supported."
        ) from Exception

    if "params" in SUPPORTED_METHODS[attack]:
        crafter.set_params(**SUPPORTED_METHODS[attack]["params"])
    if params:
        crafter.set_params(**params)

    return crafter


def preprocess_pipeline(pipeline, attack_crafter=None):
    from holisticai.wrappers.classification.pytorch import PyTorchClassifier
    from holisticai.wrappers.classification.scikitlearn import (
        ScikitlearnClassifier,
        ScikitlearnDecisionTreeClassifier,
    )

    return pipeline

    if type(pipeline.estimator_hdl.estimator.obj) is PyTorchClassifier:
        #'loss', 'input_shape', and 'nb_classes'
        # model = pipeline.estimator_hdl.estimator.obj
        # return PyTorchClassifier(pipeline, model.loss, model.input_shape, model.nb_classes)
        return pipeline

    if (
        (attack_crafter is not None)
        and ScikitlearnDecisionTreeClassifier in attack_crafter._estimator_requirements
    ):
        return ScikitlearnDecisionTreeClassifier(pipeline)

    else:
        return ScikitlearnClassifier(pipeline)


def adversarial_accuracy(
    x: np.ndarray,
    y: np.ndarray = None,
    attack_name: str = None,
    attack_params: Optional[Dict[str, Any]] = None,
    attack_crafter: EvasionAttack = None,
    classifier: CLASSIFIER_TYPE = None,
) -> float:
    """
    Compute the adversarial accuracy of a classifier object over the sample `x` for a given adversarial crafting
    method `attack`, with optional true labels `y`. `attack_name` can be specified
    to use a preset attack, or `attack_crafter` for wider choices
    and customized parameters.
    Note that if `y` is not specified, the score does not exclude wrong predictions by the classifier.

    :param classifier: A trained model.
    :param x: Input samples of shape that can be fed into `classifier`.
    :param y: True labels of `x`.
    :param attack_name: A string specifying the attack to be used as a key to `art.metrics.metrics.SUPPORTED_METHODS`.
    :param attack_params: A dictionary with attack-specific parameters. If the attack has a norm attribute, then it will
                          be used as the norm for calculating the robustness; otherwise the standard Euclidean distance
                          is used (norm=2).
    :param attack_crafter: EvasionAttack instance with `generate' method to apply on `x` to create adversarial examples.
    :return: The adversarial accuracy of the classifier computed on `x`.
    """

    if type(classifier) is Pipeline:
        classifier = preprocess_pipeline(classifier, attack_crafter)

    attack_crafter.initialize(classifier)

    adv_x = attack_crafter.generate(x)

    # Predict the labels for adversarial examples
    y_orig = classifier.predict(x)
    y_pred = classifier.predict(adv_x)

    if y is None:
        idxs = y_pred == y_orig
        return np.sum(idxs) / len(y_orig)

    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)
    y_corr = y_orig == y
    return np.sum((y_pred == y_orig) & y_corr) / np.sum(y_corr)


def empirical_robustness(
    classifier: CLASSIFIER_TYPE,
    x: np.ndarray,
    attack_crafter: EvasionAttack = None,
) -> Union[float, np.ndarray]:
    """
    Compute the Empirical Robustness of a classifier object over the sample `x` for a given adversarial crafting
    method `attack`. This is equivalent to computing the minimal perturbation that the attacker must introduce for a
    successful attack.

    | Paper link: https://arxiv.org/abs/1511.04599

    :param classifier: A trained model.
    :param x: Data sample of shape that can be fed into `classifier`.
    :param attack_name: A string specifying the attack to be used as a key to `art.metrics.metrics.SUPPORTED_METHODS`.
    :param attack_params: A dictionary with attack-specific parameters. If the attack has a norm attribute, then it will
                          be used as the norm for calculating the robustness; otherwise the standard Euclidean distance
                          is used (norm=2).
    :return: The average empirical robustness computed on `x`.
    """
    if type(classifier) is Pipeline:
        classifier = preprocess_pipeline(classifier, attack_crafter)

    attack_crafter.initialize(classifier)

    attack_crafter.set_params(**{"minimal": True})
    adv_x = attack_crafter.generate(x)

    # Predict the labels for adversarial examples
    y = classifier.predict(x)
    y_pred = classifier.predict(adv_x)

    idxs = y_pred != y
    if np.sum(idxs) == 0.0:
        return 0.0

    norm_type = 2
    if hasattr(attack_crafter, "norm"):
        norm_type = attack_crafter.norm  # type: ignore
    perts_norm = la.norm((adv_x - x).reshape(x.shape[0], -1), ord=norm_type, axis=1)
    perts_norm = perts_norm[idxs]

    return np.mean(
        perts_norm / la.norm(x[idxs].reshape(np.sum(idxs), -1), ord=norm_type, axis=1)
    )


def loss_sensitivity(
    classifier: "CLASSIFIER_LOSS_GRADIENTS_TYPE", x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Local loss sensitivity estimated through the gradients of the prediction at points in `x`.

    | Paper link: https://arxiv.org/abs/1706.05394

    :param classifier: A trained model.
    :param x: Data sample of shape that can be fed into `classifier`.
    :param y: Labels for sample `x`, one-hot encoded.
    :return: The average loss sensitivity of the model.
    """
    grads = classifier.loss_gradient(x, y)
    norm = la.norm(grads.reshape(grads.shape[0], -1), ord=2, axis=1)

    return np.mean(norm)


def clever(
    classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
    x: np.ndarray,
    nb_batches: int,
    batch_size: int,
    radius: float,
    norm: int,
    target: Union[int, List[int], None] = None,
    target_sort: bool = False,
    c_init: float = 1.0,
    pool_factor: int = 10,
    verbose: bool = True,
) -> Optional[np.ndarray]:
    """
    Compute CLEVER score for an untargeted attack.

    | Paper link: https://arxiv.org/abs/1801.10578

    :param classifier: A trained model.
    :param x: One input sample.
    :param nb_batches: Number of repetitions of the estimate.
    :param batch_size: Number of random examples to sample per batch.
    :param radius: Radius of the maximum perturbation.
    :param norm: Current support: 1, 2, np.inf.
    :param target: Class or classes to target. If `None`, targets all classes.
    :param target_sort: Should the target classes be sorted in prediction order. When `True` and `target` is `None`,
           sort results.
    :param c_init: initialization of Weibull distribution.
    :param pool_factor: The factor to create a pool of random samples with size pool_factor x n_s.
    :param verbose: Show progress bars.
    :return: CLEVER score.
    """
    # Find the predicted class first
    if type(classifier) is Pipeline:
        from holisticai.wrappers.classification.scikitlearn import ScikitlearnClassifier

        classifier = ScikitlearnClassifier(classifier)

    y_pred = classifier.predict(np.array([x]))[0]
    pred_class = np.argmax(y_pred, axis=1)[0]
    if target is None:
        # Get a list of untargeted classes
        if target_sort:
            target_classes = np.argsort(y_pred)[0][:-1]
        else:
            target_classes = [
                i for i in range(classifier.nb_classes) if i != pred_class
            ]
    elif isinstance(target, (int, np.integer)):
        target_classes = [target]
    else:
        # Assume it's iterable
        target_classes = target
    score_list: List[Optional[float]] = []
    for j in tqdm(target_classes, desc="CLEVER untargeted", disable=not verbose):
        if j == pred_class:
            score_list.append(None)
            continue
        score = clever_targeted(
            classifier, x, j, nb_batches, batch_size, radius, norm, c_init, pool_factor
        )
        score_list.append(score)
    return np.array(score_list)


def clever_untargeted(
    classifier: CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE,
    x: np.ndarray,
    nb_batches: int,
    batch_size: int,
    radius: float,
    norm: int,
    c_init: float = 1.0,
    pool_factor: int = 10,
    verbose: bool = True,
) -> float:
    """
    Compute CLEVER score for an untargeted attack.

    | Paper link: https://arxiv.org/abs/1801.10578

    :param classifier: A trained model.
    :param x: One input sample.
    :param nb_batches: Number of repetitions of the estimate.
    :param batch_size: Number of random examples to sample per batch.
    :param radius: Radius of the maximum perturbation.
    :param norm: Current support: 1, 2, np.inf.
    :param c_init: initialization of Weibull distribution.
    :param pool_factor: The factor to create a pool of random samples with size pool_factor x n_s.
    :param verbose: Show progress bars.
    :return: CLEVER score.
    """
    if type(classifier) is Pipeline:
        # classifier = preprocess_pipeline_with_gradient(classifier)
        classifier = classifier

    # Get a list of untargeted classes
    pred_class = classifier.predict(np.array([x]))[0]
    untarget_classes = [i for i in range(classifier.nb_classes) if i != pred_class]

    # Compute CLEVER score for each untargeted class
    score_list = []
    for j in tqdm(untarget_classes, desc="CLEVER untargeted", disable=not verbose):
        score = clever_targeted(
            classifier, x, j, nb_batches, batch_size, radius, norm, c_init, pool_factor
        )
        score_list.append(score)

    return np.min(score_list)


def clever_targeted(
    classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
    x: np.ndarray,
    target_class: int,
    nb_batches: int,
    batch_size: int,
    radius: float,
    norm: float,
    c_init: float = 1.0,
    pool_factor: int = 10,
) -> float:
    """
    Compute CLEVER score for a targeted attack.

    | Paper link: https://arxiv.org/abs/1801.10578

    :param classifier: A trained model.
    :param x: One input sample.
    :param target_class: Targeted class.
    :param nb_batches: Number of repetitions of the estimate.
    :param batch_size: Number of random examples to sample per batch.
    :param radius: Radius of the maximum perturbation.
    :param norm: Current support: 1, 2, np.inf.
    :param c_init: Initialization of Weibull distribution.
    :param pool_factor: The factor to create a pool of random samples with size pool_factor x n_s.
    :return: CLEVER score.
    """
    # Check if the targeted class is different from the predicted class
    pred_class = int(classifier.predict(np.array([x]))[0])

    if target_class == pred_class:  # pragma: no cover
        raise ValueError("The targeted class is the predicted class.")

    # Check if pool_factor is smaller than 1
    if pool_factor < 1:  # pragma: no cover
        raise ValueError("The `pool_factor` must be larger than 1.")

    # Some auxiliary vars
    rand_pool_grad_set = []
    grad_norm_set = []
    dim = reduce(lambda x_, y: x_ * y, x.shape, 1)
    shape = [pool_factor * batch_size]
    shape.extend(x.shape)

    # Generate a pool of samples
    rand_pool = np.reshape(
        random_sphere(
            nb_points=pool_factor * batch_size, nb_dims=dim, radius=radius, norm=norm
        ),
        shape,
    )
    rand_pool += np.repeat(np.array([x]), pool_factor * batch_size, 0)
    rand_pool = rand_pool.astype(ART_NUMPY_DTYPE)
    if hasattr(classifier, "clip_values") and classifier.clip_values is not None:
        np.clip(
            rand_pool,
            classifier.clip_values[0],
            classifier.clip_values[1],
            out=rand_pool,
        )

    # Change norm since q = p / (p-1)
    if norm == 1:
        norm = np.inf
    elif norm == np.inf:
        norm = 1
    elif norm != 2:  # pragma: no cover
        raise ValueError(f"Norm {norm} not supported")

    # Compute gradients for all samples in rand_pool
    for i in range(batch_size):
        rand_pool_batch = rand_pool[i * pool_factor : (i + 1) * pool_factor]

        # Compute gradients
        grad_pred_class = classifier.class_gradient(rand_pool_batch, label=pred_class)
        grad_target_class = classifier.class_gradient(
            rand_pool_batch, label=target_class
        )

        if (
            np.isnan(grad_pred_class).any() or np.isnan(grad_target_class).any()
        ):  # pragma: no cover
            raise Exception("The classifier results NaN gradients.")

        grad = grad_pred_class - grad_target_class
        grad = np.reshape(grad, (pool_factor, -1))
        grad = np.linalg.norm(grad, ord=norm, axis=1)
        rand_pool_grad_set.extend(grad)

    rand_pool_grads = np.array(rand_pool_grad_set)

    # Loop over the batches
    for _ in range(nb_batches):
        # Random selection of gradients
        grad_norm = rand_pool_grads[
            np.random.choice(pool_factor * batch_size, batch_size)
        ]
        grad_norm = np.max(grad_norm)
        grad_norm_set.append(grad_norm)

    # Maximum likelihood estimation for max gradient norms
    [_, loc, _] = weibull_min.fit(
        -np.array(grad_norm_set), c_init, optimizer=scipy_optimizer
    )

    # Compute function value
    values = classifier.predict_proba(np.array([x]))
    value = values[:, pred_class] - values[:, target_class]

    # Compute scores
    score = np.min([-value[0] / loc, radius])

    return score


def wasserstein_distance(
    u_values: np.ndarray,
    v_values: np.ndarray,
    u_weights: Optional[np.ndarray] = None,
    v_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute the first Wasserstein distance between two 1D distributions.

    :param u_values: Values of first distribution with shape (nb_samples, feature_dim_1, ..., feature_dim_n).
    :param v_values: Values of second distribution with shape (nb_samples, feature_dim_1, ..., feature_dim_n).
    :param u_weights: Weight for each value. If None, equal weights will be used.
    :param v_weights: Weight for each value. If None, equal weights will be used.
    :return: The Wasserstein distance between the two distributions.
    """
    import scipy

    assert u_values.shape == v_values.shape
    if u_weights is not None:
        assert v_weights is not None
    if u_weights is None:
        assert v_weights is None
    if u_weights is not None and v_weights is not None:
        assert u_weights.shape == v_weights.shape
    if u_weights is not None:
        assert u_values.shape[0] == u_weights.shape[0]

    u_values = u_values.flatten().reshape(u_values.shape[0], -1)
    v_values = v_values.flatten().reshape(v_values.shape[0], -1)

    if u_weights is not None and v_weights is not None:
        u_weights = u_weights.flatten().reshape(u_weights.shape[0], -1)
        v_weights = v_weights.flatten().reshape(v_weights.shape[0], -1)

    w_d = np.zeros(u_values.shape[0])

    for i in range(u_values.shape[0]):
        if u_weights is None and v_weights is None:
            w_d[i] = scipy.stats.wasserstein_distance(u_values[i], v_values[i])
        elif u_weights is not None and v_weights is not None:
            w_d[i] = scipy.stats.wasserstein_distance(
                u_values[i], v_values[i], u_weights[i], v_weights[i]
            )

    return w_d


ATTACK_CATALOG = {"HopSkipJump": HopSkipJump}


def check_classifiers_allowed(fn):
    return fn.__annotations__["classifier"].__args__


def classification_robustness_metrics(
    classifier,
    x,
    y,
    attack_crafter=None,
    attack_name: Optional[str] = None,
    attack_params: Optional[dict] = None,
):

    y_pred = classifier.predict(x)

    attack_crafter = attack_crafter
    if attack_name in ATTACK_CATALOG:
        attack_crafter = ATTACK_CATALOG[attack_name](**attack_params)

    base_acc = accuracy_score(y, y_pred)
    results = {"Base Train Accuracy": base_acc}

    if is_instance_valid(
        classifier,
        check_classifiers_allowed(
            RobustnessVerificationTreeModelsCliqueMethod.__init__
        ),
    ):
        rt = RobustnessVerificationTreeModelsCliqueMethod(classifier)
        average_bound, verified_error = rt.verify(
            x=x, y=y, eps_init=0.3, nb_search_steps=10, max_clique=2, max_level=2
        )

        results.update({"Tree Model: Verification - Average bound": average_bound})
        results.update({"Tree Model: Verification - Error at eps": verified_error})
    else:
        results.update({"Tree Model: Verification - Average bound": np.nan})
        results.update({"Tree Model: Verification - Error at eps": np.nan})

    if attack_crafter is not None:
        if is_instance_valid(
            classifier, check_classifiers_allowed(adversarial_accuracy)
        ):
            acc = adversarial_accuracy(
                x=x, y=y, attack_crafter=attack_crafter, classifier=classifier
            )
            results.update({"Adverarial Accuracy": acc})

        if is_instance_valid(
            classifier, check_classifiers_allowed(empirical_robustness)
        ):
            er = empirical_robustness(
                x=x, attack_crafter=attack_crafter, classifier=classifier
            )
            results.update({"Empirical Robustness": er})

    if is_instance_valid(classifier, check_classifiers_allowed(clever_untargeted)):
        clever = clever_untargeted(
            x=x[0],
            classifier=classifier,
            nb_batches=10,
            batch_size=1,
            radius=2,
            norm=1,
            c_init=1.0,
            pool_factor=10,
        )
        results.update({"CLEVER (untargeted)": clever})
    else:
        results.update({"CLEVER (untargeted)": np.nan})

    results = pd.DataFrame.from_dict(results, orient="index")
    results.columns = ["Values"]
    return results


# def nearest_neighbour_dist(classifier, x, x_ref, attack_name, attack_params=None):
#     """
#     Compute the (average) nearest neighbour distance between the sets `x` and `x_train`: for each point in `x`,
#     measure the Euclidean distance to its closest point in `x_train`, then average over all points.
#
#     :param classifier: A trained model
#     :type classifier: :class:`.Classifier`
#     :param x: Data sample of shape that can be fed into `classifier`
#     :type x: `np.ndarray`
#     :param x_ref: Reference data sample to be considered as neighbors
#     :type x_ref: `np.ndarray`
#     :param attack_name: adversarial attack name
#     :type attack_name: `str`
#     :param attack_params: Parameters specific to the adversarial attack
#     :type attack_params: `dict`
#     :return: The average nearest neighbors distance
#     :rtype: `float`
#     """
#     # Craft the adversarial examples
#     crafter = get_crafter(classifier, attack_name, attack_params)
#     adv_x = crafter.generate(x, minimal=True)
#
#     # Predict the labels for adversarial examples
#     y = classifier.predict(x)
#     y_pred = classifier.predict(adv_x)
#
#     adv_x_ = adv_x.reshape(adv_x.shape[0], np.prod(adv_x.shape[1:]))
#     x_ = x_ref.reshape(x_ref.shape[0], np.prod(x_ref.shape[1:]))
#     dists = la.norm(adv_x_ - x_, axis=1)
#
#     # TODO check if following computation is correct ?
#     dists = np.min(dists, axis=1) / la.norm(x.reshape(x.shape[0], -1), ord=2, axis=1)
#     idxs = (np.argmax(y_pred, axis=1) != np.argmax(y, axis=1))
#     avg_nn_dist = np.mean(dists[idxs])
#
#     return avg_nn_dist
