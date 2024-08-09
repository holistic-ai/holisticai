"""
This module implements the HopSkipJump attack `HopSkipJump`. This is a black-box attack that only requires class
predictions. It is an advanced version of the Boundary attack.

| Paper link: https://arxiv.org/abs/1904.02144
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from holisticai.robustness.attackers.classification.commons import x_array_to_df, x_to_nd_array


class HopSkipJump:
    """
    Implementation of the HopSkipJump attack from Jianbo et al. (2019). This is a powerful black-box attack that
    only requires final class prediction, and is an advanced version of the boundary attack.

    Parameters
    ----------
    name : str, optional
        The name of the attack.
    batch_size : int, optional
        Batch size for the attack.
    targeted : bool, optional
        Indicates whether the attack is targeted or not. If True, the positive ground truth is used as the target.
    norm : int, float, str, optional
        The norm of the attack. Possible values: "inf", np.inf or 2.
    max_iter : int, optional
        The maximum number of iterations.
    max_eval : int, optional
        The maximum number of evaluations.
    init_eval : int, optional
        The number of initial evaluations.
    init_size : int, optional
        The number of initial samples.
    verbose : bool, optional
        Verbosity mode.
    predictor : callable, optional
        The model's prediction function. The default is None.
    input_size : int, optional
        The size of the input data.
    theta : float, optional
        The binary search threshold.
    curr_iter : int, optional
        The current iteration.

    References
    ----------
    .. [1] Chen, J., Jordan, M. I., & Wainwright, M. J. (2019). HopSkipJumpAttack: A query-efficient decision-based attack. In 2020 ieee symposium on security and privacy (sp) (pp. 1277-1294). IEEE.
    """

    def __init__(
        self,
        name="HSJ",
        batch_size=64,
        targeted=False,
        norm=2,
        max_iter=50,
        max_eval=10000,
        init_eval=100,
        init_size=100,
        verbose=True,
        predictor=None,
        input_size=0,
        theta=0.0,
        curr_iter=0,
    ):
        self.name = name
        self.batch_size = batch_size
        self.targeted = targeted
        self.norm = norm
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.init_eval = init_eval
        self.init_size = init_size
        self.verbose = verbose
        self.predictor = predictor
        self.input_size = input_size
        self.theta = theta
        self.curr_iter = curr_iter

    def predict(self, x: np.ndarray):
        """
        Perform prediction on the input data.

        Parameters
        ----------
        x : np.ndarray
            The input data.

        Returns
        -------
        np.ndarray
            The model's prediction.
        """
        x_df = x_array_to_df(x, feature_names=self.feature_names)

        return np.array(self.predictor(x_df))

    def generate(
        self, x_df: pd.DataFrame, y: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None, x_adv_init=None
    ) -> pd.DataFrame:
        """
        Generate adversarial samples and return them in an array.

        Parameters
        ----------
        x_df : pd.DataFrame
            The input data.
        y : np.ndarray, optional
            The target labels.
        mask : np.ndarray, optional
            The mask used to select the sensitive features.
        x_adv_init : np.ndarray, optional
            Initial array to act as an initial adversarial example.

        Returns
        -------
        pd.DataFrame
            The adversarial examples.
        """

        self.input_shape = tuple(x_df.shape[1:])
        self.input_size = np.prod(self.input_shape)
        if self.norm == 2:
            self.theta = 0.01 / np.sqrt(self.input_size)
        else:
            self.theta = 0.01 / self.input_size

        self.feature_names = list(x_df.columns)
        x = x_to_nd_array(x_df)

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:  # pragma: no cover
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            y = self.predict(x)

        # Check whether users need a stateful attack
        start = 0

        # Check the mask
        if mask is not None:
            if len(mask.shape) != len(x.shape):
                mask = np.array([mask] * x.shape[0])
        else:
            mask = np.array([None] * x.shape[0])

        # Get clip_min and clip_max from the input data
        clip_min, clip_max = np.min(x), np.max(x)

        self._clip_min = clip_min
        self._clip_max = clip_max

        # Prediction from the original images
        preds = self.predict(x)

        # Prediction from the initial adversarial examples if not None
        if x_adv_init is not None:
            # Add mask param to the x_adv_init
            for i in range(x.shape[0]):
                if mask[i] is not None:
                    x_adv_init[i] = x_adv_init[i] * mask[i] + x[i] * (1 - mask[i])

            # Do prediction on the init
            init_preds = self.predict(x_adv_init)

        else:
            init_preds = [None] * len(x)
            x_adv_init = [None] * len(x)

        x_adv = x.copy()

        # Generate the adversarial samples
        for ind, val in enumerate(x_adv):
            self.curr_iter = start

            if self.targeted:
                x_adv[ind] = self._perturb(
                    x=val,
                    y=y[ind],  # type: ignore
                    y_p=preds[ind],
                    init_pred=init_preds[ind],  # type: ignore
                    adv_init=x_adv_init[ind],  # type: ignore
                    mask=mask[ind],
                )

            else:
                x_adv[ind] = self._perturb(
                    x=val,
                    y=-1,
                    y_p=preds[ind],
                    init_pred=init_preds[ind],  # type: ignore
                    adv_init=x_adv_init[ind],  # type: ignore
                    mask=mask[ind],
                )

        return x_array_to_df(x_adv, feature_names=self.feature_names)

    def _perturb(
        self,
        x: np.ndarray,
        y: int,
        y_p: int,
        init_pred: int,
        adv_init: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Internal attack function for one example.

        Parameters
        ----------
        x : np.ndarray
            The original input.
        y : int
            The target label.
        y_p : int
            The predicted label of x.
        init_pred : int
            The predicted label of the initial image.
        adv_init : np.ndarray
            Initial array to act as an initial adversarial example.
        mask : np.ndarray
            An array with a mask to be applied to the adversarial perturbations. Shape needs to be broadcastable to the\\
            shape of x. Any features for which the mask is zero will not be adversarially perturbed.

        Returns
        -------
        np.ndarray
            An adversarial example.
        """
        # First, create an initial adversarial sample
        initial_sample = self._init_sample(x, y, y_p, init_pred, adv_init, mask)

        # If an initial adversarial example is not found, then return the original image
        if initial_sample is None:
            return x

        # If an initial adversarial example found, then go with HopSkipJump attack
        x_adv = self._attack(initial_sample[0], x, initial_sample[1], mask)

        return x_adv

    def _init_sample(
        self,
        x: np.ndarray,
        y: int,
        y_p: int,
        init_pred: int,
        adv_init: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Optional[Union[np.ndarray, tuple[np.ndarray, int]]]:
        """
        Find initial adversarial example for the attack.

        Parameters
        ----------
        x : np.ndarray
            The original input.
        y : int
            The target label.
        y_p : int
            The predicted label of x.
        init_pred : int
            The predicted label of the initial image.
        adv_init : np.ndarray
            Initial array to act as an initial adversarial example.
        mask : np.ndarray
            An array with a mask to be applied to the adversarial perturbations. Shape needs to be broadcastable to the\\
            shape of x. Any features for which the mask is zero will not be adversarially perturbed.

        Returns
        -------
        Optional[Union[np.ndarray, tuple[np.ndarray, int]]]
            An initial adversarial example.
        """
        nprd = np.random.RandomState()
        initial_sample = None

        if self.targeted:
            # Attack satisfied
            if y == y_p:
                return None

            # Attack unsatisfied yet and the initial image satisfied
            if adv_init is not None and init_pred == y:
                return adv_init, init_pred

            # Attack unsatisfied yet and the initial image unsatisfied
            for _ in range(self.init_size):
                random_img = nprd.uniform(self._clip_min, self._clip_max, size=x.shape).astype(x.dtype)

                if mask is not None:
                    random_img = random_img * mask + x * (1 - mask)

                random_class = self.predict(np.array([random_img]))

                if random_class == y:
                    # Binary search to reduce the l2 distance to the original image
                    random_img = self._binary_search(
                        current_sample=random_img,
                        original_sample=x,
                        target=y,
                        norm=2,
                        threshold=0.001,
                    )
                    initial_sample = random_img, random_class

                    break

        else:
            # The initial image satisfied
            if adv_init is not None and init_pred != y_p:
                return adv_init, y_p

            # The initial image unsatisfied
            for _ in range(self.init_size):
                random_img = nprd.uniform(self._clip_min, self._clip_max, size=x.shape).astype(x.dtype)

                if mask is not None:
                    random_img = random_img * mask + x * (1 - mask)

                random_class = self.predict(np.array([random_img]))

                if random_class != y_p:
                    # Binary search to reduce the l2 distance to the original image
                    random_img = self._binary_search(
                        current_sample=random_img,
                        original_sample=x,
                        target=y_p,
                        norm=2,
                        threshold=0.001,
                    )
                    initial_sample = random_img, y_p

                    break

        return initial_sample  # type: ignore

    def _attack(
        self,
        initial_sample: np.ndarray,
        original_sample: np.ndarray,
        target: int,
        mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Main function for the boundary attack.

        Parameters
        ----------
        initial_sample : np.ndarray
            The initial adversarial example.
        original_sample : np.ndarray
            The original input.
        target : int
            The target label.
        mask : np.ndarray
            An array with a mask to be applied to the adversarial perturbations. Shape needs to be broadcastable to the\\
            shape of x. Any features for which the mask is zero will not be adversarially perturbed.

        Returns
        -------
        np.ndarray
            An adversarial example.
        """
        # Set current perturbed image to the initial image
        current_sample = initial_sample

        # Main loop to wander around the boundary
        for _ in range(self.max_iter):
            # First compute delta
            delta = self._compute_delta(
                current_sample=current_sample,
                original_sample=original_sample,
            )

            # Then run binary search
            current_sample = self._binary_search(
                current_sample=current_sample,
                original_sample=original_sample,
                norm=self.norm,
                target=target,
            )

            # Next compute the number of evaluations and compute the update
            num_eval = min(int(self.init_eval * np.sqrt(self.curr_iter + 1)), self.max_eval)

            update = self._compute_update(
                current_sample=current_sample,
                num_eval=num_eval,
                delta=delta,
                target=target,
                mask=mask,
            )

            # Finally run step size search by first computing epsilon
            if self.norm == 2:
                dist = np.linalg.norm(original_sample - current_sample)
            else:
                dist = np.max(abs(original_sample - current_sample))

            epsilon = 2.0 * dist / np.sqrt(self.curr_iter + 1)
            success = False

            while not success:
                epsilon /= 2.0
                potential_sample = current_sample + epsilon * update
                success = self._adversarial_satisfactory(  # type: ignore
                    samples=potential_sample[None],
                    target=target,
                )

            # Update current sample
            current_sample = np.clip(potential_sample, self._clip_min, self._clip_max)

            # Update current iteration
            self.curr_iter += 1

            # If attack failed. return original sample
            if np.isnan(current_sample).any():  # pragma: no cover
                return original_sample

        return current_sample

    def _binary_search(
        self,
        current_sample: np.ndarray,
        original_sample: np.ndarray,
        target: int,
        norm: Union[int, float, str],  # noqa: PYI041
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Binary search to approach the boundary.

        Parameters
        ----------
        current_sample : np.ndarray
            The current adversarial example.
        original_sample : np.ndarray
            The original input.
        target : int
            The target label.
        norm : Union[int, float, str]
            Order of the norm. Possible values: "inf", np.inf or 2.
        threshold : float, optional
            The threshold for the binary search.

        Returns
        -------
        np.ndarray
            An adversarial example.
        """
        # First set upper and lower bounds as well as the threshold for the binary search
        if norm == 2:
            (upper_bound, lower_bound) = (1, 0)

            if threshold is None:
                threshold = self.theta

        else:
            (upper_bound, lower_bound) = (
                np.max(abs(original_sample - current_sample)),
                0,
            )

            if threshold is None:
                threshold = np.minimum(upper_bound * self.theta, self.theta)

        # Then start the binary search
        while (upper_bound - lower_bound) > threshold:  # type: ignore
            # Interpolation point
            alpha = (upper_bound + lower_bound) / 2.0
            interpolated_sample = self._interpolate(
                current_sample=current_sample,
                original_sample=original_sample,
                alpha=float(alpha),
                norm=norm,
            )

            # Update upper_bound and lower_bound
            satisfied = self._adversarial_satisfactory(
                samples=interpolated_sample[None],
                target=target,
            )[0]
            lower_bound = np.where(satisfied == 0, alpha, lower_bound)
            upper_bound = np.where(satisfied == 1, alpha, upper_bound)

        result = self._interpolate(
            current_sample=current_sample,
            original_sample=original_sample,
            alpha=float(upper_bound),
            norm=norm,
        )

        return result

    def _compute_delta(
        self,
        current_sample: np.ndarray,
        original_sample: np.ndarray,
    ) -> float:
        """
        Compute the delta parameter.

        Parameters
        ----------
        current_sample : np.ndarray
            The current adversarial example.
        original_sample : np.ndarray
            The original input.

        Returns
        -------
        float
            The delta parameter.
        """
        # Note: This is a bit different from the original paper, instead we keep those that are
        # implemented in the original source code of the authors
        if self.curr_iter == 0:
            return 0.1 * (self._clip_max - self._clip_min)

        if self.norm == 2:
            dist = np.linalg.norm(original_sample - current_sample)
            delta = np.sqrt(np.prod(self.input_shape)) * self.theta * dist
        else:
            dist = np.max(abs(original_sample - current_sample))
            delta = np.prod(self.input_shape) * self.theta * dist

        return float(delta)

    def _compute_update(
        self,
        current_sample: np.ndarray,
        num_eval: int,
        delta: float,
        target: int,
        mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Compute the update in Eq.(14).

        Parameters
        ----------
        current_sample : np.ndarray
            The current adversarial example.
        num_eval : int
            The number of evaluations.
        delta : float
            The delta parameter.
        target : int
            The target label.
        mask : np.ndarray
            An array with a mask to be applied to the adversarial perturbations. Shape needs to be broadcastable to the\\
            shape of x. Any features for which the mask is zero will not be adversarially perturbed.

        Returns
        -------
        np.ndarray
            The updated perturbation.
        """
        # Generate random noise
        rnd_noise_shape = [num_eval, *self.input_shape]
        if self.norm == 2:
            rnd_noise = np.random.randn(*rnd_noise_shape)
        else:
            rnd_noise = np.random.uniform(low=-1, high=1, size=rnd_noise_shape)

        # With mask
        if mask is not None:
            rnd_noise = rnd_noise * mask

        # Normalize random noise to fit into the range of input data
        rnd_noise = rnd_noise / np.sqrt(
            np.sum(
                rnd_noise**2,
                axis=tuple(range(len(rnd_noise_shape)))[1:],
                keepdims=True,
            )
        )
        eval_samples = np.clip(current_sample + delta * rnd_noise, self._clip_min, self._clip_max)
        rnd_noise = (eval_samples - current_sample) / delta

        # Compute gradient: This is a bit different from the original paper, instead we keep those that are
        # implemented in the original source code of the authors
        satisfied = self._adversarial_satisfactory(samples=eval_samples, target=target)
        f_val = 2 * satisfied.reshape([num_eval] + [1] * len(self.input_shape)) - 1.0

        if np.mean(f_val) == 1.0:
            grad = np.mean(rnd_noise, axis=0)
        elif np.mean(f_val) == -1.0:
            grad = -np.mean(rnd_noise, axis=0)
        else:
            f_val -= np.mean(f_val)
            grad = np.mean(f_val * rnd_noise, axis=0)

        # Compute update
        result = grad / np.linalg.norm(grad) if self.norm == 2 else np.sign(grad)

        return result

    def _adversarial_satisfactory(self, samples: np.ndarray, target: int) -> np.ndarray:
        """
        Check whether an image is adversarial.

        Parameters
        ----------
        samples : np.ndarray
            The input data.
        target : int
            The target label.

        Returns
        -------
        np.ndarray
            An array of 0/1.
        """
        samples = np.clip(samples, self._clip_min, self._clip_max)
        preds = self.predict(samples)

        result = preds == target if self.targeted else preds != target

        return result

    @staticmethod
    def _interpolate(
        current_sample: np.ndarray,
        original_sample: np.ndarray,
        alpha: float,
        norm: Union[int, float, str],  # noqa: PYI041
    ) -> np.ndarray:
        """
        Interpolate a new sample based on the original and the current samples.

        Parameters
        ----------
        current_sample : np.ndarray
            The current adversarial example.
        original_sample : np.ndarray
            The original input.
        alpha : float
            The interpolation factor.
        norm : Union[int, float, str]
            Order of the norm. Possible values: "inf", np.inf or 2.

        Returns
        -------
        np.ndarray
            The interpolated sample.
        """
        if norm == 2:
            result = (1 - alpha) * original_sample + alpha * current_sample
        else:
            result = np.clip(current_sample, original_sample - alpha, original_sample + alpha)

        return result
