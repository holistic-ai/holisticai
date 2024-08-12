"""
This module implements the zeroth-order optimization attack `ZooAttack`. This is a black-box attack. This attack is a
variant of the Carlini and Wagner attack which uses ADAM coordinate descent to perform numerical estimation of
gradients.

| Paper link: https://arxiv.org/abs/1708.03999
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from holisticai.robustness.attackers.classification.commons import (
    format_function_predict_proba,
    to_categorical,
    x_array_to_df,
    x_to_nd_array,
)
from scipy.ndimage import zoom

BATCH_SIZE = 1


class ZooAttack:
    """
    The black-box zeroth-order optimization attack from Pin-Yu Chen et al. (2018). This attack is a variant of the
    C&W attack which uses ADAM coordinate descent to perform numerical estimation of gradients.

    Parameters
    ----------
    name : str, optional
        The name of the attack. The default is "Zoo".
    confidence : float, optional
        Confidence of adversarial examples. A higher value produces examples that are farther away, but more strongly\\
        classified as adversarial. The default is 0.0.
    targeted : bool, optional
        Indicates whether the attack is targeted. The default is False. If True, the positive ground truth is used as the target.
    learning_rate : float, optional
        The learning rate for the ADAM optimizer. The default is 1e-2.
    max_iter : int, optional
        The maximum number of iterations. The default is 20.
    binary_search_steps : int, optional
        The number of binary search steps. The default is 10.
    initial_const : float, optional
        The initial constant used to scale the adversarial perturbation. The default is 1e-3.
    abort_early : bool, optional
        Indicates whether to abort the optimization early. The default is True.
    use_resize : bool, optional
        Indicates whether to use resizing. The default is False.
    use_importance : bool, optional
        Indicates whether to use importance sampling. The default is False.
    nb_parallel : int, optional
        The number of parallel coordinates to update. The default is 1.
    variable_h : float, optional
        The variable h. The default is 0.2.
    verbose : bool, optional
        Indicates whether to print verbose output. The default is True.
    input_is_feature_vector : bool, optional
        Indicates whether the input is a feature vector. The default is False.
    proxy : callable, optional
        The model used to predict the probabilities of the input. The default is None.
    input_size : int, optional
        The size of the input. The default is 0.
    nb_classes : int, optional
        The number of classes. The default is 2.
    adam_mean : Optional[NDArray|ArrayLike|None], optional
        The mean of the ADAM optimizer. The default is None.
    adam_var : Optional[NDArray|ArrayLike|None], optional
        The variance of the ADAM optimizer. The default is None.
    adam_epochs : Optional[NDArray|ArrayLike|None], optional
        The epochs of the ADAM optimizer. The default is None.
    """

    def __init__(
        self,
        name="Zoo",
        confidence=0.0,
        targeted=False,
        learning_rate=1e-2,
        max_iter=20,
        binary_search_steps=10,
        initial_const=1e-3,
        abort_early=True,
        use_resize=False,
        use_importance=False,
        nb_parallel=1,
        variable_h=0.2,
        verbose=True,
        input_is_feature_vector=False,
        proxy=None,
        input_size=0,
        nb_classes=2,
        adam_mean=None,
        adam_var=None,
        adam_epochs=None,
    ):
        self.name = name
        self.confidence = confidence
        self.targeted = targeted
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const
        self.abort_early = abort_early
        self.use_resize = use_resize
        self.use_importance = use_importance
        self.nb_parallel = nb_parallel
        self.batch_size = BATCH_SIZE
        self.variable_h = variable_h
        self.verbose = verbose
        self.input_is_feature_vector = input_is_feature_vector
        self.predict_proba_fn = format_function_predict_proba(proxy.learning_task, proxy.predict_proba)
        self.input_size = input_size
        self.nb_classes = nb_classes
        self.adam_mean = adam_mean
        self.adam_var = adam_var
        self.adam_epochs = adam_epochs

    def _initialize_vars(self, x: np.ndarray) -> None:
        """
        Initialize the variables.

        Parameters
        ----------
        x : np.ndarray
            The input samples.
        """
        self.input_shape = tuple(x.shape[1:])
        self.input_size = np.prod(self.input_shape)
        if len(self.input_shape) == 1:
            self.input_is_feature_vector = True
            if self.batch_size != 1:
                raise ValueError(
                    "The current implementation of Zeroth-Order Optimisation attack only supports "
                    "`batch_size=1` with feature vectors as input."
                )
        else:
            self.input_is_feature_vector = False

        # Initialize some internal variables
        self._init_size = 32
        if self.abort_early:
            self._early_stop_iters = self.max_iter // 10 if self.max_iter >= 10 else self.max_iter

        # Initialize noise variable to zero
        if self.input_is_feature_vector:
            self.use_resize = False
            self.use_importance = False

        if self.use_resize:
            dims = (self.batch_size, self.input_shape[0], self._init_size, self._init_size)
            self._current_noise = np.zeros(dims, dtype=np.float32)
        else:
            self._current_noise = np.zeros((self.batch_size, *self.input_shape), dtype=np.float32)
        self._sample_prob = np.ones(self._current_noise.size, dtype=np.float32) / self._current_noise.size

    def _loss(
        self, x: np.ndarray, x_adv: np.ndarray, target: np.ndarray, c_weight: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the loss function values.

        Parameters
        ----------
        x : np.ndarray
            The original input.
        x_adv : np.ndarray
            The adversarial input.
        target : np.ndarray
            The target values.
        c_weight : np.ndarray
            The weight of the constant.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            The predictions, the L2 distances, and the loss values.
        """
        l2dist = np.sum(np.square(x - x_adv).reshape(x_adv.shape[0], -1), axis=1)
        ratios = [1.0] + [int(new_size) / int(old_size) for new_size, old_size in zip(self.input_shape, x.shape[1:])]
        preds = self.predict_proba(np.array(zoom(x_adv, zoom=ratios)))
        z_target = np.sum(preds * target, axis=1)
        z_other = np.max(
            preds * (1 - target) + (np.min(preds, axis=1) - 1)[:, np.newaxis] * target,
            axis=1,
        )

        if self.targeted:
            # If targeted, optimize for making the target class most likely
            loss = np.maximum(z_other - z_target + self.confidence, 0)
        else:
            # If untargeted, optimize for making any other class most likely
            loss = np.maximum(z_target - z_other + self.confidence, 0)

        return preds, l2dist, c_weight * loss + l2dist

    def generate(self, x_df: pd.DataFrame, y: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Generate adversarial samples and return them in an array.

        Parameters
        ----------
        x_df : pd.DataFrame
            The input samples.
        y : Optional[np.ndarray], optional
            The target labels. The default is None.

        Returns
        -------
        pd.DataFrame
            The adversarial samples.
        """
        self._initialize_vars(x_df)
        feature_names = list(x_df.columns)
        self.predict_proba = lambda x: self.predict_proba_fn(x, feature_names)

        x = x_to_nd_array(x_df)

        self._clip_values = (np.min(x), np.max(x))

        if y is not None:
            y = to_categorical(y, nb_classes=self.nb_classes)

        # Check that `y` is provided for targeted attacks
        if self.targeted and y is None:  # pragma: no cover
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # No labels provided, use model prediction as correct class
        if y is None:
            y = self.predict_proba(x)

        if self.nb_classes == 2 and y.shape[1] == 1:  # pragma: no cover
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        # Compute adversarial examples with implicit batching
        nb_batches = int(np.ceil(x.shape[0] / float(self.batch_size)))
        x_adv_list = []
        for batch_id in range(nb_batches):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            x_batch = x[batch_index_1:batch_index_2]
            y_batch = y[batch_index_1:batch_index_2]
            res = self._generate_batch(x_batch, y_batch)
            x_adv_list.append(res)
        x_adv = np.vstack(x_adv_list)

        # Apply clip
        # clip_min, clip_max = self._clip_values
        # np.clip(x_adv, clip_min-0.1, clip_max+0.1, out=x_adv)

        return x_array_to_df(x_adv, feature_names=feature_names)

    def _generate_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        """
        Run the attack on a batch of images and labels.

        Parameters
        ----------
        x_batch : np.ndarray
            A batch of original examples.
        y_batch : np.ndarray
            A batch of targets (0-1 hot).

        Returns
        -------
        np.ndarray
            A batch of adversarial examples.
        """
        # Initialize binary search
        c_current = self.initial_const * np.ones(x_batch.shape[0])
        c_lower_bound = np.zeros(x_batch.shape[0])
        c_upper_bound = 1e10 * np.ones(x_batch.shape[0])

        # Initialize best distortions and best attacks globally
        o_best_dist = np.inf * np.ones(x_batch.shape[0])
        o_best_attack = x_batch.copy()

        # Start with a binary search
        for _ in range(self.binary_search_steps):
            # Run with 1 specific binary search step
            best_dist, best_label, best_attack = self._generate_bss(x_batch, y_batch, c_current)

            # Update best results so far
            o_best_attack[best_dist < o_best_dist] = best_attack[best_dist < o_best_dist]
            o_best_dist[best_dist < o_best_dist] = best_dist[best_dist < o_best_dist]

            # Adjust the constant as needed
            c_current, c_lower_bound, c_upper_bound = self._update_const(
                y_batch, best_label, c_current, c_lower_bound, c_upper_bound
            )

        return o_best_attack

    def _update_const(
        self,
        y_batch: np.ndarray,
        best_label: np.ndarray,
        c_batch: np.ndarray,
        c_lower_bound: np.ndarray,
        c_upper_bound: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update constant `c_batch` from the ZOO objective. This characterizes the trade-off between attack strength and
        amount of noise introduced.

        Parameters
        ----------
        y_batch : np.ndarray
            A batch of targets (0-1 hot).
        best_label : np.ndarray
            The best labels.
        c_batch : np.ndarray
            A batch of constants.
        c_lower_bound : np.ndarray
            The lower bound of the constant.
        c_upper_bound : np.ndarray
            The upper bound of the constant.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            The updated constant, lower bound, and upper bound.
        """

        comparison = [
            self._compare(best_label[i], np.argmax(y_batch[i])) and best_label[i] != -np.inf
            for i in range(len(c_batch))
        ]
        for i, comp in enumerate(comparison):
            if comp:
                # Successful attack
                c_upper_bound[i] = min(c_upper_bound[i], c_batch[i])
                if c_upper_bound[i] < 1e9:
                    c_batch[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2
            else:
                # Failure attack
                c_lower_bound[i] = max(c_lower_bound[i], c_batch[i])
                c_batch[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2 if c_upper_bound[i] < 1e9 else c_batch[i] * 10

        return c_batch, c_lower_bound, c_upper_bound

    def _compare(self, object1: Any, object2: Any) -> bool:
        """
        Check two objects for equality if the attack is targeted, otherwise check for inequality.

        Parameters
        ----------
        object1 : Any
            The first object.
        object2 : Any
            The second object.

        Returns
        -------
        bool
            The result of the comparison.
        """
        return object1 == object2 if self.targeted else object1 != object2

    def _generate_bss(
        self, x_batch: np.ndarray, y_batch: np.ndarray, c_batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate adversarial examples for a batch of inputs with a specific batch of constants.

        Parameters
        ----------
        x_batch : np.ndarray
            A batch of original examples.
        y_batch : np.ndarray
            A batch of targets (0-1 hot).
        c_batch : np.ndarray
            A batch of constants.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            The best distortions, the best labels, and the best attacks.
        """

        x_orig = x_batch.astype(np.float32)
        fine_tuning = np.full(x_batch.shape[0], False, dtype=bool)
        prev_loss = 1e6 * np.ones(x_batch.shape[0])
        prev_l2dist = np.zeros(x_batch.shape[0])

        # Resize and initialize Adam
        if self.use_resize:
            x_orig = self._resize_image(x_orig, self._init_size, self._init_size, True)
            assert (x_orig != 0).any()
            x_adv = x_orig.copy()
        else:
            x_orig = x_batch
            self._reset_adam(np.prod(self.input_shape).item())
            if x_batch.shape == self._current_noise.shape:
                self._current_noise.fill(0)
            else:
                self._current_noise = np.zeros(x_batch.shape, dtype=np.float32)
            x_adv = x_orig.copy()

        # Initialize best distortions, best changed labels and best attacks
        best_dist = np.inf * np.ones(x_adv.shape[0])
        best_label = -np.inf * np.ones(x_adv.shape[0])
        best_attack = np.array([x_adv[i] for i in range(x_adv.shape[0])])

        for iter_ in range(self.max_iter):
            # Upscaling for very large number of iterations
            if self.use_resize:
                if iter_ == 2000:
                    x_adv = self._resize_image(x_adv, 64, 64)
                    x_orig = zoom(
                        x_orig,
                        [
                            1,
                            x_adv.shape[1] / x_orig.shape[1],
                            x_adv.shape[2] / x_orig.shape[2],
                            x_adv.shape[3] / x_orig.shape[3],
                        ],
                    )
                elif iter_ == 10000:
                    x_adv = self._resize_image(x_adv, 128, 128)
                    x_orig = zoom(
                        x_orig,
                        [
                            1,
                            x_adv.shape[1] / x_orig.shape[1],
                            x_adv.shape[2] / x_orig.shape[2],
                            x_adv.shape[3] / x_orig.shape[3],
                        ],
                    )

            # Compute adversarial examples and loss
            x_adv = self._optimizer(x_adv, y_batch, c_batch)
            preds, l2dist, loss = self._loss(x_orig, x_adv, y_batch, c_batch)

            # Reset Adam if a valid example has been found to avoid overshoot
            mask_fine_tune = (~fine_tuning) & (loss == l2dist) & (prev_loss != prev_l2dist)
            fine_tuning[mask_fine_tune] = True
            self._reset_adam(self.adam_mean.size, np.repeat(mask_fine_tune, x_adv[0].size))  # type: ignore
            prev_l2dist = l2dist

            # Abort early if no improvement is obtained
            if self.abort_early and iter_ % self._early_stop_iters == 0:
                if (loss > 0.9999 * prev_loss).all():
                    break
                prev_loss = loss

            # Adjust the best result
            labels_batch = np.argmax(y_batch, axis=1)
            for i, (dist, pred) in enumerate(zip(l2dist, np.argmax(preds, axis=1))):
                if dist < best_dist[i] and self._compare(pred, labels_batch[i]):
                    best_dist[i] = dist
                    best_attack[i] = x_adv[i]
                    best_label[i] = pred

        # Resize images to original size before returning
        best_attack = np.array(best_attack)
        if self.use_resize:
            if not self.channels_first:
                best_attack = zoom(
                    best_attack,
                    [
                        1,
                        int(x_batch.shape[1]) / best_attack.shape[1],
                        int(x_batch.shape[2]) / best_attack.shape[2],
                        1,
                    ],
                )
            else:
                best_attack = zoom(
                    best_attack,
                    [
                        1,
                        1,
                        int(x_batch.shape[2]) / best_attack.shape[2],
                        int(x_batch.shape[2]) / best_attack.shape[3],
                    ],
                )

        return best_dist, best_label, best_attack

    def _optimizer(self, x: np.ndarray, targets: np.ndarray, c_batch: np.ndarray) -> np.ndarray:
        """
        Run the ADAM optimizer for a batch of inputs.

        Parameters
        ----------
        x : np.ndarray
            A batch of original examples.
        targets : np.ndarray
            A batch of targets (0-1 hot).
        c_batch : np.ndarray
            A batch of constants.

        Returns
        -------
        np.ndarray
            A batch of adversarial examples.
        """
        # Variation of input for computing loss, same as in original implementation
        coord_batch = np.repeat(self._current_noise, 2 * self.nb_parallel, axis=0)
        coord_batch = coord_batch.reshape(2 * self.nb_parallel * self._current_noise.shape[0], -1)

        # Sample indices to prioritize for optimization
        if self.use_importance and np.unique(self._sample_prob).size != 1:
            indices = (
                np.random.choice(
                    coord_batch.shape[-1] * x.shape[0],
                    self.nb_parallel * self._current_noise.shape[0],
                    replace=False,
                    p=self._sample_prob.flatten(),
                )
                % coord_batch.shape[-1]
            )
        else:
            try:
                indices = (
                    np.random.choice(
                        coord_batch.shape[-1] * x.shape[0],
                        self.nb_parallel * self._current_noise.shape[0],
                        replace=False,
                    )
                    % coord_batch.shape[-1]
                )
            except ValueError as error:  # pragma: no cover
                if "Cannot take a larger sample than population when 'replace=False'" in str(error):
                    raise ValueError(
                        "Too many samples are requested for the random indices. Try to reduce the number of parallel"
                        "coordinate updates `nb_parallel`."
                    ) from error

                raise

        # Create the batch of modifications to run
        for i in range(self.nb_parallel * self._current_noise.shape[0]):
            coord_batch[2 * i, indices[i]] += self.variable_h
            coord_batch[2 * i + 1, indices[i]] -= self.variable_h

        # Compute loss for all samples and coordinates, then optimize
        expanded_x = np.repeat(x, 2 * self.nb_parallel, axis=0).reshape((-1,) + x.shape[1:])
        expanded_targets = np.repeat(targets, 2 * self.nb_parallel, axis=0).reshape((-1,) + targets.shape[1:])
        expanded_c = np.repeat(c_batch, 2 * self.nb_parallel)
        _, _, loss = self._loss(
            expanded_x,
            expanded_x + coord_batch.reshape(expanded_x.shape),
            expanded_targets,
            expanded_c,
        )
        if self.adam_mean is not None and self.adam_var is not None and self.adam_epochs is not None:
            self._current_noise = self._optimizer_adam_coordinate(
                loss,
                indices,
                self.adam_mean,
                self.adam_var,
                self._current_noise,
                self.learning_rate,
                self.adam_epochs,
                True,
            )
        else:
            raise ValueError("Unexpected `None` in `adam_mean`, `adam_var` or `adam_epochs` detected.")

        if self.use_importance and self._current_noise.shape[2] > self._init_size:
            self._sample_prob = self._get_prob(self._current_noise).flatten()

        return x + self._current_noise

    def _optimizer_adam_coordinate(
        self,
        losses: np.ndarray,
        index: np.ndarray,
        mean: np.ndarray,
        var: np.ndarray,
        current_noise: np.ndarray,
        learning_rate: float,
        adam_epochs: np.ndarray,
        proj: bool,
    ) -> np.ndarray:
        """
        Implementation of the ADAM optimizer for coordinate descent.

        Parameters
        ----------
        losses : np.ndarray
            Overall loss.
        index : np.ndarray
            Indices of the coordinates to update.
        mean : np.ndarray
            The mean of the gradient (first moment).
        var : np.ndarray
            The uncentered variance of the gradient (second moment).
        current_noise : np.ndarray
            The current noise.
        learning_rate : float
            Learning rate for Adam optimizer.
        adam_epochs : np.ndarray
            Epochs to run the Adam optimizer.
        proj : bool
            Whether to project the noise to the L_p ball.

        Returns
        -------
        np.ndarray
            Updated noise for coordinate descent.
        """
        beta1, beta2 = 0.9, 0.999

        # Estimate grads from loss variation (constant `h` from the paper is fixed to .0001)
        grads = np.array([(losses[i] - losses[i + 1]) / (2 * self.variable_h) for i in range(0, len(losses), 2)])

        # ADAM update
        mean[index] = beta1 * mean[index] + (1 - beta1) * grads
        var[index] = beta2 * var[index] + (1 - beta2) * grads**2

        corr = (np.sqrt(1 - np.power(beta2, adam_epochs[index]))) / (1 - np.power(beta1, adam_epochs[index]))
        orig_shape = current_noise.shape
        current_noise = current_noise.reshape(-1)
        current_noise[index] -= learning_rate * corr * mean[index] / (np.sqrt(var[index]) + 1e-8)
        adam_epochs[index] += 1

        if proj and hasattr(self, "_clip_values") and self._clip_values is not None:
            clip_min, clip_max = self._clip_values
            current_noise[index] = np.clip(current_noise[index], clip_min, clip_max)

        return current_noise.reshape(orig_shape)

    def _reset_adam(self, nb_vars: int, indices: Optional[np.ndarray] = None) -> None:
        """
        Reset the ADAM optimizer.

        Parameters
        ----------
        nb_vars : int
            The number of variables.
        indices : Optional[np.ndarray], optional
            The indices to reset. The default is None.
        """
        # If variables are already there and at the right size, reset values
        if self.adam_mean is not None and self.adam_mean.size == nb_vars:
            if indices is None:
                self.adam_mean.fill(0)
                self.adam_var.fill(0)  # type: ignore
                self.adam_epochs.fill(1)  # type: ignore
            else:
                self.adam_mean[indices] = 0
                self.adam_var[indices] = 0  # type: ignore
                self.adam_epochs[indices] = 1  # type: ignore
        else:
            # Allocate Adam variables
            self.adam_mean = np.zeros(nb_vars, dtype=np.float32)
            self.adam_var = np.zeros(nb_vars, dtype=np.float32)
            self.adam_epochs = np.ones(nb_vars, dtype=int)

    def _resize_image(self, x: np.ndarray, size_x: int, size_y: int, reset: bool = False) -> np.ndarray:
        """
        Resize the image to a specific size.

        Parameters
        ----------
        x : np.ndarray
            The input image.
        size_x : int
            The size in the x direction.
        size_y : int
            The size in the y direction.
        reset : bool, optional
            Indicates whether to reset the image. The default is False.

        Returns
        -------
        np.ndarray
            The resized image.
        """
        if not self.channels_first:
            dims = (x.shape[0], size_x, size_y, x.shape[-1])
        else:
            dims = (x.shape[0], x.shape[1], size_x, size_y)
        nb_vars = np.prod(dims).item()

        if reset:
            # Reset variables to original size and value
            if dims == x.shape:
                resized_x = x
                if x.shape == self._current_noise.shape:
                    self._current_noise.fill(0)
                else:
                    self._current_noise = np.zeros(x.shape, dtype=np.float32)
            else:
                resized_x = zoom(
                    x,
                    (
                        1,
                        dims[1] / x.shape[1],
                        dims[2] / x.shape[2],
                        dims[3] / x.shape[3],
                    ),
                )
                self._current_noise = np.zeros(dims, dtype=np.float32)
            self._sample_prob = np.ones(nb_vars, dtype=np.float32) / nb_vars
        else:
            # Rescale variables and reset values
            resized_x = zoom(x, (1, dims[1] / x.shape[1], dims[2] / x.shape[2], dims[3] / x.shape[3]))
            self._sample_prob = self._get_prob(self._current_noise, double=True).flatten()
            self._current_noise = np.zeros(dims, dtype=np.float32)

        # Reset Adam
        self._reset_adam(nb_vars)

        return resized_x

    def _get_prob(self, prev_noise: np.ndarray, double: bool = False) -> np.ndarray:
        """
        Compute the probability of each pixel to be selected for optimization.

        Parameters
        ----------
        prev_noise : np.ndarray
            The previous noise.
        double : bool, optional
            Indicates whether to double the size. The default is False.

        Returns
        -------
        np.ndarray
            The probability of each pixel to be selected for optimization.
        """
        dims = list(prev_noise.shape)
        channel_index = 1 if self.channels_first else 3

        # Double size if needed
        if double:
            dims = [2 * size if i not in [0, channel_index] else size for i, size in enumerate(dims)]

        prob = np.empty(shape=dims, dtype=np.float32)
        image = np.abs(prev_noise)

        for channel in range(prev_noise.shape[channel_index]):
            if not self.channels_first:
                image_pool = self._max_pooling(image[:, :, :, channel], dims[1] // 8)
                if double:
                    prob[:, :, :, channel] = np.abs(zoom(image_pool, [1, 2, 2]))
                else:
                    prob[:, :, :, channel] = image_pool
            elif self.channels_first:
                image_pool = self._max_pooling(image[:, channel, :, :], dims[2] // 8)
                if double:
                    prob[:, channel, :, :] = np.abs(zoom(image_pool, [1, 2, 2]))
                else:
                    prob[:, channel, :, :] = image_pool

        prob /= np.sum(prob)

        return prob

    @staticmethod
    def _max_pooling(image: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Perform max pooling on the image.

        Parameters
        ----------
        image : np.ndarray
            The input image.
        kernel_size : int
            The size of the kernel.

        Returns
        -------
        np.ndarray
            The pooled image.
        """
        img_pool = np.copy(image)
        for i in range(0, image.shape[1], kernel_size):
            for j in range(0, image.shape[2], kernel_size):
                img_pool[:, i : i + kernel_size, j : j + kernel_size] = np.max(
                    image[:, i : i + kernel_size, j : j + kernel_size],
                    axis=(1, 2),
                    keepdims=True,
                )

        return img_pool
