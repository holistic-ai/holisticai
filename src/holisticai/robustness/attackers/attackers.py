import numpy as np
from holisticai.robustness.attackers.classification import HopSkipJump, ZooAttack
from holisticai.robustness.attackers.classification.commons import format_function_predict_proba


class BinClassAttacker:
    """
    A binary classification attacker to generate adversarial examples using different attack strategies.
    The available attackers are: zeroth-order optimization (Zoo) and HopSkipJump (HSJ).

    Parameters
    ----------
    attacker_name : str
        The name of the attacker. Available options are: "HSJ" and "Zoo".
    x : pd.DataFrame
        The input data.
    mask : np.ndarray
        The mask used to select the sensitive features.
    """

    def __init__(self, attacker_name, x, model):
        self.attacker_name = attacker_name
        self.x = x
        self.model = model
        self.attacker = self._get_attacker(x, model)

    def generate(self, mask=None):
        """
        Generates adversarial examples using the attacker object.

        Parameters
        ----------
        mask : np.ndarray, optional
            The mask used to select the sensitive features. The default is None if the input data is not sensitive.

        Returns
        -------
        np.ndarray
            The generated adversarial examples.

        """
        return self.attacker.generate(x_df=self.x, mask=mask)

    def _get_attacker(self, x, model):
        """
        Returns the attacker object based on the attacker name.

        Parameters
        ----------
        x : pd.DataFrame
            The input data.
        model : object
            The model used for prediction.

        Returns
        -------
        object
            The attacker object.

        Raises
        ------
        NotImplementedError
            If the attacker name is not supported.

        """
        match self.attacker_name:
            case "HSJ":
                kargs = {
                    "predictor": model.predict,
                    "clip_values": (np.min(x), np.max(x)),
                    "input_shape": tuple(x.shape[1:]),
                }
                return HopSkipJump(name=self.attacker_name, **kargs)
            case "Zoo":
                predict_proba_fn = format_function_predict_proba(model.learning_task, model.predict_proba)  # type: ignore
                kargs = {
                    "predict_proba_fn": predict_proba_fn,
                    "clip_values": (np.min(x), np.max(x)),
                    "input_shape": tuple(x.shape[1:]),
                }
                return ZooAttack(name=self.attacker_name, **kargs)
        raise NotImplementedError
