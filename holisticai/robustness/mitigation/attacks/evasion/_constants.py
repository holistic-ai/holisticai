import numpy as np

from holisticai.robustness.mitigation.attacks.evasion import (
    AutoAttack,
    BoundaryAttack,
    BrendelBethgeAttack,
    DecisionTreeAttack,
    DeepFool,
    FastGradientMethod,
    HopSkipJump,
    ProjectedGradientDescentNumpy,
    ZooAttack,
)

ALL_ATTACKERS = {
    "HopSkipJump": {
        "attacker": HopSkipJump,
        "params": {"targeted": False, "max_iter": 0, "max_eval": 1000, "init_eval": 10},
    },
    "DeepFool": {
        "attacker": DeepFool,
        "params": {"max_iter": 5, "batch_size": 11, "verbose": False},
    },
    "ProjectedGradientDescentNumpy": {
        "attacker": ProjectedGradientDescentNumpy,
        "params": {
            "eps": 1.0,
            "eps_step": 0.1,
            "max_iter": 5,
            "norm": np.inf,
            "targeted": False,
            "num_random_init": 0,
            "batch_size": 3,
            "random_eps": False,
            "verbose": False,
        },
    },
    "ZooAttack": {
        "attacker": ZooAttack,
        "params": {
            "confidence": 0.0,
            "targeted": False,
            "learning_rate": 1e-1,
            "max_iter": 20,
            "binary_search_steps": 10,
            "initial_const": 1e-3,
            "abort_early": True,
            "use_resize": False,
            "use_importance": False,
            "nb_parallel": 1,
            "batch_size": 1,
            "variable_h": 0.2,
        },
    },
    "BoundaryAttack": {
        "attacker": BoundaryAttack,
        "params": {"targeted": True, "max_iter": 10, "verbose": False},
    },
    "BrendelBethgeAttack": {
        "attacker": BrendelBethgeAttack,
        "params": {"targeted": True, "max_iter": 10, "verbose": False},
    },
    "FastGradientMethod": {"attacker": FastGradientMethod, "params": {"eps": 1}},
    "DecisionTreeAttack": {"attacker": DecisionTreeAttack, "params": {}},
}

PYTORCH_ATTACKERS = [
    "HopSkipJump",
    "DeepFool",
    "ProjectedGradientDescentNumpy",
    "BoundaryAttack",
    "BrendelBethgeAttack",
    "FastGradientMethod",
]


SKLEARN_ATTACKERS = [
    "HopSkipJump",
    "ZooAttack",
    "BoundaryAttack",
]

DECISION_TREE_ATTACKERS = [
    "DecisionTreeAttack",
    "HopSkipJump",
    "ZooAttack",
]


def Attacker(attacker_name, **attacker_params):
    params = ALL_ATTACKERS[attacker_name]["params"]
    params.update(attacker_params)
    return ALL_ATTACKERS[attacker_name]["attacker"](**params)
