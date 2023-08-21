from holisticai.utils.models.feature_importance.global_feature_importance.permutation_feature_importance import compute_permutation_feature_importance
from holisticai.utils.models.feature_importance.global_feature_importance.surrogate_feature_importance import compute_surrogate_feature_importance
from holisticai.utils.models.feature_importance.local_feature_importance.lime_feature_importance import compute_lime_feature_importance

def Explainer(based_on, strategy_type, model_type, model, x, y):
    if based_on == 'feature_importance':

        if strategy_type == 'permutation':
            return compute_permutation_feature_importance(model_type, model, x, y)
        
        elif strategy_type == 'surrogate':
            return compute_surrogate_feature_importance(model_type, model, x, y)
        
        elif strategy_type == 'lime':
            return compute_lime_feature_importance(model_type, model, x, y)
        else:
            raise NotImplementedError