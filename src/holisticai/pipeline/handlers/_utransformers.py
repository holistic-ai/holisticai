from xml.dom import NotSupportedErr

from holisticai.utils.transformers.bias import (
    BIAS_TAGS,
    BMInprocessing,
    BMPostprocessing,
    BMPreprocessing,
)


class UTransformersHandler:
    def __init__(self, steps, params_hdl, estimator_hdl):
        """
        Initialize step groups and apply some validations.

        Description
        ----------
        Create bias mitigation groups for preprocessing, inprocessing and postprocessing strategies.
        Pipeline support only one postprocessing in the pipeline.

        Parameters
        ----------
        steps : list
            Pipeline steps

        params_hdl : UTransformersHandler
            Pipeline parameters handler during fit, fit_transform, transform function execution.
        """
        self.bias_mitigators_validation(steps)
        self.steps_groups = {tag: [step for step in steps if step[0].startswith(tag)] for tag in BIAS_TAGS}
        for steps in self.steps_groups.values():
            for step in steps:
                step[1]._link_parameters(params_hdl, estimator_hdl)  # noqa: SLF001

    def bias_mitigators_validation(self, steps):
        """Validate stem words and bias mitigator position in the pipeline"""
        tag2info = {
            BIAS_TAGS.PRE: BMPreprocessing,
            BIAS_TAGS.INP: BMInprocessing,
            BIAS_TAGS.POST: BMPostprocessing,
        }

        mitigator_groups_by_name = {tag: [step for step in steps if step[0].startswith(tag)] for tag in BIAS_TAGS}
        mitigator_groups_by_object = {
            tag: [step for step in steps if isinstance(step[1], tag2info[tag])] for tag in BIAS_TAGS
        }

        # Validate if all objects with bias mitigator stem (BIAS_TAGS) are linked with a correct Bias Mitigator Transformer objects
        if mitigator_groups_by_name != mitigator_groups_by_object:
            msg = f"Mitigator objects and name doesn't match, grouped by name: {mitigator_groups_by_name} \
                and grouped by object type:{mitigator_groups_by_object}"
            raise NameError(msg)

        num_post_mitigators = len(mitigator_groups_by_name[BIAS_TAGS.POST])

        # Validate if postprocessor bias mitigators are defined after classifier
        if num_post_mitigators > 0:
            post_classifier_step_names, _ = zip(*mitigator_groups_by_name[BIAS_TAGS.POST])
            assert all(
                name.startswith(BIAS_TAGS.POST) for name in post_classifier_step_names
            ), f"Only bias mitigators postprocessor are supported, utransformer postprocessors founded: {post_classifier_step_names}"

        # Validate that exists only one bias mitigator postprocessor
        # TODO: Evaluate in other cases.
        if not len(mitigator_groups_by_name[BIAS_TAGS.POST]) <= 1:
            msg = "Pipeline supports max 1 postprocessor mitigator."
            raise NotSupportedErr(msg)

    def drop_post_processing_steps(self, steps):
        """
        Drop post-processing steps from input steps list.

        Description
        ----------
        The function check the steps names and drop names with prefix 'bm_pos'.

        Parameters
        ----------
        steps : list
            Pipeline steps

        Returns
        -------
        list
            New steps list
        """
        return [step for step in steps if not step[0].startswith(BIAS_TAGS.POST)]

    @property
    def pos_processing_steps(self):
        """
        Return bias mitigation post-estimator steps.
        """
        return self.steps_groups[BIAS_TAGS.POST]

    def fit_postprocessing(self, Xt, y, **kargs):
        """
        Fit the pos-estimator transformers.

        Description
        ----------
        Call `fit` of each post-estimator transformer in the pipeline.

        Parameters
        ----------
        Xt : numpy array
            Transformer input data

        y: numpy array
            Target vector

        Returns
        -------
        None
        """
        step = self.pos_processing_steps[0]
        step[1].fit(X=Xt, y=y, **kargs)

    def transform_postprocessing(self, **kargs):
        """
        Compute the transformed prediction vector with pos-estimator transformers.

        Description
        ----------
        Call `transform` of each post-estimator transformer in the pipeline.

        Parameters
        ----------
        y_pred : list
            predicted label vector

        y_proba : list
            predicted probability vector

        Returns
        -------
        dict
            Dictionaty with post-processed prediction vectors
        """
        step = self.pos_processing_steps[0]
        return step[1].transform(**kargs)
