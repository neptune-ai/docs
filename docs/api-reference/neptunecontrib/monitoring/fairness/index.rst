

:mod:`neptunecontrib.monitoring.fairness`
=========================================

.. py:module:: neptunecontrib.monitoring.fairness


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   neptunecontrib.monitoring.fairness.log_fairness_classification_metrics
   neptunecontrib.monitoring.fairness._make_dataset
   neptunecontrib.monitoring.fairness._fmt_priveleged_info
   neptunecontrib.monitoring.fairness._log_fairness_metrics
   neptunecontrib.monitoring.fairness._plot_confusion_matrix_by_group
   neptunecontrib.monitoring.fairness._plot_performance_by_group
   neptunecontrib.monitoring.fairness._add_annotations
   neptunecontrib.monitoring.fairness._format_aif360_to_sklearn


.. function:: log_fairness_classification_metrics(y_true, y_pred_class, y_pred_score, sensitive_attributes, favorable_label, unfavorable_label, privileged_groups, unprivileged_groups, experiment=None, prefix='')

   Creates fairness metric charts, calculates fairness classification metrics and logs them to Neptune.

   Class-based metrics that are logged: 'true_positive_rate_difference','false_positive_rate_difference',
   'false_omission_rate_difference', 'false_discovery_rate_difference', 'error_rate_difference',
   'false_positive_rate_ratio', 'false_negative_rate_ratio', 'false_omission_rate_ratio',
   'false_discovery_rate_ratio', 'error_rate_ratio', 'average_odds_difference', 'disparate_impact',
   'statistical_parity_difference', 'equal_opportunity_difference', 'theil_index',
   'between_group_theil_index', 'between_all_groups_theil_index', 'coefficient_of_variation',
   'between_group_coefficient_of_variation', 'between_all_groups_coefficient_of_variation',
   'generalized_entropy_index', 'between_group_generalized_entropy_index',
   'between_all_groups_generalized_entropy_index'

   Charts are logged to the 'metric_by_group' channel: 'confusion matrix', 'TPR', 'TNR', 'FPR', 'FNR', 'PPV', 'NPV',
   'FDR', 'FOR', 'ACC', 'error_rate', 'selection_rate', 'power', 'precision', 'recall',
   'sensitivity', 'specificity'.

   :param y_true: Ground truth (correct) target values.
   :type y_true: array-like, shape (n_samples)
   :param y_pred_class: Class predictions with values 0 or 1.
   :type y_pred_class: array-like, shape (n_samples)
   :param y_pred_score: Class predictions with values from 0 to 1. Default None.
   :type y_pred_score: array-like, shape (n_samples)
   :param sensitive_attributes: datafame containing only sensitive columns.
   :type sensitive_attributes: pandas.DataFrame, shape (n_samples, k)
   :param favorable_label: label that is favorable, brings positive value to a person being classified.
   :type favorable_label: str or int
   :param unfavorable_label: label that is unfavorable, brings positive value to a person being classified.
   :type unfavorable_label: str or int
   :param privileged_groups: dictionary with column names and list of values for those columns that
                             belong to the privileged groups.
   :type privileged_groups: dict
   :param unprivileged_groups: dictionary with column names and list of values for those columns that
                               belong to the unprivileged groups.
   :type unprivileged_groups: dict
   :param experiment: Neptune experiment. Default is None.
   :type experiment: `neptune.experiments.Experiment`
   :param prefix: Prefix that will be added before metric name when logged to Neptune.
   :type prefix: str

   .. rubric:: Examples

   Train the model and make predictions on test.
   Log metrics and performance curves to Neptune::

       import neptune
       from neptunecontrib.monitoring.fairness import log_fairness_classification_metrics

       neptune.init()
       with neptune.create_experiment():
           log_fairness_classification_metrics(y_true, y_pred_class, y_pred_score, test[['race']],
                                               favorable_label='granted_parole',
                                               unfavorable_label='not_granted_parole',
                                               privileged_groups={'race':['Caucasian']},
                                               privileged_groups={'race':['African-American','Hispanic]},
                                               )

   Check out this experiment https://ui.neptune.ai/jakub-czakon/model-fairness/e/MOD-92/logs.


.. function:: _make_dataset(features, labels, scores=None, protected_columns=None, privileged_groups=None, unprivileged_groups=None, favorable_label=None, unfavorable_label=None)


.. function:: _fmt_priveleged_info(privileged_groups, unprivileged_groups)


.. function:: _log_fairness_metrics(aif_metric, experiment, prefix)


.. function:: _plot_confusion_matrix_by_group(aif_metric, figsize=None)


.. function:: _plot_performance_by_group(aif_metric, metric_name, ax=None)


.. function:: _add_annotations(ax)


.. function:: _format_aif360_to_sklearn(aif360_mat)



.. External links

.. |Neptune| raw:: html

    <a href="/api-reference/neptune/index.html#functions" target="_blank">Neptune</a>

.. |Session| raw:: html

    <a href="/api-reference/neptune/sessions/index.html?highlight=neptune%20sessions%20session#neptune.sessions.Session" target="_blank">Session</a>

.. |Project| raw:: html

    <a href="/api-reference/neptune/projects/index.html#neptune.projects.Project" target="_blank">Project</a>

.. |Experiment| raw:: html

    <a href="/api-reference/neptune/experiments/index.html?highlight=neptune%20experiment#neptune.experiments.Experiment" target="_blank">Experiment</a>

.. |Notebook| raw:: html

    <a href="/api-reference/neptune/notebook/index.html?highlight=notebook#neptune.notebook.Notebook" target="_blank">Notebook</a>

.. |Git Info| raw:: html

    <a href="/api-reference/neptune/git_info/index.html#neptune.git_info.GitInfo" target="_blank">Git Info</a>