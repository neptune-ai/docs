

:mod:`neptunecontrib.monitoring.optuna`
=======================================

.. py:module:: neptunecontrib.monitoring.optuna


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   neptunecontrib.monitoring.optuna.NeptuneCallback



Functions
~~~~~~~~~

.. autoapisummary::

   neptunecontrib.monitoring.optuna.log_study_info
   neptunecontrib.monitoring.optuna.log_study
   neptunecontrib.monitoring.optuna.NeptuneMonitor


.. py:class:: NeptuneCallback(experiment=None, log_charts=False, log_study=False, params=None)

   Logs hyperparameter optimization process to Neptune.

   For each iteration it logs run metric and run parameters as well as the best score to date.

   :param experiment: Neptune experiment. Default is None.
   :type experiment: `neptune.experiments.Experiment`
   :param log_charts: Whether optuna visualization charts should be logged. By default no charts are logged.
   :type log_charts: 'bool'
   :param log_study: Whether optuna study object should be pickled and logged. By default it is not.
   :type log_study: 'bool'
   :param params: List of parameters to be visualized. Default is all parameters.
   :type params: `list`

   .. rubric:: Examples

   Initialize neptune_monitor::

       import neptune
       import neptunecontrib.monitoring.optuna as opt_utils

       neptune.init(api_token='ANONYMOUS',
                    project_qualified_name='shared/showroom')
       neptune.create_experiment(name='optuna sweep')

       neptune_callback = opt_utils.NeptuneCallback()

   Run Optuna training passing neptune_callback as callback::

       ...
       study = optuna.create_study(direction='maximize')
       study.optimize(objective, n_trials=100, callbacks=[neptune_callback])

   You can explore an example experiment in Neptune:
   https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-1016/artifacts

   You can also log optuna visualization charts and study object after every iteration::

       neptune_callback = opt_utils.NeptuneCallback(log_charts=True, log_study=True)

   .. method:: __call__(self, study, trial)



.. function:: log_study_info(study, experiment=None, log_charts=True, params=None)

   Logs runs results and parameters to neptune.

   Logs all hyperparameter optimization results to Neptune. Those include best score ('best_score' metric),
   best parameters ('best_parameters' property), the study object itself as artifact, and interactive optuna charts
   ('contour', 'parallel_coordinate', 'slice', 'optimization_history') as artifacts in 'charts' sub folder.

   :param study: Optuna study object after training is completed.
   :type study: 'optuna.study.Study'
   :param experiment: Neptune experiment. Default is None.
   :type experiment: `neptune.experiments.Experiment`
   :param log_charts: Whether optuna visualization charts should be logged. By default all charts are logged.
   :type log_charts: 'bool'
   :param params: List of parameters to be visualized. Default is all parameters.
   :type params: `list`

   .. rubric:: Examples

   Initialize neptune_monitor::

       import neptune
       import neptunecontrib.monitoring.optuna as opt_utils

       neptune.init(project_qualified_name='USER_NAME/PROJECT_NAME')
       neptune.create_experiment(name='optuna sweep')

       neptune_callback = opt_utils.NeptuneCallback()

   Run Optuna training passing monitor as callback::

       ...
       study = optuna.create_study(direction='maximize')
       study.optimize(objective, n_trials=100, callbacks=[neptune_callback])
       opt_utils.log_study_info(study)

   You can explore an example experiment in Neptune:
   https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-1016/artifacts


.. function:: log_study(study, experiment=None, log_charts=True, params=None)


.. function:: NeptuneMonitor(experiment=None)



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