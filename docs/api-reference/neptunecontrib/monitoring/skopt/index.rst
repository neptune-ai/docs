

:mod:`neptunecontrib.monitoring.skopt`
======================================

.. py:module:: neptunecontrib.monitoring.skopt


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   neptunecontrib.monitoring.skopt.NeptuneCallback



Functions
~~~~~~~~~

.. autoapisummary::

   neptunecontrib.monitoring.skopt.log_results
   neptunecontrib.monitoring.skopt.NeptuneMonitor
   neptunecontrib.monitoring.skopt._log_best_parameters
   neptunecontrib.monitoring.skopt._log_best_score
   neptunecontrib.monitoring.skopt._log_plot_convergence
   neptunecontrib.monitoring.skopt._log_plot_regret
   neptunecontrib.monitoring.skopt._log_plot_evaluations
   neptunecontrib.monitoring.skopt._log_plot_objective
   neptunecontrib.monitoring.skopt._log_results_object
   neptunecontrib.monitoring.skopt._export_results_object
   neptunecontrib.monitoring.skopt._format_to_named_params


.. py:class:: NeptuneCallback(experiment=None, log_checkpoint=True)

   Logs hyperparameter optimization process to Neptune.

   Specifically using NeptuneCallback will log: run metrics and run parameters, best run metrics so far, and
   the current results checkpoint.

   .. rubric:: Examples

   Initialize NeptuneCallback::

       import neptune
       import neptunecontrib.monitoring.skopt as sk_utils

       neptune.init(api_token='ANONYMOUS',
                    project_qualified_name='shared/showroom')

       neptune.create_experiment(name='optuna sweep')

       neptune_callback = sk_utils.NeptuneCallback()

   Run skopt training passing neptune_callback as a callback::

       ...
       results = skopt.forest_minimize(objective, space, callback=[neptune_callback],
                           base_estimator='ET', n_calls=100, n_random_starts=10)

   You can explore an example experiment in Neptune:
   https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-1065/logs

   .. method:: __call__(self, res)


   .. staticmethod:: _get_last_params(res)



.. function:: log_results(results, experiment=None, log_plots=True, log_pickle=True)

   Logs runs results and parameters to neptune.

   Logs all hyperparameter optimization results to Neptune. Those include best score ('best_score' metric),
   best parameters ('best_parameters' property), convergence plot ('diagnostics' log),
   evaluations plot ('diagnostics' log), and objective plot ('diagnostics' log).

    Args:
        results('scipy.optimize.OptimizeResult'): Results object that is typically an
            output of the function like `skopt.forest_minimize(...)`
        experiment(`neptune.experiments.Experiment`): Neptune experiment. Default is None.
       log_plots: ('bool'): If True skopt plots will be logged to Neptune.
       log_pickle: ('bool'): if True pickled skopt results object will be logged to Neptune.

    Examples:
        Run skopt training::

            ...
            results = skopt.forest_minimize(objective, space,
                                base_estimator='ET', n_calls=100, n_random_starts=10)

        Initialize Neptune::

           import neptune

           neptune.init(api_token='ANONYMOUS',
                        project_qualified_name='shared/showroom')
           neptune.create_experiment(name='optuna sweep')

        Send best parameters to Neptune::

            import neptunecontrib.monitoring.skopt as sk_utils

            sk_utils.log_results(results)

       You can explore an example experiment in Neptune:
       https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-1065/logs


.. function:: NeptuneMonitor(*args, **kwargs)


.. function:: _log_best_parameters(results, experiment)


.. function:: _log_best_score(results, experiment)


.. function:: _log_plot_convergence(results, experiment, name='diagnostics')


.. function:: _log_plot_regret(results, experiment, name='diagnostics')


.. function:: _log_plot_evaluations(results, experiment, name='diagnostics')


.. function:: _log_plot_objective(results, experiment, name='diagnostics')


.. function:: _log_results_object(results, experiment=None)


.. function:: _export_results_object(results)


.. function:: _format_to_named_params(params, result)



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