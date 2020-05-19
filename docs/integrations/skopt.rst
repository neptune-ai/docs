Neptune-Scikit-Optimize Integration
===================================

This integation lets you monitor |Scikit-Optimize| (skopt) hyperparameter optimization in Neptune.

.. image:: ../_static/images/others/skopt_neptuneai.png
   :target: ../_static/images/others/skopt_neptuneai.png
   :alt: Scikit Optimize Neptune integration

Requirements
------------

Integration with the Scikit-Optimize framework is enabled as a part of the logging module so all you need to have installed is |neptune-client| and |neptune-contrib|.

.. code-block:: bash

    pip install neptune-client neptune-contrib['monitoring']


Initialize Neptune and create an experiment
-------------------------------------------

.. code-block:: python3

    import neptune

    neptune.init(api_token='ANONYMOUS',
                 project_qualified_name='shared/showroom')
    neptune.create_experiment(name='skopt sweep')


Create **NeptuneCallback**
--------------------------
Pass the experiment object as the first argument.

.. code-block:: python3

    import neptunecontrib.monitoring.skopt as sk_utils

    neptune_callback = sk_utils.NeptuneCallback()

Pass **neptune_callback** to **skopt.forest_minimize** or others
----------------------------------------------------------------
This causes the metrics, parameters and results pickle logged after every iteration.
Everything can be inspected live.

.. code-block:: python3

    results = skopt.forest_minimize(objective, space, callback=[neptune_callback],
                                    base_estimator='ET', n_calls=100, n_random_starts=10)

Log all results
---------------
You can log additional information from skopt results after the sweep has completed.
By running:

.. code-block:: python3

    sk_utils.log_results(results)

You log the following things to Neptune:

* Best score
* Best parameters
* Figures from plots module: plot_evaluations, plot_convergence, plot_objective, and plot_regret
* Pickled results object

.. code-block:: python3

    sk_utils.log_results(results)

Monitor your Scikit-Optimize training in Neptune
------------------------------------------------
Now you can watch your Scikit-Optimize hyperparameter optimization in Neptune!

Check out this |example experiment|.

.. image:: ../_static/images/skopt/skopt_monitoring.gif
   :target: ../_static/images/skopt/skopt_monitoring.gif
   :alt: Scikit-Optimize monitoring in Neptune

.. External links

.. |Scikit-Optimize| raw:: html

    <a href="https://scikit-optimize.github.io/stable/" target="_blank">Scikit-Optimize</a>

.. |example experiment| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-1061/logs" target="_blank">example experiment</a>

.. |neptune-client| raw:: html

    <a href="https://github.com/neptune-ai/neptune-client" target="_blank">neptune-client</a>

.. |neptune-contrib| raw:: html

    <a href="https://github.com/neptune-ai/neptune-contrib" target="_blank">neptune-contrib</a>