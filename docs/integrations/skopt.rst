Monitor Scikit Optimize hyperparameter optimization in Neptune
==============================================================
.. image:: ../_static/images/others/skopt_neptuneai.png
   :target: ../_static/images/others/skopt_neptuneai.png
   :alt: Scikit Optimize Neptune integration

Prerequisites
-------------
Integration with |Scikit Optimize| framework is introduced as a part of logging module so just need to have |neptune-client| and |neptune-contrib| installed.

.. code-block:: bash

    pip install neptune-client neptune-contrib['monitoring']


Initialize Neptune and create an experiment
-------------------------------------------

.. code-block:: python3

    import neptune
    neptune.init('jakub-czakon/blog-hpo')
    neptune.create_experiment(name='skopt sweep')


Create **NeptuneMonitor** callback
----------------------------------
Pass the experiment object as first argument.

.. note:: To be able to log information after the .fit() method finishes remember to pass ``close_after_train=False``

.. code-block:: python3

    import neptunecontrib.monitoring.skopt as sk_utils
    neptune_monitor = sk_utils.NeptuneMonitor()

Pass **neptune_monitor** to **skopt.forest_minimize** or others
---------------------------------------------------------------
It will monitor the metrics and parameters checked at each run.

.. code-block:: python3

    results = skopt.forest_minimize(objective, space, callback=[neptune_monitor],
                                    base_estimator='ET', n_calls=100, n_random_starts=10)
    sk_utils.log_results(results)

Log all results
---------------
It will log the following things to Neptune:
* best score
* best parameters
* plot_convergence figure
* plot_evaluations figure
* plot_objective figure

.. code-block:: python3

    sk_utils.log_results(results)

Monitor your Scikit Optimize training in Neptune
------------------------------------------------
Now you can watch your Scikit Optimize hyperparameter optimization in Neptune!

Check out this |example experiment|.

.. image:: ../_static/images/skopt/skopt_monitoring.gif
   :target: ../_static/images/skopt/skopt_monitoring.gif
   :alt: Scikit Optimize monitoring in Neptune

.. External links

.. |Scikit Optimize| raw:: html

    <a href="https://scikit-optimize.github.io/stable/" target="_blank">Scikit Optimize</a>

.. |example experiment| raw:: html

    <a href="https://ui.neptune.ai/jakub-czakon/blog-hpo/e/BLOG-99/logs" target="_blank">example experiment</a>

.. |neptune-client| raw:: html

    <a href="https://github.com/neptune-ai/neptune-client" target="_blank">neptune-client</a>

.. |neptune-contrib| raw:: html

    <a href="https://github.com/neptune-ai/neptune-contrib" target="_blank">neptune-contrib</a>