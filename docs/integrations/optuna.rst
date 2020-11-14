.. _integrations-optuna:

Neptune-Optuna Integration
==========================

|Run on Colab|

What will you get with this integration?
----------------------------------------

|optuna-tour|

|Optuna| is an open source hyperparameter optimization framework to automate hyperparameter search. With Neptune integration, you can:

* see the experiment as it is running,
* see charts of logged run scores,
* log the parameters tried at every run,
* log the best parameters after training,
* explore interactive optuna visualizations like plot_contour, plot_slice, plot_parallel_coordinate, optimization_history,
* save the study object.
   
.. note::

    This integration is tested with ``optuna==2.3.0`` and current latest, ``neptune-client==0.4.125`` and current latest, and ``neptune-contrib==0.24.7`` and current latest.

Where to start?
---------------
To get started with this integration, follow the :ref:`Quickstart <quickstart>` below. You can also skip the basics and take a look at the :ref:`advanced options <advanced-options>`.

If you want to try things out and focus only on the code you can either:

#. Open the Colab notebook (badge-link below) with quickstart code and run it as an anonymous user "`neptuner`" - zero setup, it just works,
#. View quickstart code as a plain Python script on |script|.

.. _quickstart:

Quickstart
----------
This quickstart will show you how to:

* Install the necessary neptune packages
* Connect Neptune to your script and create the first experiment
* Log metrics, figures, and artifacts to Neptune, and 
* Explore them in the Neptune UI.

|Run on Colab|

.. _before-you-start-basic:

Before you start
^^^^^^^^^^^^^^^^
You have ``Python 3.x`` and following libraries installed:

* ``neptune-client==0.4.125`` or newer, and ``neptune-contrib==0.24.7`` or newer: See :ref:`neptune-client installation guide <installation-neptune-client>`.
* ``optuna==2.3.0``. See |optuna-install|.

You also need minimal familiarity with Optuna. Have a look at the |optuna-guide| guide to get started.

.. code-block:: bash
	
   pip install --quiet optuna neptune-client neptune-contrib['monitoring']

Step 1: Initialize Neptune
^^^^^^^^^^^^^^^^^^^^^^^^^^
Run the code below:

.. code-block:: python3

    import neptune

    neptune.init(api_token='ANONYMOUS', project_qualified_name='shared/showroom')

.. tip::

    You can also use your personal API token. Read more about how to :ref:`securely set the Neptune API token <how-to-setup-api-token>`.

Step 2: Create an Experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Run the code below to create a Neptune experiment:

.. code-block:: python3

    neptune.create_experiment('optuna-sweep')

Open the link in a new tab. The charts will currently be empty, but keep the window open. You will be able to see live metrics once logging starts.

Step 3: Create the Neptune Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python3

   import neptunecontrib.monitoring.optuna as opt_utils

   neptune_callback = opt_utils.NeptuneCallback()

Step 4: Running Optuna with the Neptune callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Pass the ``neptune_callback`` as a callback to ``study.optimize()`` to monitor the metrics and parameters checked at each run.

.. code-block:: python3

   study = optuna.create_study(direction='maximize')
   study.optimize(objective, n_trials=100, callbacks=[neptune_callback])

Step 5: Monitor your Optuna training in Neptune
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Now you can switch to the Neptune tab which you had opened previously to watch the optimization live!

Check out this |example experiment|.

|optuna-basic-logs|

.. _advanced-options:

Advanced Options
----------------

While creating the Neptune Callback, you can set ``log_study=True`` and ``log_charts=True`` to log interactive charts from ``optuna.visualization`` and the study object itself after every iteration.

.. code-block:: python3
     
   neptune_callback = opt_utils.NeptuneCallback(log_study=True, log_charts=True)

You can also log the above information after the sweep has completed by running:

.. code-block:: python3
   
   opt_utils.log_study_info(study)

Check out this |advance experiment| with advanced logging.

|optuna-advanced-logs|

What's next
-----------

Now that you know how to integrate Neptune with Optuna, you can check:

* Other :ref:`Hyperparameter Optimization Integrations with Neptune <integrations-hyperparameter-optimization-frameworks>`
* :ref:`Downloading experiment data from Neptune <guides-download_data>`
* Other :ref:`Neptune integrations <integrations-index>`

.. External links

.. |Run on Colab| raw:: html

    <div class="run-on-colab">
        <button><a target="_blank"
                   href="https://colab.research.google.com//github/neptune-ai/neptune-examples/blob/master/integrations/optuna/docs/Neptune-Optuna.ipynb"><img
                width="50" height="50" style="margin-right:10px"
                src="https://neptune.ai/wp-content/uploads/colab_logo_120.png">Run in
            Google Colab</a></button>
        <button>
            <a target="_blank" href="https://github.com/neptune-ai/neptune-examples/blob/master/integrations/optuna/docs/Neptune-Optuna.py">
                <img width="50" height="50" style="margin-right:10px"
                     src="https://neptune.ai/wp-content/uploads/GitHub-Mark-120px-plus.png">
                View source on GitHub
            </a>
        </button>
    </div>

.. |optuna-tour| raw:: html

	<div style="position: relative; padding-bottom: 53.65126676602087%; height: 0;">
		<iframe src="https://www.loom.com/embed/42dfe0ca96674051aaf4c8b9bc6a2ced" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;">
		</iframe>
	</div>

.. |Optuna| raw:: html

    <a href="https://optuna.org/" target="_blank">Optuna</a>

.. |script| raw:: html

    <a href="https://github.com/neptune-ai/neptune-examples/blob/master/integrations/optuna/docs/Neptune-Optuna.py" target="_blank">GitHub</a>

.. |optuna-install| raw:: html

    <a href="https://optuna.readthedocs.io/en/stable/installation.html" target="_blank">Optuna installation guide</a>

.. |optuna-guide| raw:: html

   <a href="https://optuna.readthedocs.io/en/stable/tutorial/index.html" target="_blank">Optuna tutorial</a>
   	
.. |neptune-client| raw:: html

    <a href="https://github.com/neptune-ai/neptune-client" target="_blank">neptune-client</a>

.. |neptune-contrib| raw:: html

    <a href="https://github.com/neptune-ai/neptune-contrib" target="_blank">neptune-contrib</a>

.. |Neptune| raw:: html

    <a href="https://neptune.ai/register" target="_blank">Neptune</a>
	
.. |example experiment| raw:: html

    <a href="https://ui.neptune.ai/shared/showroom/e/SHOW-2081/logs" target="_blank">example experiment</a>
	
.. |optuna-basic-logs| raw:: html

	<div style="position: relative; padding-bottom: 53.65126676602087%; height: 0;">
		<iframe src="https://www.loom.com/embed/23eb837b8b284eaa85827c472044e95f" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;">
		</iframe>
	</div>

.. |advance experiment| raw:: html

	<a href="https://ui.neptune.ai/shared/showroom/e/SHOW-2084/artifacts" target="_blank">example experiment</a>
	
.. |optuna-advanced-logs| raw:: html
	
	<div style="position: relative; padding-bottom: 53.65126676602087%; height: 0;">
		<iframe src="https://www.loom.com/embed/e3116bbadf2b41b48edc44559441f95c" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;">
		</iframe>
	</div>
