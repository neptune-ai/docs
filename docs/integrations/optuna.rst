.. _integrations-optuna:

Neptune-Optuna Integration
==========================

|Optuna| is an open source hyperparameter optimization framework to automate hyperparameter search. 

Neptune's Optuna integration enables you to monitor |Optuna| hyperparameter optimization on the Neptune platform.

Introduction
------------
This guide will show you how to:

* Install the necessary neptune packages
* Connect Neptune to your script and create the first experiment
* Log metrics, figures, and artifacts to Neptune, and 
* Explore them in the Neptune UI.

Requirements
------------
To use Neptune's Optuna integration, you need to install the |optuna_package|, |neptune-client| and |neptune-contrib| packages.

.. code-block:: bash
	
   pip install --quiet optuna
   pip install --quiet neptune-client neptune-contrib['monitoring']


Quickstart
----------

Step 1: Initialize Neptune
^^^^^^^^^^^^^^^^^^^^^^^^^^
Neptune gives you an option of logging data under a public folder as an anonymous user. 
This is great when you are just trying out the application and don't have a Neptune account yet.  

Run the code below:

.. code-block:: python3

    import neptune

    neptune.init(api_token='ANONYMOUS', project_qualified_name='shared/showroom')

.. seealso::

    Instead of logging data to the public project 'shared/onboarding' as an anonymous user 'neptuner' you can log it to your own project.
    
    #. Create an account with |Neptune|
    #. Get your Neptune API token
    
       .. image:: ../../_static/images/getting-started/quick-starts/get_token.gif
            :target: ../../_static/images/getting-started/quick-starts/get_token.gif
            :alt: Get API token
          
    #. Pass the token to ``api_token`` argument of ``neptune.init()``
    #. Pass your username to the ``project_qualified_name`` argument of the ``neptune.init()`` method: ``project_qualified_name='YOUR_USERNAME/optuna-sweep``.
    
       .. code:: python
          
          neptune.init(project_qualified_name='YOUR_USERNAME/optuna-sweep', api_token='YOUR_API_TOKEN',)    

Step 2: Create an Experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Run the code below to create a Neptune experiment:

.. code-block:: python3

    neptune.create_experiment('optuna-sweep')

Open the link in a new tab. The charts will currently be empty, but keep the window open. You will be able to see live metrics once logging starts.

Step 3: Create the Neptune Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note:: You can pass the experiment object as the first argument if you want to do it explicitly.

.. code-block:: python3

   import neptunecontrib.monitoring.optuna as opt_utils

   neptune_callback = opt_utils.NeptuneCallback()

Step 4: Pass the Neptune Callback as a callback to ``study.optimize``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Monitor the metrics and parameters checked at each run:

.. code-block:: python3

   study = optuna.create_study(direction='maximize')
   study.optimize(objective, n_trials=100, callbacks=[neptune_callback])

Step 5: Monitor your Optuna training in Neptune
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Now you can switch to the Neptune tab which you had opened previously to watch the optimization live!

Check out this |example experiment|.

.. image:: ../_static/images/integrations/optuna_monitoring.gif
   :target: ../_static/images/integrations/optuna_monitoring.gif
   :alt: Optuna monitoring in Neptune
   
Advanced Logging
----------------
* You can log interactive charts from optuna.visualization and the study object after every iteration.
  Simply specify what you want to log during the ``NeptuneCallback`` initialization.

  .. code-block:: python3
     
     neptune_callback = opt_utils.NeptuneCallback(log_study=True, log_charts=True)

* You can also log additional information from optuna study after the sweep has completed by running:

  .. code-block:: python3
     
     opt_utils.log_study_info(study)
  
  You log the following things to Neptune:  
  * Best score
  * Best parameters
  * Interactive plotly figures from optuna.visualization: plot_contour, plot_slice, plot_parallel_coordinate, optimization_history
  * Pickled study object
  
  .. image:: ../_static/images/integrations/optuna_charts.gif
     :target: ../_static/images/integrations/optuna_charts.gif
     :alt: Optuna charts in Neptune


Sample Script
-------------

|Run on Colab|

.. code-block:: python3
   
   # Importing packages
   import lightgbm as lgb
   import neptune
   import neptunecontrib.monitoring.optuna as opt_utils
   import optuna
   from sklearn.datasets import load_breast_cancer
   from sklearn.metrics import roc_auc_score
   from sklearn.model_selection import train_test_split
   
   # Initializing Neptune
   neptune.init(api_token='ANONYMOUS', project_qualified_name='shared/showroom')
   neptune.create_experiment('optuna-sweep')
   neptune_callback = opt_utils.NeptuneCallback(log_study=True, log_charts=True)
   
   # Sample Optuna objective function
   def objective(trial):
      
      data, target = load_breast_cancer(return_X_y=True)
      train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.25)
      dtrain = lgb.Dataset(train_x, label=train_y)
   
      param = {
         'objective': 'binary',
         'metric': 'binary_logloss',
         'num_leaves': trial.suggest_int('num_leaves', 2, 256),
         'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
         'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
         'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
      }
   
      gbm = lgb.train(param, dtrain)
      preds = gbm.predict(test_x)
      accuracy = roc_auc_score(test_y, preds)
      
      return accuracy
  
   # Running Optuna and logging metrics
   study = optuna.create_study(direction='maximize')
   study.optimize(objective, n_trials=5, callbacks=[neptune_callback])
   opt_utils.log_study(study)
   
   # Stopping the experiment
   neptune.stop()

What's next
-----------

Now that you know how to integrate Neptune with Optuna, you can check:

* Other :ref:`Hyperparameter Optimization Integrations with Neptune <integrations-hyperparameter-optimization-frameworks>`
* :ref:`Downloading experiment data from Neptune <guides-download_data>`
* Other :ref:`Neptune integrations <integrations-index>`

.. External links

.. |Optuna| raw:: html

    <a href="https://optuna.org/" target="_blank">Optuna</a>

.. |optuna_package| raw:: html

    <a href="https://optuna.readthedocs.io/en/stable/installation.html" target="_blank">Optuna</a>
	
.. |neptune-client| raw:: html

    <a href="https://github.com/neptune-ai/neptune-client" target="_blank">neptune-client</a>

.. |neptune-contrib| raw:: html

    <a href="https://github.com/neptune-ai/neptune-contrib" target="_blank">neptune-contrib</a>

.. |Neptune| raw:: html

    <a href="https://neptune.ai/register" target="_blank">Neptune</a>
	
.. |example experiment| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-1018/artifacts" target="_blank">example experiment</a>
    
.. |Run on Colab| raw:: html

    <a href="https://colab.research.google.com/drive/1coMSk2w5iratCN6kYPjx8YK19VKCrMZo?usp=sharing" target="_blank">
        <img width="200" height="200"src="https://colab.research.google.com/assets/colab-badge.svg"></img>
    </a>