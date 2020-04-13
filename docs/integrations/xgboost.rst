Automatic logging of the XGBoost training
=========================================
.. image:: ../_static/images/xgboost/xgboost_0.png
   :target: ../_static/images/xgboost/xgboost_0.png
   :alt: XGBoost overview

XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. Integration with |Neptune| lets you log multiple training artifacts out-of-the-box. It is implemented as XGBoost callback and includes:

* |metrics| (train and eval) after each boosting iteration.
* |model| (Booster) to Neptune after last boosting iteration.
* |feature| to Neptune as image after last boosting iteration.
* |tree| to Neptune as images after last boosting iteration.

.. tip:: Try integration right away with this |google-colab|.

Prerequisites
-------------
This integration makes use of the XGBoost library and is part of the |neptune-contrib|.

Make sure you have all dependencies installed. You can use bash command below:

.. code-block:: bash

    pip install 'neptune-contrib[monitoring]>=0.18.4'

Basic example
-------------
Make sure you created an experiment before you start XGBoost training. Use |create-experiment|.

Here is how to use Neptune-XGBoost integration:

.. code-block:: python3

    import neptune
    ...
    # here you import `neptune_callback` that does the magic (the open source magic :)
    from neptunecontrib.monitoring.xgboost_monitor import neptune_callback

    ...

    # Use neptune callback
    neptune.create_experiment(name='xgb', tags=['train'], params=params)
    xgb.train(params, dtrain, num_round, watchlist,
              callbacks=[neptune_callback()])  # neptune_callback is here

Example results: https://ui.neptune.ai/o/shared/org/XGBoost-integration/e/XGB-41/charts

Logged metrics
^^^^^^^^^^^^^^
They are logged for train and eval (or whatever you defined in watchlist) after each boosting iteration.

.. image:: ../_static/images/xgboost/xgboost_metrics.png
   :target: ../_static/images/xgboost/xgboost_metrics.png
   :alt: XGBoost overview

Logged model
^^^^^^^^^^^^
The model (Booster) is logged to Neptune after last boosting iteration. If you run cross validation, you get model for each fold.

.. image:: ../_static/images/xgboost/xgboost_model.png
   :target: ../_static/images/xgboost/xgboost_model.png
   :alt: XGBoost overview

Logged feature importance
^^^^^^^^^^^^^^^^^^^^^^^^^
Very useful chart showing feature importance is logged to Neptune as image after last boosting iteration. If you run cross validation, you get feature importance chart for each folds' model.

.. image:: ../_static/images/xgboost/xgboost_importance.png
   :target: ../_static/images/xgboost/xgboost_importance.png
   :alt: XGBoost overview

Logged visualized trees
^^^^^^^^^^^^^^^^^^^^^^^
Selected trees are logged to Neptune as image after last boosting iteration. If you run cross validation, you get trees visualization for each folds' model independently.

.. image:: ../_static/images/xgboost/xgboost_trees.png
   :target: ../_static/images/xgboost/xgboost_trees.png
   :alt: XGBoost overview

Resources
---------
* Open source implementation is on |github-project|,
* Docstrings / reference documentation is |docstrings|,
* Example Neptune project: |neptune-project|.

Notebooks with examples
-----------------------
* Try integration right away with this |google-colab|.
* Notebook logged to Neptune: |xgboost-integration-demo|. Feel free to download it and try yourself.

Full script
-----------
Example result: https://ui.neptune.ai/o/shared/org/XGBoost-integration/e/XGB-64

.. code-block:: python3

    import neptune
    import pandas as pd
    import xgboost as xgb
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

    # here you import `neptune_callback` that does the magic (the open source magic :)
    from neptunecontrib.monitoring.xgboost_monitor import neptune_callback

    # Set project
    # For this demonstration, I use public user: neptuner, who has 'ANONYMOUS' token .
    # Thanks to this you can run this code as is and see results in Neptune :)
    neptune.init('shared/XGBoost-integration',
                 api_token='ANONYMOUS')

    # Data
    boston = load_boston()
    data = pd.DataFrame(boston.data)
    data.columns = boston.feature_names
    data['PRICE'] = boston.target
    X, y = data.iloc[:,:-1], data.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=102030)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Params
    params = {'max_depth': 5,
              'eta': 0.5,
              'gamma': 0.1,
              'silent': 1,
              'subsample': 1,
              'lambda': 1,
              'alpha': 0.35,
              'objective': 'reg:linear',
              'eval_metric': ['mae', 'rmse']}
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 20

    # Train model
    neptune.create_experiment(name='xgb', tags=['train'], params=params)
    xgb.train(params, dtrain, num_round, watchlist,
              callbacks=[neptune_callback(log_tree=[0,1,2])])

.. External links

.. |Neptune| raw:: html

    <a href="https://neptune.ai/" target="_blank">Neptune</a>

.. |metrics| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/XGBoost-integration/e/XGB-42/charts">Log metrics</a>

.. |model| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/XGBoost-integration/e/XGB-42/artifacts">Log model</a>

.. |feature| raw:: html

    <a href="https://ui.neptune.ai/api/leaderboard/v1/images/b15cefdc-7272-4ad8-85a9-2859c3841f6c/d53b5bb7-d75f-4d7c-bc6c-f878e66ef37f/15414e28-dde2-4c30-8dd9-4fbb2f71f22a.PNG">Log feature importance</a>

.. |tree| raw:: html

    <a href="https://ui.neptune.ai/api/leaderboard/v1/images/b15cefdc-7272-4ad8-85a9-2859c3841f6c/94dcef8f-b0a4-42a9-86df-4ea325757283/95b8c689-a2c5-47d6-bd17-4155dae1b189.PNG">Log visualized trees</a>

.. |neptune-contrib| raw:: html

    <a href="https://docs.neptune.ai/integrations/neptune-contrib.html" target="_blank">neptune-contrib</a>

.. |google-colab| raw:: html

    <a href="https://colab.research.google.com/github/neptune-ai/neptune-colab-examples/blob/master/xgboost-integration.ipynb" target="_blank">Google Colab</a>

.. |github-project| raw:: html

    <a href="https://github.com/neptune-ai/neptune-contrib/blob/master/neptunecontrib/monitoring/xgboost_monitor.py" target="_blank">GitHub</a>

.. |docstrings| raw:: html

    <a href="https://neptune-contrib.readthedocs.io/user_guide/monitoring/xgboost.html" target="_blank">here</a>

.. |neptune-project| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/XGBoost-integration/experiments" target="_blank">XGBoost-integration</a>

.. |xgboost-integration-demo| raw:: html

    <a href="https://ui.neptune.ai/shared/XGBoost-integration/n/demo-notebooks-code-8f65f556-37b8-48d9-b8e0-bde6286c749d/e6c0e2a0-994b-46ff-bb4b-ba615ff46d04" target="_blank">xgboost-integration-demo</a>

.. |create-experiment| raw:: html

    <a href="https://docs.neptune.ai/neptune-client/docs/project.html#neptune.projects.Project.create_experiment" target="_blank">neptune.create_experiment()</a>
