.. _integration-xgboost:

Neptune-XGBoost Integration
===========================

|Youtube Video|

XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. The integration with Neptune lets you log multiple training artifacts with no further customization.

.. image:: ../_static/images/integrations/xgboost_0.png
   :target: ../_static/images/integrations/xgboost_0.png
   :alt: XGBoost overview


The integration is implemented as an XGBoost callback and provides the following capabilities:

* |metrics| (train and eval) after each boosting iteration.
* |model| (Booster) to Neptune after the last boosting iteration.
* |feature| to Neptune as an image after the last boosting iteration.
* |tree| to Neptune as images after the last boosting iteration.

.. tip:: Try the integration right away with this |google-colab|.

Requirements
------------
This integration makes use of the XGBoost library and is part of :ref:`neptune-contrib <integration-neptune-contrib>`.

Make sure you have all dependencies installed. You can use the bash command below:

.. code-block:: bash

    pip install 'neptune-contrib[monitoring]>=0.18.4'

Basic example
-------------
Make sure you have created an experiment before you start XGBoost training. Use the :meth:`~neptune.projects.Project.create_experiment` method to do this.

Here is how to use the Neptune-XGBoost integration:

.. code-block:: python3

    import neptune
    ...
    # here you import `neptune_callback` that does the magic (the open source magic :)
    from neptunecontrib.monitoring.xgboost import neptune_callback

    ...

    # Use neptune callback
    neptune.create_experiment(name='xgb', tags=['train'], params=params)
    xgb.train(params, dtrain, num_round, watchlist,
              callbacks=[neptune_callback()])  # neptune_callback is here

|Example results|

Logged metrics
^^^^^^^^^^^^^^
These are logged for train and eval (or whatever you defined in the watchlist) after each boosting iteration.

.. image:: ../_static/images/integrations/xgboost_metrics.png
   :target: ../_static/images/integrations/xgboost_metrics.png
   :alt: XGBoost overview

Logged model
^^^^^^^^^^^^
The model (Booster) is logged to Neptune after the last boosting iteration. If you run cross-validation, you get a model for each fold.

.. image:: ../_static/images/integrations/xgboost_model.png
   :target: ../_static/images/integrations/xgboost_model.png
   :alt: XGBoost overview

Logged feature importance
^^^^^^^^^^^^^^^^^^^^^^^^^
This is a very useful chart, as it shows feature importance. It is logged to Neptune as an image after the last boosting iteration. If you run cross-validation, you get a feature importance chart for each fold's model.

.. image:: ../_static/images/integrations/xgboost_importance.png
   :target: ../_static/images/integrations/xgboost_importance.png
   :alt: XGBoost overview

Logged visualized trees
^^^^^^^^^^^^^^^^^^^^^^^
Selected trees are logged to Neptune as an image after the last boosting iteration. If you run cross-validation, you get a tree visualization for each fold's model, independently.

.. image:: ../_static/images/integrations/xgboost_trees.png
   :target: ../_static/images/integrations/xgboost_trees.png
   :alt: XGBoost overview

Resources
---------
* Open source implementation is on |github-project|.
* Example Neptune project: |neptune-project|.

Notebooks with examples
-----------------------
* Try the integration right away with this |google-colab|.
* Notebook logged to Neptune: |xgboost-integration-demo|. Feel free to download it and try it yourself.

Full script
-----------

|Example results|

.. code-block:: python3

    import neptune
    import pandas as pd
    import xgboost as xgb
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

    # here you import `neptune_callback` that does the magic (the open source magic :)
    from neptunecontrib.monitoring.xgboost import neptune_callback

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

    <a href="https://ui.neptune.ai/o/shared/org/XGBoost-integration/e/XGB-42/charts" target="_blank">Log metrics</a>

.. |model| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/XGBoost-integration/e/XGB-42/artifacts" target="_blank">Log model</a>

.. |feature| raw:: html

    <a href="https://ui.neptune.ai/api/leaderboard/v1/images/b15cefdc-7272-4ad8-85a9-2859c3841f6c/d53b5bb7-d75f-4d7c-bc6c-f878e66ef37f/15414e28-dde2-4c30-8dd9-4fbb2f71f22a.PNG" target="_blank">Log feature importance</a>

.. |tree| raw:: html

    <a href="https://ui.neptune.ai/api/leaderboard/v1/images/b15cefdc-7272-4ad8-85a9-2859c3841f6c/94dcef8f-b0a4-42a9-86df-4ea325757283/95b8c689-a2c5-47d6-bd17-4155dae1b189.PNG" target="_blank">Log visualized trees</a>

.. |google-colab| raw:: html

    <a href="https://colab.research.google.com/github/neptune-ai/neptune-colab-examples/blob/master/xgboost-integration.ipynb" target="_blank">Google Colab</a>

.. |github-project| raw:: html

    <a href="https://github.com/neptune-ai/neptune-contrib/blob/master/neptunecontrib/monitoring/xgboost_monitor.py" target="_blank">GitHub</a>

.. |neptune-project| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/XGBoost-integration/experiments" target="_blank">XGBoost-integration</a>

.. |xgboost-integration-demo| raw:: html

    <a href="https://ui.neptune.ai/shared/XGBoost-integration/n/demo-notebooks-code-8f65f556-37b8-48d9-b8e0-bde6286c749d/e6c0e2a0-994b-46ff-bb4b-ba615ff46d04" target="_blank">xgboost-integration-demo</a>

.. |Example results| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/XGBoost-integration/e/XGB-41/charts" target="_blank">Example results</a>


.. |Youtube Video| raw:: html

    <iframe width="720" height="420" src="https://www.youtube.com/embed/xc5gsJvf5Wo" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
