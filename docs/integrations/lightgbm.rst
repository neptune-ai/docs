Neptune-LightGBM Integration
============================

|lightGBM| is a popular gradient boosting library. The integration with Neptune lets you log training and evaluation metrics and have them visualized in Neptune.

.. image:: ../_static/images/integrations/lightgbm_neptuneml.png
   :target: ../_static/images/integrations/lightgbm_neptuneml.png
   :alt: lightGBM neptune.ai integration

Requirements
------------
To use Neptune + lightGBM integration you need to have installed is |neptune-client| and |neptune-contrib|.

.. code-block:: bash

    pip install neptune-client neptune-contrib[monitoring]

Initialize Neptune and create an experiment
-------------------------------------------

.. code-block:: python3

    import neptune

    neptune.init(api_token='ANONYMOUS',
                 project_qualified_name='shared/showroom')
    neptune.create_experiment(name='lightGBM-training')

Pass **neptune_monitor** to **lgb.train**
-----------------------------------------
Simply pass ``neptune_monitor`` to the callbacks argument of ``lgb.train``

.. code-block:: python3

    from neptunecontrib.monitoring.lightgbm import neptune_monitor

    gbm = lgb.train(params,
            lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_train, lgb_eval],
            valid_names=['train','valid'],
            callbacks=[neptune_monitor()], # Just add this callback
           )

Monitor your lightGBM training in Neptune
-----------------------------------------
Now you can watch your lightGBM training in Neptune!

Check out this |example experiment|.

.. image:: ../_static/images/integrations/lightgbm_neptuneml.png
   :target: ../_static/images/integrations/lightgbm_neptuneml.png
   :alt: lightGBM neptune.ai integration

Full script
-----------

.. code-block:: python3

    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_wine
    import neptune
    from neptunecontrib.monitoring.lightgbm import neptune_monitor

    neptune.init(api_token='ANONYMOUS', project_qualified_name='shared/showroom')

    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.1)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {'boosting_type': 'gbdt',
                  'objective': 'multiclass',
                  'num_class': 3,
                  'num_leaves': 31,
                  'learning_rate': 0.05,
                  'feature_fraction': 0.9
                  }

    neptune.create_experiment('lightGBM-integration')

    gbm = lgb.train(params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=['train','valid'],
        callbacks=[neptune_monitor()],
       )

.. |lightGBM| raw:: html

    <a href="https://lightgbm.readthedocs.io/en/latest/" target="_blank">lightGBM</a>

.. |example experiment| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-1093" target="_blank">example experiment</a>

.. |neptune-client| raw:: html

    <a href="https://github.com/neptune-ai/neptune-client" target="_blank">neptune-client</a>

.. |neptune-contrib| raw:: html

    <a href="https://github.com/neptune-ai/neptune-contrib" target="_blank">neptune-contrib</a>
