Neptune-LightGBM Integration
============================

This integration lets you customize training scripts written in |LightGBM| to log metrics to Neptune.


.. image:: ../_static/images/others/lightgbm_neptuneml.png
   :target: ../_static/images/others/lightgbm_neptuneml.png
   :alt: lightGBM neptune.ai integration

Say your training script looks like this:

.. code-block::

   import lightgbm as lgb
   from sklearn.model_selection import train_test_split
   from sklearn.datasets import load_wine

   data = load_wine()

   X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.1)
   lgb_train = lgb.Dataset(X_train, y_train)
   lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

   params = {
       'boosting_type': 'gbdt',
       'objective': 'multiclass',
       'num_class': 3,
       'num_leaves': 31,
       'learning_rate': 0.05,
       'feature_fraction': 0.9,
   }

   gbm = lgb.train(params,
                   lgb_train,
                   num_boost_round=500,
                   valid_sets=[lgb_train, lgb_eval],
                   valid_names=['train','valid'],
                   )

Add LightGBM callbacks to pass log metrics to Neptune, so:

1. Take this callback:

.. code-block::

   import neptune

   neptune.init('shared/onboarding')
   neptune.create_experiment()

   def neptune_monitor():
       def callback(env):
           for name, loss_name, loss_value, _ in env.evaluation_result_list:
               neptune.send_metric('{}_{}'.format(name, loss_name), x=env.iteration, y=loss_value)
       return callback

2. Pass it to ``lgb.train`` object using the ``callbacks`` parameter:

.. code-block::

   gbm = lgb.train(params,
                   lgb_train,
                   num_boost_round=500,
                   valid_sets=[lgb_train, lgb_eval],
                   valid_names=['train','valid'],
                   callbacks=[neptune_monitor()],
                   )

All your metrics are now logged to Neptune:

.. image:: ../_static/images/how-to/ht-log-lightgbm-1.png
   :target: ../_static/images/how-to/ht-log-lightgbm-1.png
   :alt: image

.. External links

.. |LightGBM| raw:: html

   <a href="https://lightgbm.readthedocs.io/en/latest/#" target="_blank">LightGBM</a>