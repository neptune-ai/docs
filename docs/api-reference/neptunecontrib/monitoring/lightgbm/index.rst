

:mod:`neptunecontrib.monitoring.lightgbm`
=========================================

.. py:module:: neptunecontrib.monitoring.lightgbm


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   neptunecontrib.monitoring.lightgbm.neptune_monitor


.. function:: neptune_monitor(experiment=None, prefix='')

   Logs lightGBM learning curves to Neptune.

   Goes over the list of metrics and valid_sets passed to the `lgb.train`
   object and logs them to a separate channels. For example with 'objective': 'multiclass'
   and `valid_names=['train','valid']` there will be 2 channels created:
   `train_multiclass_logloss` and `valid_multiclass_logloss`.

   :param experiment: Neptune experiment.
   :type experiment: `neptune.experiments.Experiment`
   :param prefix: Prefix that should be added before the `metric_name`
                  and `valid_name` before logging to the appropriate channel.
   :type prefix: str

   :returns:

             Callback function that should be passed to the `callbacks` parameter of
                the `lgb.train` function.
   :rtype: `func`

   .. rubric:: Examples

   Prepare dataset::

       import lightgbm as lgb
       from sklearn.model_selection import train_test_split
       from sklearn.datasets import load_wine
       data = load_wine()
       X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.1)
       lgb_train = lgb.Dataset(X_train, y_train)
       lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

   Define model parameters::

       params = {'boosting_type': 'gbdt',
                 'objective': 'multiclass',
                 'num_class': 3,
                 'num_leaves': 31,
                 'learning_rate': 0.05,
                 'feature_fraction': 0.9
                 }

   Define your Neptune monitor::

       monitor = neptune_monitor()

   Run `lgb.train` passing `neptune_monitor()` to the `callbacks` parameter::

       gbm = lgb.train(params,
                       lgb_train,
                       num_boost_round=500,
                       valid_sets=[lgb_train, lgb_eval],
                       valid_names=['train','valid'],
                       callbacks=[monitor],
                      )

   .. note::

      If you are running a k-fold validation it is a good idea to add the k-fold prefix
      and pass it to the `neptune_monitor` function::
      
          prefix='fold{}_'.format(fold_id)
          monitor = neptune_monitor(prefix)



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