:mod:`neptunecontrib.api.explainers`
====================================

.. py:module:: neptunecontrib.api.explainers


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   neptunecontrib.api.explainers.log_explainer
   neptunecontrib.api.explainers.log_local_explanations
   neptunecontrib.api.explainers.log_global_explanations


.. function:: log_explainer(filename, explainer, experiment=None)

   Logs dalex explainer to Neptune.

   Dalex explainer is pickled and logged to Neptune.

   :param filename: filename that will be used as an artifact's destination.
   :type filename: :obj:`str`
   :param explainer: an instance of dalex explainer
   :type explainer: :obj:`dalex.Explainer`
   :param experiment:
                      | For advanced users only. Pass Neptune
                        `Experiment <https://docs.neptune.ai/neptune-client/docs/experiment.html#neptune.experiments.Experiment>`_
                        object if you want to control to which experiment data is logged.
                      | If ``None``, log to currently active, and most recent experiment.
   :type experiment: :obj:`neptune.experiments.Experiment`, optional, default is ``None``

   .. rubric:: Examples

   Start an experiment::

       import neptune

       neptune.init(api_token='ANONYMOUS',
                    project_qualified_name='shared/dalex-integration')
       neptune.create_experiment(name='logging explanations')

   Train your model and create dalex explainer::

       ...
       clf.fit(X, y)

       expl = dx.Explainer(clf, X, y, label="Titanic MLP Pipeline")

       log_explainer('explainer.pkl', expl)

   .. note::

      Check out how the logged explainer looks in Neptune:
      `example experiment <https://ui.neptune.ai/o/shared/org/dalex-integration/e/DAL-48/artifacts>`_


.. function:: log_local_explanations(explainer, observation, experiment=None)

   Logs local explanations from dalex to Neptune.

   Dalex explanations are converted to interactive HTML objects and then uploaded to Neptune
   as an artifact with path charts/{name}.html.

   The following explanations are logged: break down, break down with interactions, shap, ceteris paribus,
   and ceteris paribus for categorical variables. Explanation charts are created and logged with default settings.
   To log charts with custom settings, create a custom chart and use `neptunecontrib.api.log_chart`.
   For more information about Dalex go to `Dalex Website <https://modeloriented.github.io/DALEX/>`_.

   :param explainer: an instance of dalex explainer
   :type explainer: :obj:`dalex.Explainer`
   :param observation (: obj): an observation that can be fed to the classifier for which the explainer was created
   :param experiment:
                      | For advanced users only. Pass Neptune
                        `Experiment <https://docs.neptune.ai/neptune-client/docs/experiment.html#neptune.experiments.Experiment>`_
                        object if you want to control to which experiment data is logged.
                      | If ``None``, log to currently active, and most recent experiment.
   :type experiment: :obj:`neptune.experiments.Experiment`, optional, default is ``None``

   .. rubric:: Examples

   Start an experiment::

       import neptune

       neptune.init(api_token='ANONYMOUS',
                    project_qualified_name='shared/dalex-integration')
       neptune.create_experiment(name='logging explanations')

   Train your model and create dalex explainer::

       ...
       clf.fit(X, y)

       expl = dx.Explainer(clf, X, y, label="Titanic MLP Pipeline")

       new_observation = pd.DataFrame({'gender': ['male'],
                                       'age': [25],
                                       'class': ['1st'],
                                       'embarked': ['Southampton'],
                                       'fare': [72],
                                       'sibsp': [0],
                                       'parch': 0},
                                      index=['John'])

       log_local_explanations(expl, new_observation)

   .. note::

      Check out how the logged explanations look in Neptune:
      `example experiment <https://ui.neptune.ai/o/shared/org/dalex-integration/e/DAL-48/artifacts?path=charts%2F>`_


.. function:: log_global_explanations(explainer, categorical_features=None, numerical_features=None, experiment=None)

   Logs global explanations from dalex to Neptune.

   Dalex explanations are converted to interactive HTML objects and then uploaded to Neptune
   as an artifact with path charts/{name}.html.

   The following explanations are logged: variable importance. If categorical features are specified partial dependence
   and accumulated dependence are also logged. Explanation charts are created and logged with default settings.
   To log charts with custom settings, create a custom chart and use `neptunecontrib.api.log_chart`.
   For more information about Dalex go to `Dalex Website <https://modeloriented.github.io/DALEX/>`_.

   :param explainer: an instance of dalex explainer
   :type explainer: :obj:`dalex.Explainer`
   :param categorical_features (: list): list of categorical features for which you want to create
                                  accumulated dependence plots.
   :param numerical_features (: list): list of numerical features for which you want to create
                                partial dependence plots.
   :param experiment:
                      | For advanced users only. Pass Neptune
                        `Experiment <https://docs.neptune.ai/neptune-client/docs/experiment.html#neptune.experiments.Experiment>`_
                        object if you want to control to which experiment data is logged.
                      | If ``None``, log to currently active, and most recent experiment.
   :type experiment: :obj:`neptune.experiments.Experiment`, optional, default is ``None``

   .. rubric:: Examples

   Start an experiment::

       import neptune

       neptune.init(api_token='ANONYMOUS',
                    project_qualified_name='shared/dalex-integration')
       neptune.create_experiment(name='logging explanations')

   Train your model and create dalex explainer::

       ...
       clf.fit(X, y)

       expl = dx.Explainer(clf, X, y, label="Titanic MLP Pipeline")
       log_global_explanations(expl, categorical_features=["gender", "class"], numerical_features=["age", "fare"])

   .. note::

      Check out how the logged explanations look in Neptune:
      `example experiment <https://ui.neptune.ai/o/shared/org/dalex-integration/e/DAL-48/artifacts?path=charts%2F>`_


