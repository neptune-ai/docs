:mod:`neptune_mlflow.data_loader`
=================================

.. py:module:: neptune_mlflow.data_loader


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   neptune_mlflow.data_loader.DataLoader



.. py:class:: DataLoader(project, path)

   Bases: :class:`object`

   .. attribute:: MLFLOW_EXPERIMENT_ID_PROPERTY
      :annotation: = mlflow/experiment/id

      

   .. attribute:: MLFLOW_EXPERIMENT_NAME_PROPERTY
      :annotation: = mlflow/experiment/name

      

   .. attribute:: MLFLOW_RUN_ID_PROPERTY
      :annotation: = mlflow/run/uuid

      

   .. attribute:: MLFLOW_RUN_NAME_PROPERTY
      :annotation: = mlflow/run/name

      

   .. method:: run(self)


   .. method:: _create_neptune_experiment(self, experiment, run)


   .. staticmethod:: _create_metric(neptune_exp, experiment, run, metric_key)


   .. staticmethod:: _get_params(run)


   .. staticmethod:: _get_properties(experiment, run)


   .. staticmethod:: _get_tags(experiment, run)


   .. staticmethod:: _to_proper_tag(string)


   .. staticmethod:: _get_metric_file(experiment, run_info, metric_key)


   .. staticmethod:: _get_name_for_experiment(experiment)


   .. staticmethod:: _get_run_qualified_name(experiment, run_info)


   .. staticmethod:: _get_mlflow_run_name(run)



