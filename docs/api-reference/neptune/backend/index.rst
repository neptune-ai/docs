:mod:`neptune.backend`
======================

.. py:module:: neptune.backend


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   neptune.backend.Backend



.. py:class:: Backend

   Bases: :class:`object`

   .. attribute:: api_address
      

      

   .. attribute:: display_address
      

      

   .. method:: get_project(self, project_qualified_name)


   .. method:: get_projects(self, namespace)


   .. method:: get_project_members(self, project_identifier)


   .. method:: get_leaderboard_entries(self, project, entry_types, ids, states, owners, tags, min_running_time)


   .. method:: get_channel_points_csv(self, experiment, channel_internal_id)


   .. method:: get_metrics_csv(self, experiment)


   .. method:: create_experiment(self, project, name, description, params, properties, tags, abortable, monitored, git_info, hostname, entrypoint, notebook_id, checkpoint_id)


   .. method:: get_notebook(self, project, notebook_id)


   .. method:: get_last_checkpoint(self, project, notebook_id)


   .. method:: create_notebook(self, project)


   .. method:: create_checkpoint(self, notebook_id, jupyter_path, _file)


   .. method:: get_experiment(self, experiment_id)


   .. method:: update_experiment(self, experiment, properties)


   .. method:: update_tags(self, experiment, tags_to_add, tags_to_delete)


   .. method:: upload_experiment_source(self, experiment, data, progress_indicator)


   .. method:: extract_experiment_source(self, experiment, data)


   .. method:: create_channel(self, experiment, name, channel_type)


   .. method:: reset_channel(self, channel_id)


   .. method:: create_system_channel(self, experiment, name, channel_type)


   .. method:: get_system_channels(self, experiment)


   .. method:: send_channels_values(self, experiment, channels_with_values)


   .. method:: mark_succeeded(self, experiment)


   .. method:: mark_failed(self, experiment, traceback)


   .. method:: ping_experiment(self, experiment)


   .. method:: create_hardware_metric(self, experiment, metric)


   .. method:: send_hardware_metric_reports(self, experiment, metrics, metric_reports)


   .. method:: upload_experiment_output(self, experiment, data, progress_indicator)


   .. method:: extract_experiment_output(self, experiment, data)


   .. method:: download_data(self, project, path, destination)



