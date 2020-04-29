.. _query-api:

Fetching Data From Neptune
--------------------------

In addition to uploading experiment data to Neptune, neptune-client can be used to fetch data directly from Neptune for further processing, such as for custom analysis.

`Study the example <https://ui.neptune.ai/USERNAME/example-project/n/Experiments-analysis-with-Query-API-and-Seaborn-31510158-04e2-47a5-a823-1cd97a0d8fcd/91350522-2b98-482d-bc14-a6ff5c061b6b>`_.

On the Project Level
====================

On the level of project, you can fetch a list of :class:`~neptune.experiments.Experiment` objects,
fetch the entire experiments view as a Pandas DataFrame and get all members of the project.

The following methods are provided:

* :meth:`~neptune.projects.Project.get_experiments`: Gets a list of experiments matching the specified criteria.
* :meth:`~neptune.projects.Project.get_leaderboard`: Gets the entire Neptune experiment view in the dashboard as a Pandas DataFrame.
* :meth:`~neptune.projects.Project.get_members`: Gets a list of project members.

On the Experiment Level
=======================

You can fetch multiple types of tracked :class:`~neptune.experiments.Experiment` objects, ranging from numeric log values to tags.

The following methods are provided:

* :meth:`~neptune.experiments.Experiment.get_hardware_utilization`: Gets GPU, CPU and memory utilization data.
* :meth:`~neptune.experiments.Experiment.get_logs`: Gets all log names with their most recent values for this experiment.
* :meth:`~neptune.experiments.Experiment.get_numeric_channels_values`: Gets values of specified metrics (numeric logs).
* :meth:`~neptune.experiments.Experiment.get_parameters`: Gets parameters for this experiment.
* :meth:`~neptune.experiments.Experiment.get_properties`: Gets user-defined properties for this experiment.
* :meth:`~neptune.experiments.Experiment.get_system_properties`: Gets experiment properties.
* :meth:`~neptune.experiments.Experiment.get_tags`: Gets the tags associated with this experiment.