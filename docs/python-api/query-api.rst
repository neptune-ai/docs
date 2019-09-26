.. _query-api:

Query API
=========
Neptune client allows you to fetch data directly form Neptune. This let users make custom analysis of their data.

Project-level
-------------
On the level of project you can fetch list of :class:`~neptune.experiments.Experiment` objects,
fetch experiments view as Pandas DataFrame and get all members of the project.

* :meth:`~neptune.projects.Project.get_experiments`
* :meth:`~neptune.projects.Project.get_leaderboard`
* :meth:`~neptune.projects.Project.get_members`

Experiment-level
----------------
On the level of :class:`~neptune.experiments.Experiment`, you can fetch multiple types of tracked objects, ranging from numeric logs values to tags.

* :meth:`~neptune.experiments.Experiment.get_hardware_utilization`
* :meth:`~neptune.experiments.Experiment.get_logs`
* :meth:`~neptune.experiments.Experiment.get_numeric_channels_values`
* :meth:`~neptune.experiments.Experiment.get_parameters`
* :meth:`~neptune.experiments.Experiment.get_properties`
* :meth:`~neptune.experiments.Experiment.get_system_properties`
* :meth:`~neptune.experiments.Experiment.get_tags`
