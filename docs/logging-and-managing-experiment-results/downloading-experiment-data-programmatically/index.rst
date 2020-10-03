.. _guides-download_data:

Downloading experiment data programmatically
============================================

|Youtube Video|

In addition to uploading experiment data to Neptune, neptune-client can be used to fetch data directly from Neptune for further processing, such as for custom analysis.

This page presents the methods in the that enable you to download project and experiment data.

You may want to check out |this example| project for more information.

Getting Project Data
--------------------

On the level of project, you can fetch a list of :class:`neptune.projects.Project` objects,
fetch the entire experiments view as a Pandas DataFrame and get all members of the project.

The following methods are provided:

* :meth:`~neptune.projects.Project.get_experiments`: Gets a list of experiments matching the specified criteria.
* :meth:`~neptune.projects.Project.get_leaderboard`: Gets the entire Neptune experiment view in the dashboard as a Pandas DataFrame.
* :meth:`~neptune.projects.Project.get_members`: Gets a list of project members.

Getting Experiment Data
-----------------------

You can fetch multiple types of tracked :class:`~neptune.experiments.Experiment` objects, ranging from numeric log values to tags.

The following methods are provided:

* :meth:`~neptune.experiments.Experiment.get_hardware_utilization`: Gets GPU, CPU and memory utilization data.
* :meth:`~neptune.experiments.Experiment.get_logs`: Gets all log names with their most recent values for this experiment.
* :meth:`~neptune.experiments.Experiment.get_numeric_channels_values`: Gets values of specified metrics (numeric logs).
* :meth:`~neptune.experiments.Experiment.get_parameters`: Gets parameters for this experiment.
* :meth:`~neptune.experiments.Experiment.get_properties`: Gets user-defined properties for this experiment.
* :meth:`~neptune.experiments.Experiment.get_system_properties`: Gets experiment properties.
* :meth:`~neptune.experiments.Experiment.get_tags`: Gets the tags associated with this experiment.
* :meth:`~neptune.experiments.Experiment.download_artifact`: Download an artifact (file) from the experiment storage.
* :meth:`~neptune.experiments.Experiment.download_artifacts`: Download a directory or a single file from experiment’s artifacts as a ZIP archive.
* :meth:`~neptune.experiments.Experiment.download_sources`: Download a directory or a single file from experiment’s sources as a ZIP archive.

.. External Links

.. |this example| raw:: html

 <a href="https://ui.neptune.ai/USERNAME/example-project/n/Experiments-analysis-with-Query-API-and-Seaborn-31510158-04e2-47a5-a823-1cd97a0d8fcd/91350522-2b98-482d-bc14-a6ff5c061b6b>" target="_blank">Study this example</a>

.. |Youtube Video| raw:: html

    <iframe width="720" height="420" src="https://www.youtube.com/embed/ILnM4owoJqw" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>