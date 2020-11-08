.. _guides-download_data:

Downloading experiment data programmatically
============================================
|Youtube Video|

.. _download_data-basics:

Basics of downloading
---------------------
Almost all data that is logged to the project (experiments and notebooks) can be downloaded to the local machine. You may want to do it for a variety of reasons:

* Build custom analysis or visualizations using experiment data.
* Use saved model checkpoint elsewhere.
* Get sources of experiment and run it again.
* Build report that uses data across projects.
* Archive old project.

How to download
^^^^^^^^^^^^^^^
There are three ways to download data from Neptune:

#. Programmatically, by using neptune-client: for example downloading experiment dashboard as pandas DataFrame. Check :ref:`Simple example <download_data-basics-example>` below.
#. Directly from the UI: for example downloading notebook checkpoint or experiments dashboard as csv.
#. From the JupyterLab interface: for example :ref:`downloading checkpoint <download-notebook>`.

On the level of project, you can fetch a list of :class:`neptune.projects.Project` objects,
fetch the entire experiments view as a Pandas DataFrame and get all members of the project.

The following methods are provided:

* :meth:`~neptune.projects.Project.get_experiments`: Gets a list of experiments matching the specified criteria.
* :meth:`~neptune.projects.Project.get_leaderboard`: Gets the entire Neptune experiment view in the dashboard as a Pandas DataFrame.
* :meth:`~neptune.projects.Project.get_members`: Gets a list of project members.



Fetch experiment list
"""""""""""""""""""""
Fetch dashboard as DF
"""""""""""""""""""""

Experiment level
^^^^^^^^^^^^^^^^

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


Fetch metrics
"""""""""""""
Fetch scripts
"""""""""""""
Fetch artifacts
""""""""""""""

.. _download_data-how-to:

How to download step by step
----------------------------
Download helpers and integrations
---------------------------------




















.. External Links

.. |Youtube Video| raw:: html

    <iframe width="720" height="420" src="https://www.youtube.com/embed/ILnM4owoJqw" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
