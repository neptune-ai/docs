.. _guides-download_data:

Downloading experiment data
===========================
|Youtube Video|

.. _download_data-basics:

Basics of downloading
---------------------
Why
^^^
Almost all data that is logged to the project (experiments and notebooks) can be downloaded to the local machine. You may want to do it for a variety of reasons:

* Build custom analysis or visualizations using experiment data.
* Use saved model checkpoint elsewhere.
* Get sources of experiment and run it again.
* Build report that uses data across projects.
* Archive old project.

How
^^^
There are three ways to download data from Neptune:

#. Programmatically, by using neptune-client: for example downloading experiment dashboard as pandas DataFrame. Check :ref:`download programmatically <download-programmatically>` below.
#. Directly from the UI: for example downloading notebook checkpoint or experiments dashboard as csv. Check :ref:`downloading from Neptune UI <download-from-neptune-ui>` below.
#. From the JupyterLab interface: Check :ref:`downloading checkpoint <download-notebook>` documentation.

.. _download_data-basics-example:

Simple example
""""""""""""""
In this example you will programmatically download experiment dashboard as pandas DataFrame. It's a decent representation of the overall idea behind downloading data from Neptune by using Neptune Python library.

Download experiments dashboard data as pandas DataFrame. Use :meth:`~neptune.projects.Project.get_leaderboard` to do it.

.. code-block::

    import neptune

    # Get project
    project = neptune.init('my_workspace/my_project')

    # Download experiments dashboard as pandas DataFrame
    exp_data_df = project.get_leaderboard()

``exp_data_df`` is a pandas DataFrame, where each row is an experiment and columns represent all system properties, metrics and text logs, parameters and properties in these experiments. For metrics, the latest value is returned.

.. _download_data-can-cannot:

What you can/cannot download
----------------------------

.. _download-programmatically:

Download programmatically
^^^^^^^^^^^^^^^^^^^^^^^^^
Project level
"""""""""""""
On the level of the project, you can do two major actions:

* Fetch a list of :class:`neptune.experiments.Experiment` objects, using :meth:`~neptune.projects.Project.get_experiments`.
* Download the entire experiments view as a Pandas DataFrame, using :meth:`~neptune.projects.Project.get_leaderboard`. That exact action was done in the :ref:`simple example <download_data-basics-example>`.

.. note::

    Check :class:`~neptune.projects.Project` for other related methods.

Fetch a list of :class:`~neptune.experiments.Experiment` objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let's fetch a list of :class:`~neptune.experiments.Experiment` objects that match some criteria.

.. code-block:: python3

    import neptune

    # Get project
    project = neptune.init('my_workspace/my_project')

    # Get list of experiment objects created by 'sophia'
    sophia_experiments = project.get_experiments(owner='sophia')

    # Get another list of experiment objects that have 'cycleLR' assigned
    cycleLR_experiments = project.get_experiments(tag='cycleLR')

First, you need to get correct project, then you simply run :meth:`~neptune.projects.Project.get_experiments` with appropriate parameters. ``sophia_experiments`` and ``cycleLR_experiments`` are lists of :class:`neptune.experiments.Experiment` objects. You can use it either to download data from experiments or update them:

* For updating check :ref:`this guide <update-existing-experiment-basics-simple-example>`.
* For downloading continue reading this page.

Download experiment dashboard as DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let's download the filtered experiments dashboard view as a Pandas DataFrame.

.. code-block::

    import neptune

    # Get project
    project = neptune.init('my_workspace/my_project')

    # Get dashboard with experiments contributed by 'sophia'
    sophia_df = project.get_leaderboard(owner='sophia')

    # Get another dashboard with experiments tagged 'cycleLR'
    cycleLR_df = project.get_leaderboard(tag='cycleLR')

First, you need to get correct project, then you simply run :meth:`~neptune.projects.Project.get_leaderboard` with appropriate parameters. ``sophia_df`` and ``cycleLR_df`` are pandas DataFrames where each row is an experiment and columns represent all system properties, metrics and text logs, parameters and properties in these experiments. For metrics, the latest value is returned.

Experiment level
""""""""""""""""

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

Fetch scripts

Fetch artifacts


.. _download-from-neptune-ui:

Downloading from Neptune UI
^^^^^^^^^^^^^^^^^^^^^^^^^^^
When in Neptune, simply look for the download icon. It's located in multiple places in the application and allows you to download various parts of the experiments or notebook. Below are three examples:

+--------------------------------------------------------------------------------------------------------------------------+
| .. image:: ../_static/images/logging-and-managing-experiment-results/downloading-experiment-data/download-artifact.png   |
|     :target: ../_static/images/logging-and-managing-experiment-results/downloading-experiment-data/download-artifact.png |
|     :alt: Download model checkpoint                                                                                      |
+==========================================================================================================================+
| Download model checkpoint                                                                                                |
+--------------------------------------------------------------------------------------------------------------------------+

+--------------------------------------------------------------------------------------------------------------------------+
| .. image:: ../_static/images/logging-and-managing-experiment-results/downloading-experiment-data/download-dashboard.png  |
|    :target: ../_static/images/logging-and-managing-experiment-results/downloading-experiment-data/download-dashboard.png |
|    :alt: Download experiments dashboard as csv                                                                           |
+==========================================================================================================================+
| Download experiments dashboard as csv                                                                                    |
+--------------------------------------------------------------------------------------------------------------------------+

+-------------------------------------------------------------------------------------------------------------------------+
| .. image:: ../_static/images/logging-and-managing-experiment-results/downloading-experiment-data/download-notebook.png  |
|    :target: ../_static/images/logging-and-managing-experiment-results/downloading-experiment-data/download-notebook.png |
|    :alt: Download notebook checkpoint                                                                                   |
+=========================================================================================================================+
| Download notebook checkpoint                                                                                            |
+-------------------------------------------------------------------------------------------------------------------------+

Download helpers
----------------
You can combine few downloading options to build custom visualizations or analysis. One example is to plot multiple metrics on the single chart, by using :meth:`~neptune.projects.Project.get_experiments` and :meth:`~neptune.experiments.Experiment.get_numeric_channels_values`.

Build seaborn visualization - overlay metric from multiple experiment on the same plot.

.. code-block:: python3

    # Import libraries
    import neptune
    import pandas as pd
    import seaborn as sns

    # Set project
    project = neptune.init('my_workspace/my_project')

    # Get list of experiments
    experiments = project.get_experiments(owner='...', tag='...')

    # Prepare metrics to visualise
    logs_names = ['epoch_accuracy', 'epoch_loss', ...]

    # Initialize metrics DataFrame for seaborn
    metrics_df = pd.DataFrame(columns=['id', *logs_names])

    # Download metrics data from all experiments from the list
    for experiment in experiments:
        df = experiment.get_numeric_channels_values(*logs_names)
        df.insert(loc=0, column='id', value=experiment.id)
        metrics_df = metrics_df.append(df, sort=True)

    # Prepare df with top 10 experiments with respect to epoch accuracy
    top_10_epoch_acc_df = ...

    # Make seaborn plot
    g = sns.relplot(x='x', y='epoch_accuracy', kind='line', hue='id',
                    height=9, aspect=1,
                    data=top_10_epoch_acc_df[['x', 'epoch_accuracy', 'id']])

The result will look like this:

.. image:: ../_static/images/logging-and-managing-experiment-results/downloading-experiment-data/download-and-plot.png
   :target: ../_static/images/logging-and-managing-experiment-results/downloading-experiment-data/download-and-plot.png
   :alt: Metrics plotted in single chart

|example-neptune-notebook|


.. External Links

.. |Youtube Video| raw:: html

    <iframe width="720" height="420" src="https://www.youtube.com/embed/ILnM4owoJqw" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

.. Buttons

.. |example-neptune-notebook| raw:: html

    <div class="see-in-neptune">
        <button><a target="_blank"
                   href="https://ui.neptune.ai/USERNAME/example-project/n/analysis-v1-final-final-31510158-04e2-47a5-a823-1cd97a0d8fcd/fa835a93-9d8d-40a4-a043-36879d5f7471">
                <img width="50" height="50" style="margin-right:10px"
                     src="https://gist.githubusercontent.com/kamil-kaczmarek/7ac1e54c3b28a38346c4217dd08a7850/raw/8880e99a434cd91613aefb315ff5904ec0516a20/neptune-ai-blue-vertical.png">See example in Neptune</a>
        </button>
    </div>
