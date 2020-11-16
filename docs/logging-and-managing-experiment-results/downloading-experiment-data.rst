.. _guides-download_data:

Downloading experiment data
===========================
|Youtube Video|

Almost all data that is logged to the project (experiments and notebooks) can be downloaded to the local machine. You may want to do it for a variety of reasons:

* Build custom analysis or visualizations using experiment data.
* Use saved model checkpoint elsewhere.
* Get sources of experiment and run it again.
* Build report that uses data across projects.
* Archive old project.

There are three ways to download data from Neptune:

#. Programmatically, by using neptune-client: for example downloading experiment dashboard as pandas DataFrame. Check :ref:`download programmatically <download-programmatically>` below.
#. Directly from the UI: for example downloading notebook checkpoint or experiments dashboard as csv. Check :ref:`downloading from Neptune UI <download-from-neptune-ui>` below.
#. From the JupyterLab interface: Check :ref:`downloading checkpoint <download-notebook>` documentation.

.. _download-programmatically:

Downloading programmatically
----------------------------
You can download experiment data programmatically.

Snippet below shows how to download experiment dashboard as pandas DataFrame. It's a decent representation of the overall idea behind downloading data from Neptune.

To download experiments dashboard data as pandas DataFrame use :meth:`~neptune.projects.Project.get_leaderboard`.

.. code-block::

    import neptune

    # Get project
    project = neptune.init('my_workspace/my_project')

    # Download experiments dashboard as pandas DataFrame
    data = project.get_leaderboard()
    data.head()

``data`` is a pandas DataFrame, where each row is an experiment and columns represent all system properties, metrics and text logs, parameters and properties in these experiments. For metrics, the latest value is returned.

.. image:: ../_static/images/logging-and-managing-experiment-results/downloading-experiment-data/data-head.png
    :target: ../_static/images/logging-and-managing-experiment-results/downloading-experiment-data/data-head.png
    :alt: Example downloaded dashboard data

Downloading from the project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
On the level of the project, you can do two major actions:

* Fetch a list of :class:`neptune.experiments.Experiment` objects, using :meth:`~neptune.projects.Project.get_experiments`, then access or download information from them.
* Download the entire experiments view as a Pandas DataFrame, using :meth:`~neptune.projects.Project.get_leaderboard`. That exact action was done in :ref:`this example snippet <download-programmatically>`.

.. note::

    Check :class:`~neptune.projects.Project` for other related methods.

Fetch a list of :class:`~neptune.experiments.Experiment` objects
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
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
""""""""""""""""""""""""""""""""""""""""""
Let's download the filtered experiments dashboard view as a Pandas DataFrame, using :meth:`~neptune.projects.Project.get_leaderboard`.

.. code-block::

    import neptune

    # Get project
    project = neptune.init('my_workspace/my_project')

    # Get dashboard with experiments contributed by 'sophia'
    sophia_df = project.get_leaderboard(owner='sophia')

    # Get another dashboard with experiments tagged 'cycleLR'
    cycleLR_df = project.get_leaderboard(tag='cycleLR')

First, you need to get correct project, then you simply run :meth:`~neptune.projects.Project.get_leaderboard` with appropriate parameters. ``sophia_df`` and ``cycleLR_df`` are pandas DataFrames where each row is an experiment and columns represent all system properties, metrics and text logs, parameters and properties in these experiments. For metrics, the latest value is returned.

Note that prefixes are added to metrics, parameters and properties:

* ``channel_`` for metrics and text logs, for example: ``channel_epoch/accuracy``
* ``parameter_`` for example: ``parameter_optimizer``
* ``property_`` for example: ``property_test_images_version``

Example dataframe will look like this:

.. image:: ../_static/images/logging-and-managing-experiment-results/downloading-experiment-data/data-head-2.png
    :target: ../_static/images/logging-and-managing-experiment-results/downloading-experiment-data/data-head-2.png
    :alt: Example downloaded dashboard data

.. note::

    To download only experiments that you want, you can filter them by ``id``, ``state``, ``owner``, ``tag`` and ``min_running_time``. Check :meth:`~neptune.projects.Project.get_leaderboard` documentation for details.

Downloading from the experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
On this level you can use all methods that get/download data from the :class:`neptune.experiments.Experiment` object. Three types of data are especially useful: metrics, artifacts and source code.

First step in all cases is to get experiment object.

.. code-block:: python3

    import neptune

    # Get project
    project = neptune.init('my_workspace/my_project')

    # Get experiment object for appropriate experiment, here 'SHOW-2066'
    my_exp = project.get_experiments(id='SHOW-2066')[0]

Have a look at :ref:`this section <update-existing-experiment-basics-simple-example>` about updating experiments to learn more about it.

Here, ``my_exp`` is :class:`neptune.experiments.Experiment` object that will be used in the following section about downloading metrics, artifacts and source code.

Metrics
"""""""
You can download metrics data as pandas DataFrame.

.. code-block:: python3

    # 'my_exp' is experiment object
    data = my_exp.get_numeric_channels_values('epoch/accuracy', 'epoch/loss')

:meth:`~neptune.experiments.Experiment.get_numeric_channels_values` accepts comma separated metric names. ``data`` is a pandas DataFrame with metrics data.

You can also use :meth:`~neptune.experiments.Experiment.get_logs` to see all logs (types: metrics, text, images) names in the experiment.

.. code-block:: python3

    # exp is Experiment object
    print(exp.get_logs().keys())

Result looks like this:

.. image:: ../_static/images/logging-and-managing-experiment-results/downloading-experiment-data/logs-names.png
    :target: ../_static/images/logging-and-managing-experiment-results/downloading-experiment-data/logs-names.png
    :alt: Example logs names printed in notebook

.. note::

    It’s good idea to get metrics with common temporal pattern (like iteration or batch/epoch number). Thanks to this each row of returned DataFrame has metrics from the same moment in experiment. For example, combine epoch metrics to one DataFrame and batch metrics to the other.

Files
"""""
Download files from the experiment. Any file that is logged to the |artifacts| section can be downloaded.

Notice that there are two methods for this:

* :meth:`~neptune.experiments.Experiment.download_artifact`: single file download.
* :meth:`~neptune.experiments.Experiment.download_artifacts`: multiple files download as a ZIP archive.

.. code-block:: python3

    # Download csv file
    my_exp.download_artifact('aux_data/preds_test.csv', 'data/')

    # Download all model checkpoints to the cwd
    my_exp.download_artifacts('model_checkpoints/')

Source code
"""""""""""
Download source code used un the experiment as a ZIP archive.

.. code-block:: python3

    # Download all sources to the cwd
    my_exp.download_sources()

.. note::

    You can also download source directly from the UI: :ref:`here is how <download-from-neptune-ui>`.

More options
""""""""""""
Besides metrics, artifacts and scripts covered above, you can use other methods as well. Here is a full list of methods that download data:

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
* :meth:`~neptunecontrib.api.utils.get_pickle`: Download pickled artifact (file) from Neptune and returns a Python object.

Combining downloading methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can combine few downloading options to build custom visualizations or analysis. Example below shows how to use :meth:`~neptune.projects.Project.get_experiments` and :meth:`~neptune.experiments.Experiment.get_numeric_channels_values` and seaborn library to overlay metric from multiple experiments on the same plot.

Get list of :class:`~neptune.experiments.Experiment` objects.

.. code-block:: python3

    import neptune

    # Set project
    project = neptune.init('my_workspace/my_project')

    # Get list of experiments
    experiments = project.get_experiments(owner='...', tag='...')

Download metrics data from all experiments in the list, by using :meth:`~neptune.experiments.Experiment.get_numeric_channels_values`

.. code-block:: python3

    for experiment in experiments:
        raw_df = experiment.get_numeric_channels_values(*metrics_names)
        metrics_df = raw_df[...]

Prepare dataframe with top performing experiments and make seaborn plot

.. code-block:: python3

    # Prepare dataframe
    top_acc_df = metrics_df.sort_values(...)

    # Make seaborn plot
    g = sns.relplot(x='x', y='epoch_accuracy', data=top_acc_df)

The result will look like this:

.. image:: ../_static/images/logging-and-managing-experiment-results/downloading-experiment-data/download-and-plot.png
   :target: ../_static/images/logging-and-managing-experiment-results/downloading-experiment-data/download-and-plot.png
   :alt: Metrics plotted in single chart

|example-neptune-notebook|

.. _download-from-neptune-ui:

Downloading from Neptune UI
---------------------------
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

Downloading from Jupyter Notebook
---------------------------------
You can download notebook checkpoint directly from Neptune to the Jupyter or JupyterLab interface. Check :ref:`downloading checkpoint <download-notebook>` documentation for details.


.. External Links

.. |Youtube Video| raw:: html

    <iframe width="720" height="420" src="https://www.youtube.com/embed/ILnM4owoJqw" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

.. |artifacts| raw:: html

    <a href="https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-325/artifacts" target="_blank">artifacts</a>

.. Buttons

.. |example-neptune-notebook| raw:: html

    <div class="see-in-neptune">
        <button><a target="_blank"
                   href="https://ui.neptune.ai/USERNAME/example-project/n/analysis-v1-final-final-31510158-04e2-47a5-a823-1cd97a0d8fcd/fa835a93-9d8d-40a4-a043-36879d5f7471">
                <img width="50" height="50" style="margin-right:10px"
                     src="https://gist.githubusercontent.com/kamil-kaczmarek/7ac1e54c3b28a38346c4217dd08a7850/raw/8880e99a434cd91613aefb315ff5904ec0516a20/neptune-ai-blue-vertical.png">See example in Neptune</a>
        </button>
    </div>
