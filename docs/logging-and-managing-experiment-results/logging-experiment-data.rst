.. _guides-logging-data-to-neptune:

Logging experiment data
=======================
During machine learning experimentation, you need to keep track of many different types of (meta)-data. Neptune help you in this task by logging, keeping track of and visualizing experiment (meta)-data.

You can track many different types of (meta)-data to the experiment. It can be metric, loss, image, interactive visualization, model checkpoint, pandas DataFrame and many more. Simply check :ref:`what you can log <what-you-can-log>` section below for complete listing.

Basics of logging
-----------------
Logging experiments data to Neptune is simple and straightforward.

Minimal example
^^^^^^^^^^^^^^^
Let's create minimal code snippet that log single value to the experiment: 'acc'=0.95.

.. code-block:: python3

    import neptune

    neptune.init('my_workspace/my_project')
    neptune.create_experiment()

    # log 'acc' value 0.95
    neptune.log_metric('acc', 0.95)

Above snippet sets project, creates experiment and log one value to it. When script ends, the experiment is closed automatically. As a result you have new experiment with one value in one metric ('acc'=0.95).

Everything that is evaluated after ``neptune.create_experiment()`` and before the end of the script can be logged to the experiment.

**[loom-placeholder]**

.. _what-you-can-log:

What you can log
----------------
Neptune supports numerous types of data that you can log to the experiment. Here, you can find all of them listed and described:

+----------------------------------------------------------------------------------------------+--------------------+
|                                       Data to log/track                                      | Video overview     |
+==============================================================================================+====================+
| :ref:`Metrics <logging-experiment-data-metrics>`                                             | [loom-placeholder] |
+----------------------------------------------------------------------------------------------+--------------------+
| :ref:`Parameters <logging-experiment-data-parameters>`                                       | [loom-placeholder] |
+----------------------------------------------------------------------------------------------+--------------------+
| :ref:`Code <logging-experiment-data-code>`                                                   | [loom-placeholder] |
|                                                                                              |                    |
| * :ref:`Git <logging-experiment-data-code-git>`                                              |                    |
| * :ref:`Code Snapshot <logging-experiment-data-code-code-snapshot>`                          |                    |
| * :ref:`Notebook Snapshot <logging-experiment-data-code-notebook-snapshot>`                  |                    |
+----------------------------------------------------------------------------------------------+--------------------+
| :ref:`Experiment information <logging-experiment-data-experiment-information>`               | [loom-placeholder] |
|                                                                                              |                    |
| * :ref:`Experiment name <logging-experiment-data-experiment-information-name>`               |                    |
| * :ref:`Experiment description <logging-experiment-data-experiment-information-description>` |                    |
| * :ref:`Experiment tags <logging-experiment-data-experiment-information-tags>`               |                    |
+----------------------------------------------------------------------------------------------+--------------------+
| :ref:`Hardware consumption <logging-experiment-data-hardware-consumption>`                   | [loom-placeholder] |
+----------------------------------------------------------------------------------------------+--------------------+
| :ref:`Text <logging-experiment-data-text>`                                                   | [loom-placeholder] |
+----------------------------------------------------------------------------------------------+--------------------+
| :ref:`Properties <logging-experiment-data-properties>`                                       | [loom-placeholder] |
+----------------------------------------------------------------------------------------------+--------------------+
| :ref:`Data versions <logging-experiment-data-data-versions>`                                 | [loom-placeholder] |
+----------------------------------------------------------------------------------------------+--------------------+
| :ref:`Files <logging-experiment-data-files>`                                                 | [loom-placeholder] |
|                                                                                              |                    |
| * :ref:`Model checkpoints <logging-experiment-data-files-model-checkpoints>`                 |                    |
| * :ref:`HTML objects <logging-experiment-data-files-html-objects>`                           |                    |
+----------------------------------------------------------------------------------------------+--------------------+
| :ref:`Images <logging-experiment-data-images>`                                               | [loom-placeholder] |
|                                                                                              |                    |
| * :ref:`Matplotlib <logging-experiment-data-images-matplotlib>`                              |                    |
| * :ref:`PIL <logging-experiment-data-images-pil>`                                            |                    |
| * :ref:`NumPy <logging-experiment-data-images-numpy>`                                        |                    |
+----------------------------------------------------------------------------------------------+--------------------+
| :ref:`Interactive charts <logging-experiment-data-interactive-charts>`                       | [loom-placeholder] |
|                                                                                              |                    |
| * :ref:`Matplotlib <logging-experiment-data-interactive-charts-matplotlib>`                  |                    |
| * :ref:`Altair <logging-experiment-data-interactive-charts-altair>`                          |                    |
| * :ref:`Bokeh <logging-experiment-data-interactive-charts-bokeh>`                            |                    |
| * :ref:`Plotly <logging-experiment-data-interactive-charts-plotly>`                          |                    |
+----------------------------------------------------------------------------------------------+--------------------+
| :ref:`Video <logging-experiment-data-images-video>`                                          | [loom-placeholder] |
|                                                                                              |                    |
| :ref:`Audio <logging-experiment-data-images-audio>`                                          |                    |
+----------------------------------------------------------------------------------------------+--------------------+
| :ref:`Tables <logging-experiment-data-images-tables>`                                        | [loom-placeholder] |
|                                                                                              |                    |
| * :ref:`pandas <logging-experiment-data-images-pandas>`                                      |                    |
| * :ref:`csv <logging-experiment-data-images-csv>`                                            |                    |
+----------------------------------------------------------------------------------------------+--------------------+
| :ref:`Python objects <logging-experiment-data-images-python-objects>`                        | [loom-placeholder] |
|                                                                                              |                    |
| * :ref:`Explainers (DALEX) <logging-experiment-data-images-python-objects-dalex>`            |                    |
+----------------------------------------------------------------------------------------------+--------------------+

.. _logging-experiment-data-metrics:

Metrics
^^^^^^^
You can log one or multiple metrics to a log section with the :meth:`~neptune.experiments.Experiment.log_metric` method. These could be machine learning metrics like accuracy, MSE or any numerical value.

.. code-block:: python3

    # single value
    neptune.log_metric('test_accuracy', 0.76)

    # single value for each epoch
    for epoch in range(epoch_nr):
        epoch_accuracy = ...
        neptune.log_metric('test_accuracy', epoch_accuracy)

.. _logging-experiment-data-parameters:

Parameters
^^^^^^^^^^
Define parameters as Python dictionary and pass them to the :meth:`~neptune.projects.Project.create_experiment` method to keep track of them. You can also use them later to compare experiments.

.. code-block:: python3

    # define parameters
    PARAMS = {'batch_size': 64,
              'dense_units': 128,
              'dropout': 0.2,
              'learning_rate': 0.001,
              'optimizer': 'Adam'}

    # pass parameters to create experiment
    neptune.create_experiment(params=PARAMS)

.. note::

    Experiment parameters are read-only. You cannot change or update them during experiment.

.. _logging-experiment-data-code:

Code
^^^^
You can version your code with Neptune. Few options in that regard are available.

.. _logging-experiment-data-code-git:

Git
"""
Neptune automatically discovers if you start experiment from directory that is part of the git repository. It logs commit information (id, message, author, date), branch, and remote address to your experiment. Neptune also logs the entrypoint file so that you have all the information about the run.

All that info is in the 'Details' section of the experiment.

.. _logging-experiment-data-code-code-snapshot:

Code Snapshot
"""""""""""""
You can snapshot code files or folders and have them displayed in the "Source code" section of the UI.
To do that just specify all the files (or regex) that you want to snapshot when you create an experiment.

.. code-block:: python3

    neptune.create_experiment(upload_source_files=['model.py', 'prep_data.py'])

.. _logging-experiment-data-code-notebook-snapshot:

Notebook Code Snapshot
""""""""""""""""""""""
You can also save notebook checkpoints to Neptune. You just need to install :ref:`notebook extension <installation-notebook-extension>`. With that you can log entire notebook by clicking a button or let Neptune auto-snapshot your experiments whenever you create a new one inside notebook.

.. _logging-experiment-data-text:

Text
^^^^
Log text information to experiment. You will have it in the 'Logs' section of the experiment. Use :meth:`~neptune.experiments.Experiment.log_text` method to do it.

.. code-block:: python3

    data_item = ...
    neptune.log_text('my_text_data', str(data_item))

.. _logging-experiment-data-hardware-consumption:

Hardware consumption
^^^^^^^^^^^^^^^^^^^^
You can monitor the hardware for your experiment runs automatically. You can see the utilization of the CPU (average of all cores), memory and - for each GPU unit - memory usage and utilization.

.. note::

    Install ``psutil`` to log hardware consumption metrics:

    **pip**

    .. code:: bash

        pip install psutil

    **conda**

    .. code:: bash

        conda install -c anaconda psutil


.. _logging-experiment-data-experiment-information:

Experiment information
^^^^^^^^^^^^^^^^^^^^^^
To better describe an experiment you can use 'name', 'description' and 'tags'.

.. _logging-experiment-data-experiment-information-name:

Experiment name
"""""""""""""""
You can add name to experiment when you :meth:`~neptune.projects.Project.create_experiment`. Try to keep it short and descriptive. Experiment name appears in the 'Details' section of the experiment and can be displayed as a column on the experiments dashboard.

.. code-block:: python3

        neptune.create_experiment(name='Mask R-CNN data-v2')

.. note::

    Edit 'name' directly from the UI, either in the 'Details' section of the experiment or in the experiments dashboard.

.. _logging-experiment-data-experiment-information-description:

Experiment description
""""""""""""""""""""""
You can add longer note to the experiment when you :meth:`~neptune.projects.Project.create_experiment`. Description appears in the 'Details' section of the experiment and can be displayed as a column on the experiments dashboard.

.. code-block:: python3

        neptune.create_experiment(description='neural net trained on Fashion-MNIST with high LR and low dropout')

.. note::

    Edit 'description' directly from the UI, either in the 'Details' section of the experiment or in the experiments dashboard.

.. _logging-experiment-data-experiment-information-tags:

Experiment tags
"""""""""""""""
You can add tags to the experiment when you :meth:`~neptune.projects.Project.create_experiment`. Tags are convenient way to organize or group experiments.

.. code-block:: python3

        neptune.create_experiment(tags=['classification', 'pytorch', 'prod_v2.0.1'])

Tags appear in the 'Details' section of the experiment and can be displayed as a column on the experiments dashboard.

.. tip::

    You can quickly filter by tag by clicking on it in the experiments dashboard.

.. note::

    Add or remove tags directly from the UI, either in the 'Details' section of the experiment or in the experiments dashboard.

.. _logging-experiment-data-properties:

Properties
^^^^^^^^^^
You can log 'key', 'value' pairs to the experiment. Those could be data versions, URL or path to the model on your filesystem, or anything else that fit the generic ``'key'-'value'`` scheme. What distinguishes them from :ref:`parameters <logging-experiment-data-parameters>` is that they are editable after experiment is created. Properties appear in the 'Details' section of the experiment and can be displayed as a column on the experiments dashboard.

You can add properties to your experiment when you :meth:`~neptune.projects.Project.create_experiment`. Pass Python dictionary with properties like this:

.. code-block:: python3

    neptune.create_experiment(properties={'data_version': 'fd5c084c-ff7c',
                                          'model_id': 'a44521d0-0fb8'})

Another option is to add it anytime during an experiment using :meth:`~neptune.experiments.Experiment.set_property`:

.. code-block:: python3

    # single key-value pair at a time
    neptune.set_property('model_id', 'a44521d0-0fb8')

.. _logging-experiment-data-data-versions:

Data versions
^^^^^^^^^^^^^
[text]

.. _logging-experiment-data-files:

Files
^^^^^
[text]

.. _logging-experiment-data-files-model-checkpoints:

Model checkpoints
"""""""""""""""""
[text]

.. _logging-experiment-data-files-html-objects:

HTML objects
""""""""""""
[text]

.. _logging-experiment-data-images:

Images
^^^^^^
[text]

.. _logging-experiment-data-images-matplotlib:

Matplotlib
""""""""""
[text]

.. _logging-experiment-data-images-pil:

PIL
"""
[text]

.. _logging-experiment-data-images-numpy:

NumPy
"""""
[text]

.. _logging-experiment-data-interactive-charts:

Interactive charts
^^^^^^^^^^^^^^^^^^
[text]

.. _logging-experiment-data-interactive-charts-matplotlib:

Matplotlib
""""""""""
[text]

.. _logging-experiment-data-interactive-charts-altair:

Altair
""""""
[text]

.. _logging-experiment-data-interactive-charts-bokeh:

Bokeh
"""""
[text]

.. _logging-experiment-data-interactive-charts-plotly:

Plotly
""""""
[text]

.. _logging-experiment-data-images-video:

Video
^^^^^
[text]

.. _logging-experiment-data-images-audio:

Audio
^^^^^
[text]

.. _logging-experiment-data-images-tables:

Tables
^^^^^^
[text]

.. _logging-experiment-data-images-pandas:

pandas
""""""
[text]

.. _logging-experiment-data-images-csv:

csv
"""
[text]

.. _logging-experiment-data-images-python-objects:

Python objects
^^^^^^^^^^^^^^
[text]

.. _logging-experiment-data-images-python-objects-dalex:

Explainers (DALEX)
""""""""""""""""""













Logging with integrations
-------------------------
Besides logging using Neptune Python library, you can also use integrations that let you log relevant data with almost no code changes. Have a look at :ref:`Integrations page <integrations-index>` for more information or find your favourite library in one of the following categories:

- :ref:`Deep learning frameworks <integrations-deep-learning-frameworks>`
- :ref:`Machine learning frameworks <integrations-machine-learning-frameworks>`
- :ref:`Hyperparameter optimization libraries <integrations-hyperparameter-optimization-frameworks>`
- :ref:`Visualization libraries <integrations-visualization-tools>`
- :ref:`Experiment tracking frameworks <integrations-experiment-tracking-frameworks>`
- :ref:`Other integrations <integrations-other-integrations>`

**[loom-placeholder]**

Advanced
--------
Minimal example revisited
^^^^^^^^^^^^^^^^^^^^^^^^^
Let's create minimal code snippet that log single value to the experiment: 'acc'=0.96.

.. code-block:: python3

    import neptune

    neptune.init('my_workspace/my_project')
    exp = neptune.create_experiment()

    # log 'acc' value 0.96
    exp.log_metric('acc', 0.96)

``neptune.create_experiment()`` returns :class:`~neptune.experiments.Experiment` object, that allows you to pass it around your code base and perform logging from multiple Python files to the single experiment.

**[loom-placeholder]**


.. External links
