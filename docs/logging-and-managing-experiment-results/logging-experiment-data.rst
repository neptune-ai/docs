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

Logging with integrations
-------------------------

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

Troubleshooting
---------------
