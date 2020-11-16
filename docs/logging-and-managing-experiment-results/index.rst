.. _guides-logging-and-managing-experiment-results:

Logging and managing experiment results
=======================================
Log and manage data in Neptune via |neptune-client-github| (Python and R clients are available).

[loom-placeholder]

What is neptune-client?
-----------------------
|neptune-client-github| is an open source Python library that serves two purposes:

#. logging machine learning experiments,
#. update existing experiment with new data and visualizations,
#. download experiment data from Neptune to local machine.

It is designed to be:

* **lightweight**: low setup effort,
* **generic**: capable of logging any kind of machine learning work
* **straightforward**: user defines what to keep track of during experiment to use.

You can :ref:`log experiments from anywhere <execution-environments-index>` (local machine, cluster, cloud, Colab, etc.) and they will be tracked in the same, standardized way. You will be able to quickly compare experiments run by you on your workstation with experiments run on AWS by your team-mate.

What you can do?
----------------
You can think of three main actions around experiments:

#. **Log experiments** - explained above, where you log metrics and other data to the experiment
#. **Update experiments** - you can log more data to previously closed experiment. Here is how to :ref:`update experiment <update-existing-experiment>`
#. **Download experiments** - all logged data can be :ref:`downloaded programmatically <guides-download_data>`.

What you need to add to your code to start logging
--------------------------------------------------
[loom-placeholder]

Bare minimum are one import and two methods:

.. code-block:: python3

    import neptune

    # Set project
    neptune.init('my_workspace/my_project')

    # Create experiment
    neptune.create_experiment()

These are usually just copy&paste into your existing project code.

.. note::

    Remember to :ref:`create project <create-project>` and :ref:`setup API token <how-to-setup-api-token>` before you create an experiment using snippet above.

Now, that the experiment is created you can start logging metrics, losses, images, model weights or whatever you feel relevant to keep track of in your experiment.

Overall idea for logging is to use methods similar to this:

.. code-block:: python3

    neptune.log_metric('acc', 0.95)

Generic recipe being:

.. code-block:: python3

    neptune.log_X('some_name', some_value)

Where ``X`` could be metric, artifact, chart, pickle, etc. Check the :ref:`logging section <what-you-can-log>` for a complete list.

.. note::

    If you work in Notebooks, you need to place ``neptune.stop()`` at the very end of your experiment to make sure that everything will be closed properly.

    Note that you are still able to :ref:`update an experiment <update-existing-experiment>` that was closed before.

Essential Neptune client concepts
---------------------------------
:ref:`Project <logging_project>` and :ref:`Experiment <logging_experiment>` are two important entities in Neptune.

Basic snippet below, sets project and creates new experiment in that project.

.. code-block:: python3

    # Set project
    neptune.init('my_workspace/my_project')

    # Create new experiment
    neptune.create_experiment()

.. _logging_project:

Project
^^^^^^^
It is a **collection of Experiments**, created by user (or users) assigned to the project.

You can log experiments to the project or fetch all experiments that satisfy some criteria.

.. code-block:: python3

    # Set project and get project object
    project = neptune.init('my_workspace/my_project')

    # Use project to create experiment
    project.create_experiment()

    # Use project to get experiments data from the project
    project.get_leaderboard(state=['succeeded'])

Learn more about :ref:`downloading data from Neptune <guides-download_data>`. Check also, :class:`~neptune.projects.Project` to use in your Python code.

.. _logging_experiment:

Experiment
^^^^^^^^^^
Experiment is everything that you log to Neptune, beginning at ``neptune.create_experiment()`` and ending when script finishes or when you explicitly stop the experiment with :meth:`~neptune.experiments.Experiment.stop`.

Creating experiment is easy:

.. code-block:: python3

    # Set project
    neptune.init('my_workspace/my_project')

    # Create new experiment
    neptune.create_experiment()

You can now log various data to the experiment including metrics, losses, model weights, images, predictions and much more. Have a look at the complete list of :ref:`what you can log <what-you-can-log>` to the experiment

Besides logging data, you can also :ref:`download experiment data <guides-download_data>` to you local machine or :ref:`update an existing experiment <update-existing-experiment>` even when it's closed.

.. note::

    ``neptune.log_metric('some_name', some_value)`` is for tracking all numeric values to Neptune (metric, loss, score, variances, etc.). Learn, what else can be tracked to experiment from :ref:`this list <what-you-can-log>`.

Learn more about :class:`~neptune.experiments.Experiment` to use in your Python code.


.. Local navigation

.. toctree::
   :maxdepth: 1
   :hidden:

   Logging experiment data <logging-experiment-data.rst>
   Downloading experiment data <downloading-experiment-data.rst>
   Updating existing experiment <updating-existing-experiment.rst>


.. External links

.. |neptune-client-github| raw:: html

    <a href="https://github.com/neptune-ai/neptune-client" target="_blank">Neptune client</a>
