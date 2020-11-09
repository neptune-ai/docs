.. _guides-logging-and-managing-experiment-results:

Logging and managing experiment results
=======================================
What is neptune-client?
-----------------------
|neptune-client-github| is an open source Python library that serves two purposes:

1. logging machine learning experiments,
2. fetching experiment data from Neptune to local machine.

It is designed to be **lightweight** (low setup effort), **generic** (capable of logging any kind of machine learning work) and **straightforward** (user defines what to keep track of during experiment) to use.

What you need to add to your code to start logging
--------------------------------------------------
[loom-placeholder]

Not much really. Bare minimum are one import and two methods:

.. code-block:: python3

    # import library
    import neptune

    # set project
    neptune.init('my_workspace/my_project')

    # create experiment
    neptune.create_experiment()

These are usually just copy&paste into your existing project code. Now, you are ready to start logging metrics, losses, images, model weights or whatever you feel relevant to keep track of in your experiment.

Overall idea for logging is to use methods similar to this:

.. code-block:: python3

    neptune.log_metric('acc', 0.95)

Generic recipe being:

.. code-block:: python3

    neptune.log_X('some_name', some_value)

Check the :ref:`logging section <what-you-can-log>` for a complete list.

.. note::

    If you work in Notebooks, you need to place ``neptune.stop()`` at the very end of your experiment to make sure that everything will be closed properly.

    Note that you are still able to :ref:`update an experiment <update-existing-experiment>` that was closed before.

Essential parts
---------------
[loom-placeholder]

Snippet above sets project and creates experiment. Indeed, :ref:`Project <logging_project>` and :ref:`Experiment <logging_experiment>` are two important entities in Neptune. Read on, to learn a bit about them.

.. _logging_project:

Project
^^^^^^^
It is a **collection of Experiments**, created by user(s) assigned to the project.

.. note::

    Always use ``neptune.init('my_workspace/my_project')`` to set project that you will be logging into. Check |docs-neptune-init| for more details.

Learn more about |docs-project| to use in your Python code.

.. _logging_experiment:

Experiment
^^^^^^^^^^
Experiment is everything that you log to Neptune, beginning at ``neptune.create_experiment()`` and ending when script finishes.

You can log experiments from :ref:`anywhere <execution-environments-index>` (local machine, cluster, cloud, Colab, etc.) and they will be tracked in the same, standardized way. You will be able to quickly compare experiments run by you on your workstation with experiments run on AWS by your team-mate.

.. note::

    ``neptune.log_metric('some_name', some_value)`` is for tracking all numeric values to Neptune (metric, loss, score, variances, etc.). Learn, what else can be tracked to experiment from :ref:`this list <what-you-can-log>`.

Learn more about |docs-experiment| to use in your Python code.

What you can do?
----------------
You can think of three main actions around experiments:

#. **Log experiments** - explained above, where you log metrics and other data to the experiment
#. **Update experiments** - you can log more data to previously closed experiment. Here is how to :ref:`update experiment <update-existing-experiment>`
#. **Download experiments** - all logged data can be :ref:`downloaded programmatically <guides-download_data>`.


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

.. |docs-neptune-init| raw:: html

    <a href="https://docs.neptune.ai/api-reference/neptune/index.html#neptune.init" target="_blank">docs</a>

.. |docs-project| raw:: html

    <a href="https://docs.neptune.ai/api-reference/neptune/projects/index.html#neptune.projects.Project" target="_blank">Project methods</a>

.. |docs-experiment| raw:: html

    <a href="https://docs.neptune.ai/api-reference/neptune/experiments/index.html#neptune.experiments.Experiment" target="_blank">Experiment methods</a>
