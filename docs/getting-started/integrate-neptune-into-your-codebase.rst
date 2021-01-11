Integrating Neptune into your codebase
======================================

Adding Neptune to your workflow is a really simple and quick process.
We describe major logging features in the step by step guide below.

To make things even easier we have created integrations with most major ML frameworks and open-source experiment tracking tools.

Jump to the relevant section:

- :ref:`Using integrations with ML frameworks <integrating-with-my-framework>`
- :ref:`Migrating from other experiment tracking tools <migrating-from-other-experiment-tracking-tools>`
- :ref:`Not using Python <not-using-python>`
- :ref:`How to connect Neptune to your codebase step by step <how-to-connect-neptune-to-your-codebase>`

.. _integrating-with-my-framework:

Using integrations with ML frameworks
-------------------------------------

Neptune supports any machine learning framework but there are a lot of integrations with particular frameworks that will get you started faster.

Popular integrations include:

- :ref:`TensorFlow / Keras <integrations-tensorflow-keras>`
- :ref:`PyTorch <integrations-pytorch>`
- :ref:`PyTorch Lightning <integrations-pytorch-lightning>`
- :ref:`XGBoost <integrations-xgboost>`
- :ref:`lightGBM <integrations-lightgbm>`
- :ref:`Optuna <integrations-optuna>`

Check out the :ref:`full list of integrations <list-of-all-integrations>`.

.. _migrating-from-other-experiment-tracking-tools:

Migrating from other experiment tracking tools
----------------------------------------------

Neptune has utilities that let you use other open-source experiment tracking tools together with Neptune
They also make the migration from those tools easy and quick.

Neptune integrates with the following experiment tracking frameworks:

- :ref:`MLflow <integrations-mlflow>`
- :ref:`TensorBoard <integrations-tensorboard>`
- :ref:`Sacred <integrations-sacred>`

.. _not-using-python:

Not using Python
----------------

If you are not using Python no worries, you can still log experiments to Neptune.

Read our guides on:

- How to use :ref:`Neptune client for R <integrations-r>`
- How to log experiments for :ref:`any other language <integrations-any-language>`

.. _how-to-connect-neptune-to-your-codebase:

How to connect Neptune to your codebase step by step
----------------------------------------------------

Adding Neptune is a simple process that only takes a few steps.
We'll go through those one by one.

Before you start
^^^^^^^^^^^^^^^^

Make sure you meet the following prerequisites before starting:

- Have Python 3.x installed
- :ref:`Have Neptune installed<installation-neptune-client>`
- :ref:`Create a project <create-project>`
- :ref:`Configure Neptune API token on your system <how-to-setup-api-token>`


Step 1: Connect Neptune client to your script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    import neptune

    neptune.init(project_qualified_name='shared/onboarding',
                 api_token='ANONYMOUS',
                 )

You need to tell Neptune who you are and where you want to log things.

To do that you should specify:

- ``project_qualified_name=USERNAME/PROJECT_NAME``: Neptune username and project
- ``api_token=YOUR_API_TOKEN``: your Neptune API token.

.. note::

    If you followed suggested prerequisites:

    - :ref:`Configure Neptune API token on your system <how-to-setup-api-token>`
    - :ref:`Create a project <create-project>`

    You can skip ``api_token`` and change the ``project_qualified_name`` to your ``USERNAME`` and ``PROJECT_NAME``

    .. code:: python

        neptune.init(project_qualified_name='USERNAME/PROJECT_NAME')

Step 2. Create an experiment and log parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    PARAMS = {'lr': 0.1, 'epoch_nr': 10, 'batch_size': 32}
    neptune.create_experiment(name='great-idea', params=PARAMS)

This opens a new "experiment" namespace in Neptune to which you can log various objects.
It also logs your ``PARAMS`` dictionary with all the parameters that you want to keep track of.

.. note::

    Right now parameters can only be passed at experiment creation.

.. tip::

    You may want to read our article on:

    - See |how to track hyperparameters of ML models|

Step 3. Add logging of training metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    neptune.log_metric('loss', 0.26)

The first argument is the name of the log. You can have one or multiple log names (like 'acc', 'f1_score', 'log-loss', 'test-acc').
The second argument is the value of the log.

Typically during training there will be some sort of a loop where those losses are logged.
You can simply call ``neptune.log_metric`` multiple times on the same log name to log it at each step.

.. code:: python

    for i in range(epochs):
        ...
        neptune.log_metric('loss', loss)
        neptune.log_metric('metric', accuracy)

.. note::

    You can specifically log value at given step by using ``x`` and ``y`` arguments.

    .. code:: python

        neptune.log_metric('loss', x=12, y=0.32)

.. tip::

    You may want to read our articles on:

    - See |how to log other objects and monitor training in Neptune|
    - See |how to track metrics and losses|
    - See |how to monitor ML/DL experiments|

Step 4. Add logging of test metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    neptune.log_metric('test-accuracy', 0.82)

You can log metrics in the same way after the training loop is done.

.. note::

    You can also update experiments after the script is done running.

    Read about :ref:`updating existing experiments <update-existing-experiment>`.

Step 5: Add logging of performance charts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    neptune.log_image('predictions', 'pred_img.png')
    neptune.log_image('performance charts', fig)

.. tip::

    There are many other object that you can log to Neptune.
    You may want to read our articles on:

    - See |how to log other objects and monitor training in Neptune|

Step 6: Add logging of model binary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    neptune.log_artifact('model.pkl')

You save your model to a file and log that file to Neptune.

.. tip::

    There is a helper function in neptune-contrib called log pickle for logging picklable Python objects without saving them to disk.

    It works like this:

    .. code:: python

        from neptunecontrib.api import log_pickle

        log_pickle(model)

Step 7: Run your script and see your experiment in Neptune UI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|Logging video|

What is next?
-------------

- See :ref:`Full list of what you can log <what-you-can-log>`
- Check out the :ref:`full list of integrations <list-of-all-integrations>`
- Read |how to log other objects and monitor training in Neptune|

.. External links

.. |Logging video| raw:: html

    <iframe width="720" height="420" src="https://www.youtube.com/embed/of4Q7TkUAVA" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

.. |how to log other objects and monitor training in Neptune| raw:: html

    <a href="https://neptune.ai/blog/monitoring-machine-learning-experiments-guide" target="_blank">how to log other objects and monitor training in Neptune</a>

.. |how to track hyperparameters of ML models| raw:: html

    <a href="https://neptune.ai/blog/how-to-track-hyperparameters" target="_blank">how to track hyperparameters of ML models</a>

.. |how to track metrics and losses| raw:: html

    <a href="https://neptune.ai/blog/how-to-track-machine-learning-model-metrics" target="_blank">how to track metrics and losses</a>

.. |how to monitor ML/DL experiments| raw:: html

    <a href="https://neptune.ai/blog/how-to-monitor-machine-learning-and-deep-learning-experiments" target="_blank">how to monitor ML/DL experiments</a>