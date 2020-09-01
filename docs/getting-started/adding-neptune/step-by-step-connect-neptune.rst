How to connect Neptune to your Python codebase step by step
===========================================================

Adding Neptune is a simple process that only takes a few steps.
We'll go through those one by one.

Introduction
------------

This guide will show you how to:

- Connect Neptune to your script from scratch
- Find ML framework integrations to start tracking your runs with even less work
- Find tracking tool integrations to sync/convert experiment information from your current tool to Neptune experiments

By the end of it, you will have experiments from your project logged and versioned in Neptune!

Before you start
----------------

Make sure you meet the following prerequisites before starting:

- Have Python 3.x installed
- |Have Neptune installed|
- |Create a project|
- |Configure Neptune API token on your system|

Step 1: Choose a language: Are you using Python?
------------------------------------------------

Neptune was built for Python first and if this is your language of choice jump to the next Step!

If you are not using Python, no worries, Neptune plays nicely with other languages as well.

Read |how to use Neptune with other languages| here.

Step 2: See if Neptune integrates with your current experiment tracking tool
----------------------------------------------------------------------------

Neptune has utilities that let you use other open-source experiment tracking tools together with Neptune
They also make the migration from those tools easy and quick.

Neptune integrates with the following experiment tracking frameworks:

- |MLflow|
- |TensorBoard|
- |Sacred|

Step 3: See if Neptune integrates with your ML framework
--------------------------------------------------------

Neptune supports any machine learning framework but there are a lot of integrations with particular frameworks that will get you started in minutes.

Deep learning frameworks:
^^^^^^^^^^^^^^^^^^^^^^^^^

- |Keras|
- |PyTorch|
- |PyTorch Lightning|
- |PyTorch Ignite|
- |Catalyst|
- |Fastai|
- |Skorch|

Machine learning frameworks:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- |lightGBM|
- |XGBoost|

Hyperparameter Optimization frameworks:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- |Optuna|
- |Scikit-Optimize|

Check out the |full list of integrations|.

Step 4: Add Neptune logging explicitly
--------------------------------------

1. Connect Neptune to your script

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

    - |Configure Neptune API token on your system|
    - |Create a project|

    You can skip ``api_token`` and change the ``project_qualified_name`` to your ``USERNAME`` and ``PROJECT_NAME``

    .. code:: python

        neptune.init(project_qualified_name='USERNAME/PROJECT_NAME')

2. Create an experiment and log parameters

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

3. Add logging of training metrics

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

4. Add logging of test metrics

.. code:: python

    neptune.log_metric('test-accuracy', 0.82)

You can log metrics in the same way after the training loop is done.

.. note::

    You can also update experiments after the script is done running.

    Read about |updating existing experiments|.

5. Add logging of performance charts

.. code:: python

    neptune.log_image('predictions', 'pred_img.png')
    neptune.log_image('performance charts', fig)

.. tip::

    There are many other object that you can log to Neptune.
    You may want to read our articles on:

    - See |how to log other objects and monitor training in Neptune|

6. Add logging of model binary

.. code:: python

    neptune.log_artifact('model.pkl')

You save your model to a file and log that file to Neptune.

.. tip::

    There is a helper function in neptune-contrib called |log pickle| for logging picklable Python objects without saving them to disk.

    It works like this:

    .. code:: python

        from neptunecontrib.api import log_pickle

        log_pickle(model)

Run your script and see your experiment in Neptune UI
-----------------------------------------------------

|Logging video|

What is next?
-------------

- See |how to log other objects and monitor training in Neptune|
- See |how to track hyperparameters of ML models|
- See |how to track metrics and losses|
- See |how to monitor ML/DL experiments|
- Check the |full list of integrations|

.. |installation guides| raw:: html

    <a href="/getting-started/installation/index.html">installation guides</a>

.. |how to use Neptune with other languages| raw:: html

    <a href="/getting-started/adding-neptune/not-using-python.html">how to use Neptune with other languages</a>

.. |MLflow| raw:: html

    <a href="/integrations/mlflow.html">MLflow</a>

.. |TensorBoard| raw:: html

    <a href="/integrations/tensorboard.html">TensorBoard</a>

.. |Sacred| raw:: html

    <a href="/integrations/sacred.html">Sacred</a>

.. |Logging video| raw:: html

    <iframe width="720" height="420" src="https://www.youtube.com/embed/of4Q7TkUAVA" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

.. |Keras| raw:: html

    <a href="/integrations/keras.html">Keras</a>

.. |PyTorch| raw:: html

    <a href="/integrations/pytorch.html">PyTorch</a>

.. |PyTorch Lightning| raw:: html

    <a href="/integrations/pytorch_lightning.html">PyTorch Lightning</a>

.. |PyTorch Ignite| raw:: html

    <a href="/integrations/pytorch_ignite.html">PyTorch Ignite</a>

.. |Catalyst| raw:: html

    <a href="/integrations/catalyst.html">Catalyst</a>

.. |Fastai| raw:: html

    <a href="/integrations/fastai.html">Fastai</a>

.. |Skorch| raw:: html

    <a href="/integrations/skorch.html">Skorch</a>

.. |lightGBM| raw:: html

    <a href="/integrations/lightgbm.html">lightGBM</a>

.. |XGBoost| raw:: html

    <a href="/integrations/xgboost.html">XGBoost</a>

.. |Optuna| raw:: html

    <a href="/integrations/optuna.html">Optuna</a>

.. |Scikit-Optimize| raw:: html

    <a href="/integrations/skopt.html">Scikit-Optimize</a>


.. |full list of integrations| raw:: html

    <a href="/integrations/index.html">full list of integrations</a>

.. |how to log other objects and monitor training in Neptune| raw:: html

    <a href="https://neptune.ai/blog/monitoring-machine-learning-experiments-guide" target="_blank">how to log other objects and monitor training in Neptune</a>


.. |how to track hyperparameters of ML models| raw:: html

    <a href="https://neptune.ai/blog/how-to-track-hyperparameters" target="_blank">how to track hyperparameters of ML models</a>

.. |how to track metrics and losses| raw:: html

    <a href="https://neptune.ai/blog/how-to-track-machine-learning-model-metrics" target="_blank">how to track metrics and losses</a>

.. |how to monitor ML/DL experiments| raw:: html

    <a href="https://neptune.ai/blog/how-to-monitor-machine-learning-and-deep-learning-experiments" target="_blank">how to monitor ML/DL experiments</a>

.. |updating existing experiments| raw:: html

    <a href="/logging-data-to-neptune/updating/index.html">full list of integrations</a>

.. |log pickle| raw:: html

    <a href="/api-reference/neptunecontrib/api/index.html?highlight=log%20pickle#neptunecontrib.api.log_pickle">log_pickle</a>

.. |Have Neptune installed| raw:: html

    <a href="/getting-started/installation/index.html">Have Neptune installed</a>

.. |Create a project| raw:: html

    <a href="/teamwork-and-user-management/how-to/create-project.html" target="_blank">Create a project in Neptune</a>

.. |Configure Neptune API token on your system| raw:: html

    <a href="/security-privacy/api-tokens/how-to-api-token.html" target="_blank">Configure Neptune API token on your system</a>