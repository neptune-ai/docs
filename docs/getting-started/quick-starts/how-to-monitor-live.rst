.. _use-cases-monitor-runs-live:

How to monitor ML runs live: step by step guide
===============================================

|run on colab button|

Introduction
------------

This guide will show you how to:

* Monitor training and evaluation metrics and losses live
* Monitor hardware resources during training

By the end of it, you will monitor your metrics, losses, and hardware live in Neptune!

Before you start
----------------

Make sure you meet the following prerequisites before starting:

- Have Python 3.x installed
- Have Tensorflow 2.x with Keras installed
- :ref:`Have Neptune installed <installation-neptune-client>`
- :ref:`Create a project <create-project>`
- :ref:`Configure Neptune API token on your system <how-to-setup-api-token>`

.. note::

    You can run this how-to on Google Colab with zero setup.

    Just click on the ``Open in Colab`` button on the top of the page.

Step 1: Create a basic training script
--------------------------------------

As an example I'll use a script that trains a Keras model on mnist dataset.

.. note::

    You **don't have to use Keras** to monitor your training runs live with Neptune.

    I am using it as an easy to follow example.

    There are links to integrations with other ML frameworks and useful articles about monitoring in the text.

1. Create a file ``train.py`` and copy the script below.

``train.py``

.. code:: python

    import keras

    PARAMS = {'epoch_nr': 100,
              'batch_size': 256,
              'lr': 0.005,
              'momentum': 0.4,
              'use_nesterov': True,
              'unit_nr': 256,
              'dropout': 0.05}

    mnist = keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = keras.models.Sequential([
      keras.layers.Flatten(),
      keras.layers.Dense(PARAMS['unit_nr'], activation=keras.activations.relu),
      keras.layers.Dropout(PARAMS['dropout']),
      keras.layers.Dense(10, activation=keras.activations.softmax)
    ])

    optimizer = keras.optimizers.SGD(lr=PARAMS['lr'],
                                     momentum=PARAMS['momentum'],
                                     nesterov=PARAMS['use_nesterov'],)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=PARAMS['epoch_nr'],
              batch_size=PARAMS['batch_size'])

2. Run training to make sure that it works correctly.

.. code:: bash

   python train.py

Step 2: Install psutil
----------------------

To monitor hardware consumption in Neptune you need to have ``psutil`` installed.

**pip**

.. code:: bash

    pip install psutil

**conda**

.. code:: bash

    conda install -c anaconda psutil

Step 3: Connect Neptune to your script
--------------------------------------

At the top of your script add

.. code:: python

    import neptune

    neptune.init(project_qualified_name='shared/onboarding',
                 api_token='ANONYMOUS',
                 )

You need to tell Neptune who you are and where you want to log things.

To do that you specify:

- ``project_qualified_name=USERNAME/PROJECT_NAME``: Neptune username and project
- ``api_token=YOUR_API_TOKEN``: your Neptune API token.

.. note::

    If you configured your Neptune API token correctly, as described in :ref:`Configure Neptune API token on your system <how-to-setup-api-token>`, you can skip ``api_token`` argument:

    .. code:: python

        neptune.init(project_qualified_name='YOUR_USERNAME/YOUR_PROJECT_NAME')

Step 4. Create an experiment
----------------------------

.. code:: python

    neptune.create_experiment(name='great-idea')

This opens a new "experiment" namespace in Neptune to which you can log various objects.

Step 5. Add logging for metrics and losses
------------------------------------------

To log a metric or loss to Neptune you should use ``neptune.log_metric`` method:

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

Many frameworks, like Keras, let you create a callback that is executed inside of the training loop.

Now that you know all this.

**Steps for Keras**

1. Create a Neptune callback.

.. code:: python

    class NeptuneMonitor(keras.callbacks.Callback):
         def on_epoch_end(self, epoch, logs=None):
              for metric_name, metric_value in logs.items():
                   neptune.log_metric(metric_name, metric_value)

2. Pass callback to the ``model.fit()`` method:

.. code:: python

   model.fit(x_train, y_train,
              epochs=PARAMS['epoch_nr'],
              batch_size=PARAMS['batch_size'],
              callbacks=[NeptuneMonitor()])

.. note::

    You don't actually have to implement this callback yourself and can use the Callback that we created for Keras.
    It is one of many integrations with ML frameworks that Neptune has.

    - Check our :ref:`TensorFlow / Keras integration <integrations-tensorflow-keras>`

.. tip::

    You may want to read our article on monitoring ML/DL experiments:

    - |How to Monitor Machine Learning and Deep Learning Experiments|

Step 6. Run your script and see results in Neptune
--------------------------------------------------

Run training script.

.. code:: bash

   python train.py

If it worked correctly you should see:

- a link to Neptune experiment. Click on it and go to the app
- metrics and losses in the ``Logs`` and ``Charts`` sections of the UI
- hardware consumption and console logs in the ``Monitoring`` section of the UI

|Youtube video|

|run on colab button|

What's next
-----------

Now that you know how to create experiments and log metrics you can learn:

- See :ref:`what objects you can log to Neptune <what-you-can-log>`
- See :ref:`how to connect Neptune to your codebase <how-to-connect-neptune-to-your-codebase>`
- :ref:`Check our integrations <integrations-index>` with other frameworks

Other useful articles:

- |How to Monitor Machine Learning and Deep Learning Experiments|
- |How to Track Machine Learning Model Metrics in Your Projects|
- |How to Keep Track of PyTorch Lightning Experiments with Neptune|

.. External links

.. |how to log other objects and monitor training in Neptune| raw:: html

    <a href="https://neptune.ai/blog/monitoring-machine-learning-experiments-guide" target="_blank">how to log other objects and monitor training in Neptune</a>

.. |How to Monitor Machine Learning and Deep Learning Experiments| raw:: html

    <a href="https://neptune.ai/blog/how-to-monitor-machine-learning-and-deep-learning-experiments" target="_blank">How to Monitor Machine Learning and Deep Learning Experiments</a>

.. |run on colab button| raw:: html

    <div class="run-on-colab">

        <a target="_blank" href="https://colab.research.google.com//github/neptune-ai/neptune-colab-examples/blob/master/quick-starts/monitor-ml-runs/docs/Monitor-ML-runs-live.ipynb">
            <img width="50" height="50" src="https://neptune.ai/wp-content/uploads/colab_logo_120.png">
            <span>Run in Google Colab</span>
        </a>

        <a target="_blank" href="https://github.com/neptune-ai/neptune-examples/blob/master/quick-starts/monitor-ml-runs/docs/Monitor-ML-runs-live.py">
            <img width="50" height="50" src="https://neptune.ai/wp-content/uploads/GitHub-Mark-120px-plus.png">
            <span>View source on GitHub</span>
        </a>
    </div>

.. |YouTube video|  raw:: html

    <iframe width="720" height="420" src="https://www.youtube.com/embed/Hzr8E3vmAQM" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

.. |A Complete Guide to Monitoring ML Experiments Live in Neptune| raw:: html

    <a href="https://neptune.ai/blog/monitoring-machine-learning-experiments-guide" target="_blank">A Complete Guide to Monitoring ML Experiments Live in Neptune</a>

.. |How to Keep Track of PyTorch Lightning Experiments with Neptune| raw:: html

    <a href="https://neptune.ai/blog/pytorch-lightning-neptune-integration" target="_blank">How to Keep Track of PyTorch Lightning Experiments with Neptune</a>

.. |How to Track Machine Learning Model Metrics in Your Projects| raw:: html

    <a href="https://neptune.ai/blog/how-to-track-machine-learning-model-metrics" target="_blank">How to Track Machine Learning Model Metrics in Your Projects</a>
