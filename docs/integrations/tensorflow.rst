Neptune-Tensorflow Integration
===============================

|neptune-tensorboard|, an open source project curated by the Neptune team, integrates |tensorboard| with Neptune to let you enjoy the best of both worlds.

.. image:: ../_static/images/integrations/tensorboard_neptuneml.png
   :target: ../_static/images/integrations/tensorboard_neptuneml.png
   :alt: organize TensorBoard experiments in Neptune

Neptune supports two major use cases:

1. With |neptune-tensorboard| you can have your TensorBoard visualizations hosted in Neptune. See `how to sync TensorBoard logdir <tensorboard.html>`_.
2. You can use the ``neptune.integrate_with_tensorflow()`` method to instantly log major metrics (epoch and batch related) to Neptune. For example, `Integrate with TensorBoard logging <tensorboard/tensorboard_integrate.html>`_.

Check the example |tensorboard-integration| project in Neptune.

Installation
------------

.. code-block:: bash

    pip install neptune-tensorboard

Requirements
------------
Create a simple training script with TensorBoard logging. This example uses TensorFlow version 1.x,
however, neptune-tensorboard works well with both TensorFlow 1 and TensorFlow 2.

.. code-block:: python3

    import random

    import tensorflow as tf

    PARAMS = {
        'epoch_nr': 5,
        'batch_size': 256,
        'lr': 0.1,
        'momentum': 0.4,
        'use_nesterov': True,
        'unit_nr': 256,
        'dropout': 0.0
    }

    RUN_NAME = 'run_{}'.format(random.getrandbits(64))
    EXPERIMENT_LOG_DIR = 'logs/{}'.format(RUN_NAME)

    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(PARAMS['unit_nr'], activation=tf.nn.relu),
      tf.keras.layers.Dropout(PARAMS['dropout']),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    optimizer = tf.keras.optimizers.SGD(lr=PARAMS['lr'],
                                        momentum=PARAMS['momentum'],
                                        nesterov=PARAMS['use_nesterov'],)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=EXPERIMENT_LOG_DIR)
    model.fit(x_train, y_train,
              epochs=PARAMS['epoch_nr'],
              batch_size=PARAMS['batch_size'],
              callbacks=[tensorboard])

Initialize Neptune
------------------
.. code-block:: python3

    import neptune
    neptune.init(project_qualified_name='jakub-czakon/tensorboard-integration')

Integrate with TensorBoard
--------------------------
Here you import our method that automatically logs TensorBoard metrics.

.. code-block:: python3

    import neptune_tensorboard as neptune_tb
    neptune_tb.integrate_with_tensorflow()

Start experiment
----------------
Tell Neptune to create an experiment. Give it a name and log hyperparameters.
It is recommended to have everything in the ``with`` statement if possible, to enforce auto-clean once the experiment is complete.

.. code-block:: python3

    with neptune.create_experiment(name=RUN_NAME, params=PARAMS):

Explore your experiment in the Neptune dashboard
------------------------------------------------
By adding a few lines of code, your experiment is now logged to Neptune.
You can go see it in your dashboard and share it with anyone, just as we are sharing it with you |here|.

- Overview

    .. image:: ../../_static/images/integrations/tensorboard_example_1.png
        :target: ../../_static/images/integrations/tensorboard_example_1.png
        :alt: experiment in the experiment table

- Monitor learning curves

    .. image:: ../../_static/images/integrations/tensorboard_example_2.png
        :target: ../../_static/images/integrations/tensorboard_example_2.png
        :alt: experiment in the experiment table

- Monitor hardware utilization

    .. image:: ../../_static/images/integrations/tensorboard_example_3.png
        :target: ../../_static/images/integrations/tensorboard_example_3.png
        :alt: experiment in the experiment table

- Check the source code

    .. image:: ../../_static/images/integrations/tensorboard_example_4.png
        :target: ../../_static/images/integrations/tensorboard_example_4.png
        :alt: experiment in the experiment table

Full script
-----------
Simply copy and paste it to ``tensorflow_example.py`` and run.

.. code-block:: python3

    import random

    import neptune
    import neptune_tensorboard as neptune_tb
    import tensorflow as tf

    neptune.init(project_qualified_name='USER_NAME/PROJECT_NAME')
    neptune_tb.integrate_with_tensorflow()

    PARAMS = {
        'epoch_nr': 5,
        'batch_size': 256,
        'lr': 0.1,
        'momentum': 0.4,
        'use_nesterov': True,
        'unit_nr': 256,
        'dropout': 0.0
    }
    RUN_NAME = 'run_{}'.format(random.getrandbits(64))
    EXPERIMENT_LOG_DIR = 'logs/{}'.format(RUN_NAME)

    with neptune.create_experiment(name=RUN_NAME, params=PARAMS):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(PARAMS['unit_nr'], activation=tf.nn.relu),
          tf.keras.layers.Dropout(PARAMS['dropout']),
          tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        optimizer = tf.keras.optimizers.SGD(lr=PARAMS['lr'],
                                            momentum=PARAMS['momentum'],
                                            nesterov=PARAMS['use_nesterov'],)

        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=EXPERIMENT_LOG_DIR)
        model.fit(x_train, y_train,
                  epochs=PARAMS['epoch_nr'],
                  batch_size=PARAMS['batch_size'],
                  callbacks=[tensorboard])

.. External links

.. |GitHub| raw:: html

    <a href="https://github.com/neptune-ai/neptune-tensorboard" target="_blank">GitHub</a>

.. |here| raw:: html

    <a href="https://ui.neptune.ai/jakub-czakon/tensorboard-integration/e/TEN-41/charts" target="_blank">here</a>

.. |neptune-tensorboard| raw:: html

    <a href="https://github.com/neptune-ai/neptune-tensorboard" target="_blank">neptune-tensorboard</a>


.. |tensorboard| raw:: html

    <a href="https://www.tensorflow.org/guide/summaries_and_tensorboard" target="_blank">TensorBoard</a>


.. |tensorboard-integration| raw:: html

    <a href="https://ui.neptune.ai/jakub-czakon/tensorboard-integration/experiments" target="_blank">TensorBoard project</a>


.. |experiment-link| raw:: html

    <a href="https://ui.neptune.ai/jakub-czakon/tensorboard-integration/compare?shortId=%5B%22TEN-41%22%2C%22TEN-40%22%2C%22TEN-39%22%2C%22TEN-38%22%2C%22TEN-37%22%2C%22TEN-36%22%2C%22TEN-35%22%2C%22TEN-34%22%2C%22TEN-33%22%2C%22TEN-32%22%5D" target="_blank">experiment link</a>
