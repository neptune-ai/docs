Neptune-Keras Integration
=========================

Neptune has implemented an integration with the Keras neural-network library.

Integration with Keras is enabled through the |neptune-tensorboard| package. It lets you automatically track metrics and losses (on *batch end* and *epoch end*).


.. image:: ../_static/images/others/keras_neptuneml.png
   :target: ../_static/images/others/keras_neptuneml.png
   :alt: Keras neptune.ai integration


Installation
^^^^^^^^^^^^
.. code-block:: bash

    pip install neptune-tensorboard

Usage
^^^^^
From now on, you can integrate with Keras. An integration snippet is presented below. It should be executed before you create a Neptune experiment, using the :meth:`~neptune.projects.Project.create_experiment` method.

.. code-block:: python3

    import neptune_tensorboard as neptune_tb
    neptune_tb.integrate_with_keras()

As a result, all metrics and losses are automatically tracked in Neptune.

.. image:: ../_static/images/how-to/ht-log-keras-1.png
   :target: ../_static/images/how-to/ht-log-keras-1.png
   :alt: image

.. note::

    Check for more examples in the |keras-integration| Neptune project.

Full script
^^^^^^^^^^^
.. code-block:: python3

    # imports
    import random
    import neptune
    import keras
    import neptune_tensorboard as neptune_tb

    # set project and start integration with keras
    neptune.init(api_token='ANONYMOUS',
                 project_qualified_name='shared/keras-integration')
    neptune_tb.integrate_with_keras()

    # parameters
    PARAMS = {'epoch_nr': 5,
              'batch_size': 256,
              'lr': 0.005,
              'momentum': 0.4,
              'use_nesterov': True,
              'unit_nr': 256,
              'dropout': 0.05}

    # start experiment
    neptune.create_experiment(name='keras-integration-example', params=PARAMS)

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

.. External links

.. |neptune-tensorboard| raw:: html

    <a href="https://docs.neptune.ai/integrations/tensorboard.html" target="_blank">neptune-tensorboard</a>

.. |keras-integration| raw:: html

    <a href="https://ui.neptune.ai/shared/keras-integration/experiments" target="_blank">keras-integration</a>
