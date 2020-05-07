Sync and Compare Runs
=====================

Preparation
-----------
Import libraries and define hyperparameters. Of course, you can pass hyperparameters from ``argparse`` or ``click`` as well.

.. code-block:: python3

    import tensorflow as tf
    import mlflow

    PARAMS = {
        'epoch_nr': 5,
        'batch_size': 256,
        'lr': 0.1,
        'momentum': 0.9,
        'use_nesterov': True,
        'unit_nr': 512,
        'dropout': 0.25
    }

Start experiment and log hyperparameters
-----------------------------------------
It's recommended to have everything in the ``with`` statement if possible, to enforce auto-clean once the experiment is complete.

.. code-block:: python3

    with mlflow.start_run():
        for name, value in PARAMS.items():
            mlflow.log_param(name, value)

Train your model and log metrics
--------------------------------
.. code-block:: python3

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

    model.fit(x_train, y_train,
              epochs=PARAMS['epoch_nr'],
              batch_size=PARAMS['batch_size'])

    # log metrics
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_acc)

Sync MLRuns with Neptune
------------------------
You can now sync your MLRuns directory with Neptune.

.. code-block:: python3

    neptune mlflow --project USER_NAME/PROJECT_NAME


You can now organize and collaborate on |your experiments|.


.. image:: ../../_static/images/mlflow/mlflow_1.png
   :target: ../../_static/images/mlflow/mlflow_1.png
   :alt: organize MLflow experiments in Neptune


.. External Links

.. |your experiments| raw:: html

    <a href="https://ui.neptune.ai/jakub-czakon/mlflow-integration/experiments?viewId=817b9095-103e-11ea-9a39-42010a840083" target="_blank">your experiments</a>
