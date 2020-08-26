Experiment tracking capabilities
================================
This example shows more features that Neptune offers and combines them in a single script. Specifically, you will see several methods in action:

* :meth:`~neptune.init`
* :meth:`~neptune.projects.Project.create_experiment`
* :meth:`~neptune.experiments.Experiment.log_metric`
* :meth:`~neptune.experiments.Experiment.log_text`
* :meth:`~neptune.experiments.Experiment.log_artifact`
* :meth:`~neptune.experiments.Experiment.append_tag`
* :meth:`~neptune.experiments.Experiment.set_property`

Copy the code snippet and save it as *example.py*, then run it as usual: ``python example.py``. In this tutorial, we make use of the public ``NEPTUNE_API_TOKEN`` of the public user |Neptuner|. Thus, when started, you can see your experiment at the top of |Experiments view|.

Make sure you have all the dependencies installed. 

Simply run this command:

.. code:: bash

    pip install matplotlib neptune-client numpy tensorflow

TensorFlow code example
-----------------------
Run this code and observe results |online|.

.. code:: Python

    import hashlib
    import os
    import tempfile

    import matplotlib.pyplot as plt
    import neptune
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras


    def log_data(logs):
        neptune.log_metric('epoch_accuracy', logs['accuracy'])
        neptune.log_metric('epoch_loss', logs['loss'])


    def lr_scheduler(epoch):
        if epoch < 10:
            new_lr = PARAMS['learning_rate']
        else:
            new_lr = PARAMS['learning_rate'] * np.exp(0.1 * ((epoch//50)*50 - epoch))

        neptune.log_metric('learning_rate', new_lr)
        return new_lr


    # Select project
    neptune.init('shared/onboarding',
                 api_token='ANONYMOUS')

    # Define parameters
    PARAMS = {'batch_size': 64,
              'n_epochs': 100,
              'shuffle': True,
              'activation': 'elu',
              'dense_units': 128,
              'learning_rate': 0.001,
              'early_stopping': 10,
              'optimizer': 'Adam',
              }

    # Create experiment
    neptune.create_experiment(name='classification_example',
                              description='neural net trained on the FashionMNIST',
                              tags=['classification', 'FashionMNIST'],
                              params=PARAMS)
    # Dataset
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    neptune.set_property('train_images_version', hashlib.md5(train_images).hexdigest())
    neptune.set_property('train_labels_version', hashlib.md5(train_labels).hexdigest())
    neptune.set_property('test_images_version', hashlib.md5(test_images).hexdigest())
    neptune.set_property('test_labels_version', hashlib.md5(test_labels).hexdigest())

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    neptune.set_property('class_names', class_names)

    for j, class_name in enumerate(class_names):
        plt.figure(figsize=(10, 10))
        label_ = np.where(train_labels == j)
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[label_[0][i]], cmap=plt.cm.binary)
            plt.xlabel(class_names[j])
        neptune.log_image('example_images', plt.gcf())

    # Model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(PARAMS['dense_units'], activation=PARAMS['activation']),
        keras.layers.Dense(PARAMS['dense_units'], activation=PARAMS['activation']),
        keras.layers.Dense(PARAMS['dense_units'], activation=PARAMS['activation']),
        keras.layers.Dense(10, activation='softmax')
    ])

    if PARAMS['optimizer'] == 'Adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=PARAMS['learning_rate'],
        )
    elif PARAMS['optimizer'] == 'Nadam':
        optimizer = tf.keras.optimizers.Nadam(
            learning_rate=PARAMS['learning_rate'],
        )
    elif PARAMS['optimizer'] == 'SGD':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=PARAMS['learning_rate'],
        )

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Log model summary
    model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))

    # Train model
    model.fit(train_images, train_labels,
              batch_size=PARAMS['batch_size'],
              epochs=PARAMS['n_epochs'],
              shuffle=PARAMS['shuffle'],
              callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: log_data(logs)),
                         keras.callbacks.EarlyStopping(patience=PARAMS['early_stopping'],
                                                       monitor='accuracy',
                                                       restore_best_weights=True),
                         keras.callbacks.LearningRateScheduler(lr_scheduler)]
              )

    # Log model weights
    with tempfile.TemporaryDirectory(dir='.') as d:
        prefix = os.path.join(d, 'model_weights')
        model.save_weights(os.path.join(prefix, 'model'))
        for item in os.listdir(prefix):
            neptune.log_artifact(os.path.join(prefix, item),
                                 os.path.join('model_weights', item))

    # Evaluate model
    eval_metrics = model.evaluate(test_images, test_labels, verbose=0)
    for j, metric in enumerate(eval_metrics):
        neptune.log_metric('eval_' + model.metrics_names[j], metric)

.. External links

.. |get-started-TF| raw:: html

    <a href="https://www.tensorflow.org/tutorials#get-started-with-tensorflow" target="_blank">Get Started with TensorFlow</a>

.. |online|  raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/onboarding/experiments" target="_blank">online</a>

.. |Experiments view|  raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/onboarding/experiments" target="_blank">Experiments view</a>

.. |Neptuner|  raw:: html

    <a href="https://ui.neptune.ai/o/shared/neptuner>" target="_blank">Neptuner</a>