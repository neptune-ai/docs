Keras
=====

Log keras metrics
-----------------
I have a training script written in `keras <https://keras.io>`_. How do I adjust it to log metrics to Neptune?

.. image:: ../_static/images/others/keras_neptuneml.png
   :target: ../_static/images/others/keras_neptuneml.png
   :alt: Keras neptune.ai integration

**Step 1**

Say your training script looks like this:

.. code-block::

   import keras
   from keras import backend as K

   mnist = keras.datasets.mnist
   (x_train, y_train),(x_test, y_test) = mnist.load_data()
   x_train, x_test = x_train / 255.0, x_test / 255.0

   model = keras.models.Sequential([
     keras.layers.Flatten(),
     keras.layers.Dense(512, activation=K.relu),
     keras.layers.Dropout(0.2),
     keras.layers.Dense(10, activation=K.softmax)
   ])
   model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

   model.fit(x_train, y_train, epochs=5)

**Step 2**

Now let's use Keras Callback

.. code-block::

   from keras.callbacks import Callback

   class NeptuneMonitor(Callback):
       def on_epoch_end(self, epoch, logs={}):
           innovative_metric = logs['acc'] - 2 * logs['loss']
           neptune.send_metric('innovative_metric', epoch, innovative_metric)

**Step 3**

Instantiate it and add it to your callbacks list:

.. code-block::

   with neptune.create_experiment():
       neptune_monitor = NeptuneMonitor()
       model.fit(x_train, y_train, epochs=5, callbacks=[neptune_monitor])

All your metrics are now logged to Neptune:

.. image:: ../_static/images/how-to/ht-log-keras-1.png
   :target: ../_static/images/how-to/ht-log-keras-1.png
   :alt: image
