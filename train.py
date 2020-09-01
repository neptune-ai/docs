import keras
import neptune

# set project
neptune.init(api_token='ANONYMOUS',
             project_qualified_name='shared/onboarding')

# parameters
PARAMS = {'epoch_nr': 5,
          'batch_size': 256,
          'lr': 0.005,
          'momentum': 0.4,
          'use_nesterov': True,
          'unit_nr': 256,
          'dropout': 0.05}

# start experiment
neptune.create_experiment(name='great-idea')


class NeptuneMonitor(keras.callbacks.Callback):
    def on_epoch_end(self, logs={}):
        for metric_name, metric_value in logs.items():
            neptune.log_metric(metric_name, metric_value)


mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(PARAMS['unit_nr'], activation=keras.activations.relu),
    keras.layers.Dropout(PARAMS['dropout']),
    keras.layers.Dense(10, activation=keras.activations.softmax)
])

optimizer = keras.optimizers.SGD(lr=PARAMS['lr'],
                                 momentum=PARAMS['momentum'],
                                 nesterov=PARAMS['use_nesterov'], )

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=PARAMS['epoch_nr'],
          batch_size=PARAMS['batch_size'],
          callbacks=[NeptuneMonitor()])