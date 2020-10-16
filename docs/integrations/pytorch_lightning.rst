.. _integrations-pytorch-lightning:

Neptune-PyTorch Lightning Integration
=====================================

What will you get?
------------------
[VIDEO PLACEHOLDER]

PyTorch Lightning is a lightweight PyTorch wrapper for high-performance AI research. With Neptune integration you can:

* see experiment as it is running,
* log training, validation and testing metrics, and visualize them in Neptune UI,
* log experiment parameters,
* monitor hardware usage,
* log any additional metrics of your choice,
* log performance charts and images,
* save model checkpoints.

Where to start?
---------------
To get started with this integration, follow whatever is most convenient to you:

#. :ref:`Quickstart <quickstart>` below for detailed explanations,
#. Open Colab notebook (badge-link below) with quickstart code and run it as a "`neptuner`" user - zero setup, it just works,
#. View quickstart code as a plain Python script on |script|.

You can also check this public project with example experiments: |project|.

|Run on Colab|

.. note::

    This integration is tested with ``pytorch-lightning==1.0.0`` and current latest, and ``neptune-client==0.4.123`` and current latest.

.. _quickstart:

Quickstart
----------
This quickstart will show you how to log PyTorch Lightning experiments to Neptune using ``NeptuneLogger`` (part of the pytorch-lightning library).

.. _before-you-start-basic:

Before you start
^^^^^^^^^^^^^^^^
You have ``Python 3.x`` and following libraries installed:

* ``neptune-client==0.4.123`` or newer: See :ref:`neptune-client installation guide <installation-neptune-client>`.
* ``pytorch`` and ``torchvision``. See |pytorch-install|.
* ``pytorch-lightning==1.0.0`` or newer. See |lightning-install|.

You also need minimal familiarity with the PyTorch Lightning. Have a look at the "|lightning-guide|" guide to get started.

Step 1: Import Libraries
^^^^^^^^^^^^^^^^^^^^^^^^
Import necessary libraries.

.. code-block:: python3

    import os

    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision import transforms

    import pytorch_lightning as pl

Notice ``pytorch_lightning`` at the bottom.

Step 2: Define Hyper-Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Define Python dictionary with hyper-parameters for model training.

.. code-block:: python3

    PARAMS = {'max_epochs': 3,
              'learning_rate': 0.005,
              'batch_size': 32}

This dictionary will later be passed to the Neptune logger (you will see how to do it in :ref:`step 4 <create-neptune-logger>`), so that you will see hyper-parameters in experiment `Parameters` tab.

Step 3: Define LightningModule and DataLoader
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Implement minimal example of the ``pl.LightningModule`` and simple ``DataLoader``.

.. code-block:: python3

    # pl.LightningModule
    class LitModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(28 * 28, 10)

        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            self.log('train_loss', loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=PARAMS['learning_rate'])

    # DataLoader
    train_loader = DataLoader(MNIST(os.getcwd(), download=True, transform=transforms.ToTensor()),
                              batch_size=PARAMS['batch_size'])

Few explanations here:

* Cross entropy logging is defined in the ``training_step`` method in this way:

.. code-block:: python3

    self.log('train_loss', loss)

This loss will be logged to Neptune during training as a ``train_loss``. You will see it in the Experiment's `Charts` tab (as "train_loss" chart) and `Logs` tab (as raw numeric values).

* ``DataLoader`` is a pure PyTorch object.
* Notice, that you pass ``learning_rate`` and ``batch_size`` from the ``PARAMS`` dictionary - all params will be logged as experiment parameters.

.. _create-neptune-logger:

Step 4: Create NeptuneLogger
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Instantiate ``NeptuneLogger`` with necessary parameters.

.. code-block:: python3

    from pytorch_lightning.loggers.neptune import NeptuneLogger

    neptune_logger = NeptuneLogger(
        api_key="ANONYMOUS",
        project_name="shared/pytorch-lightning-integration",
        params=PARAMS)

``NeptuneLogger`` is an object that integrates Neptune with PyTorch Lightning allowing you to track experiments. It's a part of the lightning library. In this minimalist example we use public user `"neptuner"`, who has public token: `"ANONYMOUS"`.

.. tip::

    You can also use your API token. Read more about how to :ref:`securely set Neptune API token <how-to-setup-api-token>`.

Step 5: Pass NeptuneLogger to the Trainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Pass instantiated ``NeptuneLogger`` to the ``pl.Trainer``.

.. code-block:: python3

    trainer = pl.Trainer(max_epochs=PARAMS['max_epochs'],
                         logger=neptune_logger)


Simply pass ``neptune_logger`` to the ``Trainer``, so that lightning will use this logger. Notice, that ``max_epochs`` is from the ``PARAMS`` dictionary.

Step 6: Run experiment
^^^^^^^^^^^^^^^^^^^^^^
Fit model to the data.

.. code-block:: python3

    model = LitModel()

    trainer.fit(model, train_loader)

At this point you are all set to fit the model. Neptune logger will collect metrics and show them in the UI.

Explore Results
^^^^^^^^^^^^^^^
You just learned how to start logging PyTorch Lightning experiments to Neptune, by using Neptune logger which is part of the lightning library.

Above training is logged to Neptune in near real-time. Click on the link that was outputted to the console or |go-here| to explore an experiment similar to yours. In particular check:

#. |metrics|,
#. |params|,
#. |hardware|,
#. |metadata| including git summary info.

.. image:: ../_static/images/integrations/lightning_basic.png
   :target: ../_static/images/integrations/lightning_basic.png
   :alt: PyTorchLightning neptune.ai integration

Check this experiment |exp-link| or view quickstart code as a plain Python script on |script|.

|Run on Colab|

----

Advanced options
----------------
To learn more about advanced options that Neptune logger has to offer, you can either:

#. Check the options below for detailed explanations,
#. Open Colab notebook (badge-link below) and run advanced example as a "`neptuner`" user - zero setup, it just works,
#. View advanced example code as a plain Python script on |script-advanced|.

You can also check this public project with example experiments: |project|.

|Run on Colab Advanced|

Before you start
^^^^^^^^^^^^^^^^
In addition to the contents of the ":ref:`Before you start <before-you-start-basic>`" section in Quickstart, you also need to have ``scikit-learn`` and ``scikit-plot`` installed.

.. code-block:: bash

    pip install scikit-learn==0.23.2 scikit-plot==0.3.7

Check |scikit-learn| or |scikit-plot| for more info.

Step 1: Import Libraries
^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python3

    import os
    import numpy as np

    import torch
    import torch.nn.functional as F
    from torchvision.datasets import MNIST
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from torch.utils.data import random_split
    from torch.optim.lr_scheduler import LambdaLR

    import pytorch_lightning as pl

.. _adv-step-2:

Step 2: Define Hyper-Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Define Python dictionaries with hyper-parameters.

.. code-block:: python3

    LightningModule_Params = {'image_size': 28,
                              'linear': 128,
                              'n_classes': 10,
                              'learning_rate': 0.0023,
                              'decay_factor': 0.95}

    LightningDataModule_Params = {'batch_size': 32,
                                  'num_workers': 4,
                                  'normalization_vector': ((0.1307,), (0.3081,)),}

    LearningRateLogger_Params = {'logging_interval': 'epoch'}

    ModelCheckpoint_Params = {'filepath': 'my_model/checkpoints/{epoch:02d}-{val_loss:.2f}',
                              'save_weights_only': True,
                              'save_top_k': 3}

    Trainer_Params = {'max_epochs': 7,
                      'track_grad_norm': 2,
                      'row_log_interval': 1}

    ALL_PARAMS = {**LightningModule_Params,
                  **LightningDataModule_Params,
                  **LearningRateLogger_Params,
                  **ModelCheckpoint_Params,
                  **Trainer_Params}

* Parameters are grouped into categories that follow the structure of the Pytorch Lightning workflow.
* ``ALL_PARAMS`` dictionary will be logged to Neptune, so that you will see hyper-parameters in the experiment `Parameters` tab.

Step 3: Define LightningModule, LightningDataModule and Callbacks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 3.1: Implement LightningModule
"""""""""""""""""""""""""""""""""""
.. code-block:: python3

    class LitModel(pl.LightningModule):

        def __init__(self, image_size, linear, n_classes, learning_rate, decay_factor):
            super().__init__()
            self.image_size = image_size
            self.linear = linear
            self.n_classes = n_classes
            self.learning_rate = learning_rate
            self.decay_factor = decay_factor

            self.layer_1 = torch.nn.Linear(image_size * image_size, linear)
            self.layer_2 = torch.nn.Linear(linear, n_classes)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.layer_1(x)
            x = F.relu(x)
            x = self.layer_2(x)
            return x

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            scheduler = LambdaLR(optimizer, lambda epoch: self.decay_factor ** epoch)
            return [optimizer], [scheduler]

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            result = pl.TrainResult(loss)
            result.log('train_loss', loss, prog_bar=False)
            return result

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            result = pl.EvalResult(checkpoint_on=loss)
            result.log('val_loss', loss, prog_bar=False)
            return result

        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            result = pl.EvalResult()
            result.log('test_loss', loss, prog_bar=False)
            return result

Few explanations:

* ``LitModule`` will be parametrized by values from appropriate dictionary that was created in :ref:`Step 2 <adv-step-2>`.
* learning rate scheduler is defined in the ``configure_optimizers``. It will change lr values after each epoch. These values will be tracked to Neptune.
* Metrics collected during training, validation and testing will be tracked in Neptune.

Step 3.2: Implement LightningDataModule
"""""""""""""""""""""""""""""""""""""""
.. code-block:: python3

    class MNISTDataModule(pl.LightningDataModule):

        def __init__(self, batch_size, num_workers, normalization_vector):
            super().__init__()
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.normalization_vector = normalization_vector

        def prepare_data(self):
            MNIST(os.getcwd(), train=True, download=True)
            MNIST(os.getcwd(), train=False, download=True)

        def setup(self, stage):
            # transforms
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.normalization_vector[0],
                                     self.normalization_vector[1])
            ])

            if stage == 'fit':
                mnist_train = MNIST(os.getcwd(), train=True, transform=transform)
                self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
            if stage == 'test':
                self.mnist_test = MNIST(os.getcwd(), train=False, transform=transform)

        def train_dataloader(self):
            mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)
            return mnist_train

        def val_dataloader(self):
            mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)
            return mnist_val

        def test_dataloader(self):
            mnist_test = DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)
            return mnist_test

Few notes:

* Similarly to the ``LitModule``, ``MNISTDataModule`` will be parametrized by values from appropriate dictionary that was created in :ref:`Step 2 <adv-step-2>`.
* This module contains dataloaders for training, validation and testing of the model.

### Step 3.3: Implement Callbacks and Create Them
"""""""""""""""""""""""""""""""""""""""""""""""""
Callbacks for model checkpointing and logging learning rate changes.

.. _adv-step-3-callbacks:

.. code-block:: python3

    from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint

    lr_logger = LearningRateLogger(**LearningRateLogger_Params)

    model_checkpoint = ModelCheckpoint(**ModelCheckpoint_Params)

Few notes:

* ``LearningRateMonitor`` will log new value of the learning rate for each epoch (see: :ref:`Step 2 <adv-step-2>`).
* ``ModelCheckpoint`` will save top 3 checkpoints (see: :ref:`Step 2 <adv-step-2>`).

.. _adv-step-4:

Step 4: Create NeptuneLogger
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Instantiate ``NeptuneLogger`` with advanced parameters.

.. code-block:: python3

    from pytorch_lightning.loggers.neptune import NeptuneLogger

    neptune_logger = NeptuneLogger(
        api_key="ANONYMOUS",
        project_name="shared/pytorch-lightning-integration",
        close_after_fit=False,
        experiment_name="train-on-MNIST",
        params=ALL_PARAMS,
        tags=['1.0.0', 'advanced'],
    )

When compared to the :ref:`quickstart example <create-neptune-logger>`, few more options are used:

* ``close_after_fit=False`` -> that will let us log more data after ``Trainer.fit()`` and ``Trainer.test()`` methods,
* ``experiment_name`` and ``tags`` are set. You will use them later in the UI for experiment searching and filtering.

Step 5: Pass NeptuneLogger and Callbacks to the Trainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python3

    from pytorch_lightning import Trainer

    trainer = pl.Trainer(logger=neptune_logger,
                         checkpoint_callback=model_checkpoint,
                         callbacks=[lr_logger],
                         **Trainer_Params)

Notes:

* Besides ``neptune_logger``, callbacks (created :ref:`here <adv-step-3-callbacks>`) are also passed to the trainer.
* Notes that you also used ``Trainer_Params`` defined in the :ref:`Step 2<adv-step-2>`, where you set ``max_epochs`` and specified gradient 2-norm (``track_grad_norm``) for automatic logging to Neptune.

Step 6: Run experiment
^^^^^^^^^^^^^^^^^^^^^^
Step 6.1: Initialize model and data objects
"""""""""""""""""""""""""""""""""""""""""""
.. code-block:: python3

    # init model
    model = LitModel(**LightningModule_Params)

    # init data
    dm = MNISTDataModule(**LightningDataModule_Params)

Step 6.2: Run training
""""""""""""""""""""""
.. code-block:: python3

    trainer.fit(model, dm)

Here, you log training and validation loss, learning rate scheduler values and gradient 2-norm.

Step 6.3: Run testing
"""""""""""""""""""""
.. code-block:: python3

    trainer.test(datamodule=dm)

Here, you log test loss.

Step 7: Run additional actions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 7.1: Log misclassified images
""""""""""""""""""""""""""""""""""
In the test set, identify misclassified images and log them to Neptune.

.. code-block:: python3

    model.freeze()
    test_data = dm.test_dataloader()
    y_true = np.array([])
    y_pred = np.array([])

    for i, (x, y) in enumerate(test_data):
        y = y.cpu().detach().numpy()
        y_hat = model.forward(x).argmax(axis=1).cpu().detach().numpy()

        y_true = np.append(y_true, y)
        y_pred = np.append(y_pred, y_hat)

        for j in np.where(np.not_equal(y, y_hat))[0]:
            img = np.squeeze(x[j].cpu().detach().numpy())
            img[img < 0] = 0
            img = (img / img.max()) * 256
            neptune_logger.experiment.log_image('misclassified_images',
                                                img,
                                                description='y_pred={}, y_true={}'.format(y_hat[j], y[j]))

Last line in the above snippet logs misclassified image to Neptune.

.. tip::

    Use ``neptune_logger.experiment.ABC`` to call methods that you would normally called, when working with neptune client, for example ``log_image`` or ``set_property``.

Step 7.2: Log custom metric
"""""""""""""""""""""""""""
Log test set accuracy to Neptune.

.. code-block:: python3

    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y_true, y_pred)
    neptune_logger.experiment.log_metric('test_accuracy', accuracy)

Step 7.3: Log confusion matrix
""""""""""""""""""""""""""""""
.. code-block:: python3

    import matplotlib.pyplot as plt
    from scikitplot.metrics import plot_confusion_matrix

    fig, ax = plt.subplots(figsize=(16, 12))
    plot_confusion_matrix(y_true, y_pred, ax=ax)
    neptune_logger.experiment.log_image('confusion_matrix', fig)

Step 7.4: Log model checkpoints to Neptune
""""""""""""""""""""""""""""""""""""""""""
.. code-block:: python3

    for k in model_checkpoint.best_k_models.keys():
        model_name = 'checkpoints/' + k.split('/')[-1]
        neptune_logger.experiment.log_artifact(k, model_name)

Step 7.5: Log best model checkpoint score to Neptune
""""""""""""""""""""""""""""""""""""""""""""""""""""
.. code-block:: python3

    neptune_logger.experiment.set_property('best_model_score', model_checkpoint.best_model_score.tolist())

Step 7.6 Log model summary
""""""""""""""""""""""""""
.. code-block:: python3

    for chunk in [x for x in str(model).split('\n')]:
        neptune_logger.experiment.log_text('model_summary', str(chunk))

Step 7.7: Log number of GPU units used
""""""""""""""""""""""""""""""""""""""
.. code-block:: python3

    neptune_logger.experiment.set_property('num_gpus', trainer.num_gpus)

Step 8: Stop Neptune logger
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python3

    neptune_logger.experiment.stop()

In the :ref:`Step 4 <adv-step-4>` we created ``NeptuneLogger`` with ``close_after_fit=False``, so we need to close Neptune experiment explicitly at the end.

Explore Results
^^^^^^^^^^^^^^^

You just learned how to log PyTorch Lightning experiments to Neptune, by using Neptune logger which is part of the lightning library.

Above training is logged to Neptune in near real-time. Click on the link that was outputted to the console or |adv-go-here| to explore an experiment similar to yours.

In particular check:

* train, validation and test metrics visualized as |adv-charts|,
* |adv-parameters|,
* |adv-hardware|,
* |adv-details| including git summary info, best model score, number of GPU units used in experiment.
* |adv-misclassified-images|
* |adv-confusion-matrix|
* |adv-model-checkpoints|
* |adv-model-summary|

Check this experiment (|adv-go-here|) or view above code snippets as a plain Python script on |script-advanced|.

|Run on Colab Advanced|

Common problems
---------------
This integration is tested with ``pytorch-lightning==1.0.0`` and current latest, and ``neptune-client==0.4.123`` and current latest. Make sure that you use correct versions.

How to ask for help?
--------------------
The fastest way is to simply chat with us. Chat icon is located directly in-app, in the lower right corner. Use it!

.. image:: ../_static/images/integrations/chat-icon.png
   :target: ../_static/images/integrations/chat-icon.png
   :alt: Chat icon

For more general questions go to our |forum|.

Other integrations you may like
-------------------------------
Here are other integrations with libraries from the PyTorch ecosystem:

#. |PyTorch|
#. |PyTorch Ignite|
#. |Catalyst|
#. |skorch|

You may also like these two integrations:

#. |optuna|
#. |plotly|


.. External links

.. |register| raw:: html

    <a href="https://neptune.ai/register" target="_blank">register here</a>

.. |project| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/experiments?viewId=202dcc88-c213-4da2-9720-7edc49b31665" target="_blank">PyTorch Lightning integration</a>

.. |Run on Colab| raw:: html

    <a href="https://colab.research.google.com//github/neptune-ai/neptune-examples/blob/master/integrations/pytorch-lightning/Neptune-PyTorch-Ligthning-basic.ipynb" target="_blank">
        <img width="200" height="200"src="https://colab.research.google.com/assets/colab-badge.svg"></img>
    </a>

.. |script| raw:: html

    <a href="https://github.com/neptune-ai/neptune-examples/blob/master/integrations/pytorch-lightning/docs/Neptune-PyTorch-Ligthning-basic.py" target="_blank">GitHub</a>

.. |forum| raw:: html

    <a href="https://community.neptune.ai/" target="_blank">forum</a>

.. |PyTorch| raw:: html

    <a href="https://docs.neptune.ai/integrations/pytorch.html" target="_blank">PyTorch</a>

.. |PyTorch Ignite| raw:: html

    <a href="https://docs.neptune.ai/integrations/pytorch_ignite.html" target="_blank">PyTorch Ignite</a>

.. |Catalyst| raw:: html

    <a href="https://docs.neptune.ai/integrations/catalyst.html" target="_blank">Catalyst</a>

.. |skorch| raw:: html

    <a href="https://docs.neptune.ai/integrations/skorch.html" target="_blank">skorch</a>

.. |optuna| raw:: html

    <a href="https://docs.neptune.ai/integrations/optuna.html" target="_blank">optuna</a>

.. |plotly| raw:: html

    <a href="https://docs.neptune.ai/integrations/plotly.html" target="_blank">plotly</a>

.. |metrics| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/e/PYTOR-137883/charts" target="_blank">metrics</a>

.. |params| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/e/PYTOR-137883/parameters" target="_blank">logged parameters</a>

.. |hardware| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/e/PYTOR-137883/monitoring" target="_blank">hardware usage statistics</a>

.. |metadata| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/e/PYTOR-137883/details" target="_blank">metadata information</a>

.. |go-here| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/e/PYTOR-137883/charts" target="_blank">go here</a>

.. |exp-link| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/e/PYTOR-137883/charts" target="_blank">here</a>

.. |lightning-install| raw:: html

    <a href="https://pytorch-lightning.readthedocs.io/en/latest/new-project.html#step-0-install-pytorch-lightning" target="_blank">PyTorch Lightning installation guide</a>

.. |lightning-guide| raw:: html

    <a href="https://pytorch-lightning.readthedocs.io/en/latest/new-project.html" target="_blank">Lightning in 2 steps</a>

.. |pytorch-install| raw:: html

    <a href="https://pytorch.org/get-started/locally/" target="_blank">PyTorch installation guide</a>

.. |script-advanced| raw:: html

    <a href="https://github.com/neptune-ai/neptune-examples/blob/master/integrations/pytorch-lightning/docs/Neptune-PyTorch-Ligthning-advanced.py" target="_blank">GitHub</a>

.. |Run on Colab Advanced| raw:: html

    <a href="https://colab.research.google.com//github/neptune-ai/neptune-examples/blob/master/integrations/pytorch-lightning/Neptune-PyTorch-Ligthning-advanced.ipynb" target="_blank">
        <img width="200" height="200"src="https://colab.research.google.com/assets/colab-badge.svg"></img>
    </a>

.. |scikit-learn| raw:: html

    <a href="https://scikit-learn.org/stable/install.html" target="_blank">scikit-learn installation guide</a>

.. |scikit-plot| raw:: html

    <a href="https://github.com/reiinakano/scikit-plot" target="_blank">scikit-plot github project</a>

.. |adv-charts| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/e/PYTOR-137930/charts" target="_blank">charts</a>

.. |adv-parameters| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/e/PYTOR-137930/parameters" target="_blank">parameters</a>

.. |adv-hardware| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/e/PYTOR-137930/monitoring" target="_blank">hardware</a>

.. |adv-details| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/e/PYTOR-137930/details" target="_blank">details</a>

.. |adv-misclassified-images| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/e/PYTOR-137930/logs" target="_blank">misclassified images</a>

.. |adv-confusion-matrix| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/e/PYTOR-137930/logs" target="_blank">confusion matrix</a>

.. |adv-model-checkpoints| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/e/PYTOR-137930/artifacts?path=checkpoints%2F" target="_blank">model checkpoints</a>

.. |adv-model-summary| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/e/PYTOR-137930/logs" target="_blank">model summary</a>

.. |adv-go-here| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/e/PYTOR-137930/charts" target="_blank">charts</a>
