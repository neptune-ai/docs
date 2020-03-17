Log PyTorchLightning metrics to neptune
=======================================
.. image:: ../_static/images/others/pytorchlightning_neptuneml.png
   :target: ../_static/images/others/pytorchlightning_neptuneml.png
   :alt: PyTorchLightning neptune.ai integration

Prerequisites
-------------
Integration with |PyTorchLightning| framework is introduced as a part of logging module so just need to have |neptune-client| installed.

.. code-block:: bash

    pip install neptune-client


Create the **LightningModule**
------------------------------
.. code-block:: python3

    import os
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(0)

# create data
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(1000, 20, n_informative=10, random_state=0)
X = X.astype(np.float32)


# create pytorch module
class ClassifierModule(nn.Module):
    def __init__(
            self,
            num_units=10,
            nonlin=F.relu,
            dropout=0.5,
    ):
        super(ClassifierModule, self).__init__()
        self.num_units = num_units
        self.nonlin = nonlin
        self.dropout = dropout

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X), dim=-1)
        return X


# create neptune logger and pass it to NeuralNetClassifier
from skorch import NeuralNetClassifier
import neptune
from skorch.callbacks.logging import NeptuneLogger

neptune.init('neptune-ai/skorch-integration')
experiment = neptune.create_experiment(
    name='skorch-basic-example',
    params={'max_epochs': 20,
            'lr': 0.1},
    upload_source_files=['skorch_example.py'])
neptune_logger = NeptuneLogger(experiment, close_after_train=False)

net = NeuralNetClassifier(
    ClassifierModule,
    max_epochs=20,
    lr=0.1,
    callbacks=[neptune_logger]
)

# run training
net.fit(X, y)

# log score after training
from sklearn.metrics import roc_auc_score

y_pred = net.predict_proba(X)
auc = roc_auc_score(y, y_pred[:, 1])

neptune_logger.experiment.log_metric('roc_auc_score', auc)

# log confusion matrix
from scikitplot.metrics import plot_roc
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(16, 12))
plot_roc(y, y_pred, ax=ax)
neptune_logger.experiment.log_image('roc_curve', fig)

# log model after training
net.save_params(f_params='basic_model.pkl')
neptune_logger.experiment.log_artifact('basic_model.pkl')

# close experiment
neptune_logger.experiment.stop()

"""
Added Neptune logger that:

- logs metrics `on_batch_end`
- logs metrics  `on_epoch_end`
- logs any additional information directly to `neptune_logger.experiment.log_whatever_neptune_allows` if used with `close_after_train=False`
"""
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision import transforms

    import pytorch_lightning as pl


    class CoolSystem(pl.LightningModule):

        def __init__(self):
            super(CoolSystem, self).__init__()
            # not the best model...
            self.l1 = torch.nn.Linear(28 * 28, 10)

        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))

        def training_step(self, batch, batch_idx):
            # REQUIRED
            x, y = batch
            y_hat = self.forward(x)
            loss = F.cross_entropy(y_hat, y)
            tensorboard_logs = {'train_loss': loss}
            return {'loss': loss, 'log': tensorboard_logs}

        def validation_step(self, batch, batch_idx):
            # OPTIONAL
            x, y = batch
            y_hat = self.forward(x)
            return {'val_loss': F.cross_entropy(y_hat, y)}

        def validation_end(self, outputs):
            # OPTIONAL
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            tensorboard_logs = {'val_loss': avg_loss}
            return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

        def configure_optimizers(self):
            # REQUIRED
            # can return multiple optimizers and learning_rate schedulers
            # (LBFGS it is automatically supported, no need for closure function)
            return torch.optim.Adam(self.parameters(), lr=0.02)

        @pl.data_loader
        def train_dataloader(self):
            # REQUIRED
            return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

        @pl.data_loader
        def val_dataloader(self):
            # OPTIONAL
            return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

Create the **NeptuneLogger** with all the information you want to track
------------------------------------------------------------------------
.. code-block:: python3

    from pytorch_lightning.logging.neptune import NeptuneLogger

    neptune_logger = NeptuneLogger(
        api_key="ANONYMOUS",
        project_name="shared/pytorch-lightning-integration",
        experiment_name="default",  # Optional,
        params={"max_epochs": 10,
                "batch_size": 32},  # Optional,
        tags=["pytorch-lightning", "mlp"]  # Optional,
    )

Create the **Trainer** and pass **neptune_logger** to logger
------------------------------------------------------------
.. code-block:: python3

    from pytorch_lightning import Trainer

    model = CoolSystem()
    trainer = Trainer(max_epochs=10, logger=neptune_logger)

    trainer.fit(model)

Log additional information after the **.fit** loop ends
-------------------------------------------------------

You can log additional metrics, images, model binaries or other things after the `.fit` loop is over.
You just need to specify `close_after_fit=False` in `NeptuneLogger` initialization.

.. code-block:: python3

    neptune_logger = NeptuneLogger(
        api_key="ANONYMOUS",
        project_name="shared/pytorch-lightning-integration",
        close_after_fit=False,
        ...
    )

**Log test metrics**

.. code-block:: python3

    trainer.test(model)

**Log additional metrics**

.. code-block:: python3

    from sklearn.metrics import accuracy_score
    ...
    accuracy = accuracy_score(y_true, y_pred)

    neptune_logger.experiment.log_metric('test_accuracy', accuracy)

**Log performance charts**

.. code-block:: python3

    from scikitplot.metrics import plot_confusion_matrix
    import matplotlib.pyplot as plt
    ...
    fig, ax = plt.subplots(figsize=(16, 12))
    plot_confusion_matrix(y_true, y_pred, ax=ax)

    neptune_logger.experiment.log_image('confusion_matrix', fig)

**Save checkpoints folder after training**

.. code-block:: python3

    model_checkpoint = pl.callbacks.ModelCheckpoint(filepath='my/checkpoints')

    trainer = Trainer(logger=neptune_logger,
                      checkpoint_callback=model_checkpoint)
    trainer.fit(model)

    neptune_logger.experiment.log_artifact('my/checkpoints')

**Explicitly close the logger** it is optional but you may want to close it and than do something after.

.. code-block:: python3

    neptune_logger.experiment.stop()

Monitor your PyTorchLightning training in Neptune
--------------------------------------------------
Now you can watch your pytorch-lightning model training in neptune!

.. image:: ../_static/images/pytorch_lightning/pytorch_lightning_monitoring.gif
   :target: ../_static/images/pytorch_lightning/pytorch_lightning_monitoring.gif
   :alt: PyTorchLightning logging in neptune

Full PyTorchLightning monitor script
------------------------------------
Simply copy and paste it to ``pytorch_lightning_example.py`` and run.
You can change your credentials in the **NeptuneLogger** or run some tests as anonymous user:

.. code-block:: python3

    neptune_logger = NeptuneLogger(
        api_key="ANONYMOUS",
        project_name="shared/pytorch-lightning-integration",
        ...
        )

.. code-block:: python3

    import os

    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision import transforms

    import pytorch_lightning as pl


    class CoolSystem(pl.LightningModule):

        def __init__(self):
            super(CoolSystem, self).__init__()
            # not the best model...
            self.l1 = torch.nn.Linear(28 * 28, 10)

        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))

        def training_step(self, batch, batch_idx):
            # REQUIRED
            x, y = batch
            y_hat = self.forward(x)
            loss = F.cross_entropy(y_hat, y)
            tensorboard_logs = {'train_loss': loss}
            return {'loss': loss, 'log': tensorboard_logs}

        def validation_step(self, batch, batch_idx):
            # OPTIONAL
            x, y = batch
            y_hat = self.forward(x)
            return {'val_loss': F.cross_entropy(y_hat, y)}

        def validation_end(self, outputs):
            # OPTIONAL
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            tensorboard_logs = {'val_loss': avg_loss}
            return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

        def configure_optimizers(self):
            # REQUIRED
            # can return multiple optimizers and learning_rate schedulers
            # (LBFGS it is automatically supported, no need for closure function)
            return torch.optim.Adam(self.parameters(), lr=0.02)

        @pl.data_loader
        def train_dataloader(self):
            # REQUIRED
            return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

        @pl.data_loader
        def val_dataloader(self):
            # OPTIONAL
            return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)


    from pytorch_lightning.logging.neptune import NeptuneLogger

    neptune_logger = NeptuneLogger(
        api_key="ANONYMOUS",
        project_name="shared/pytorch-lightning-integration",
        experiment_name="default",  # Optional,
        params={"max_epochs": 4,
                "batch_size": 32},  # Optional,
        tags=["pytorch-lightning", "mlp"]  # Optional,
    )

    from pytorch_lightning import Trainer

    trainer = Trainer(max_epochs=4, logger=neptune_logger)
    trainer.fit(CoolSystem())


.. External links

.. |PyTorchLightning| raw:: html

    <a href="https://github.com/PyTorchLightning/pytorch-lightning" target="_blank">PyTorchLightning</a>

.. |neptune-client| raw:: html

    <a href="https://github.com/neptune-ai/neptune-client" target="_blank">neptune-client</a>
