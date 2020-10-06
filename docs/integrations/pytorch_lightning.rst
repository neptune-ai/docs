PyTorch Lightning Integration
=============================

What will you get?
------------------
[video-placeholder]

PyTorch Lightning is a lightweight PyTorch wrapper for high-performance AI research. With Neptune integration you can:

* preview running experiment
* log training and testing metrics, and visualize them in Neptune UI,
* log experiment parameters,
* monitor hardware usage,
* log any additional metrics of your choice,
* log performance charts,
* save model checkpoints.

.. tip::

    Check this public project with example experiments: |project|.

    You can also go straight to the |exp|.

Quickstart
----------

Introduction
^^^^^^^^^^^^
This quickstart will show you how to use Neptune with your existing PyTorch Lightning code. You have three options:

1. Follow through this quickstart for detailed explanations,
2. Open starter code in Colab Notebook (badge-link below) and run it as a "`neptuner`" user - zero setup, it just works.
3. View starter code as Python script on |script|.

|Run on Colab|

Before you start
^^^^^^^^^^^^^^^^

**Prerequisites**

* Python 3,
* Neptune account |register|,
* Minimal familiarity with the PyTorch Lightning.

**Supported version**

* ``pytorch-lightning>=0.9.0``
* ``neptune-client>=0.4.105``

Installation
^^^^^^^^^^^^
**Install neptune-client**

From PyPI:

.. code-block:: bash

    pip install neptune-client -U

From Conda:

.. code-block:: bash

    conda install neptune-client -c conda-forge

**Install PyTorch Lightning and torchvision**

From PyPI:

.. code-block:: bash

    pip install pytorch-lightning torchvision

From Conda:

.. code-block:: bash

    conda install pytorch-lightning -c conda-forge
    conda install pytorch torchvision -c pytorch

Now, you have all dependencies installed. Let's move to the actual integration.

Step 1 - Necessary Imports
^^^^^^^^^^^^^^^^^^^^^^^^^^
No a big deal here, just imports. Note ``pytorch_lightning`` at the bottom.

.. code-block:: python3

    import os

    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision import transforms

    import pytorch_lightning as pl

Step 2 - Model Hyper-Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You will see them in Neptune parameters tab.

.. code-block:: python3

    PARAMS = {'max_epochs': 3,
              'LR': 0.02,
              'batch_size': 32}

Step 3 - Define Lightning Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This is minimal example of the ``pl.LightningModule``. Notice the Cross Entropy loss in the ``training_step`` method.

Also, note that you pass learning rate from the ``PARAMS`` dictionary.

.. code-block:: python3

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
            return pl.TrainResult(loss)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=PARAMS['LR'])

Step 4 - Prepare Data Loader
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``DataLoader`` (you know it from PyTorch) is necessary to fit your model. Note that you pass ``batch_size`` from the ``PARAMS`` dictionary.

.. code-block:: python3

    train_loader = DataLoader(MNIST(os.getcwd(), download=True, transform=transforms.ToTensor()),
                              batch_size=PARAMS['batch_size'])

Step 5 - Create NeptuneLogger
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``NeptuneLogger`` is an object that integrates Neptune with PyTorch Lightning allowing you to track your experiments in Neptune. It's a part of the lightning library.

In this minimalist example we use public user `"neptuner"`, who has public token: "ANONYMOUS".

.. code-block:: python3

    from pytorch_lightning.loggers.neptune import NeptuneLogger

    neptune_logger = NeptuneLogger(
        api_key="ANONYMOUS",
        project_name="shared/pytorch-lightning-integration",
        params=PARAMS)

.. tip::

    Make sure to use your API token in your projects. Read more about how to |token|.

Step 6 - Fit model to the data - Neptune tracks it automatically
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
At this point you are ready to fit the model to the data. Simply pass ``neptune_logger`` to the ``Trainer`` and run ``fit()`` loop. Neptune will collect metrics and show them in UI.

Notice, that ``max_epochs`` is from the ``PARAMS`` dictionary. All these params are logged to Neptune.

.. code-block:: python3

    trainer = pl.Trainer(max_epochs=PARAMS['max_epochs'],
                         logger=neptune_logger)
    model = LitModel()

    trainer.fit(model, train_loader)

Results
^^^^^^^
You just learned how to start logging PyTorch Lightning experiments to Neptune, by using Neptune logger which is part of the lightning library.

Above training is logged to Neptune in near real-time. You can go and explore the results. In particular check:

#. |metrics|,
#. |params|,
#. |hardware|,
#. |metadata|.

.. image:: ../_static/images/integrations/pytorchlightning_neptuneml.png
   :target: ../_static/images/integrations/pytorchlightning_neptuneml.png
   :alt: PyTorchLightning neptune.ai integration

Advanced options
----------------

Log test metrics
^^^^^^^^^^^^^^^^

Log additional metrics
^^^^^^^^^^^^^^^^^^^^^^

Log performance charts
^^^^^^^^^^^^^^^^^^^^^^

Save model checkpoints
^^^^^^^^^^^^^^^^^^^^^^

Troubleshooting
---------------

How to ask for help?
^^^^^^^^^^^^^^^^^^^^
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

#. |Keras|
#. |LightGBM|


.. External links

.. |register| raw:: html

    <a href="https://neptune.ai/register" target="_blank">register here</a>

.. |project| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/experiments?viewId=8080df27-e2d7-48e7-a04d-5fab2d2c6fd2" target="_blank">PyTorch Lightning integration</a>

.. |Run on Colab| raw:: html

    <a href="https://colab.research.google.com//github/neptune-ai/neptune-examples/blob/master/integrations/pytorch-lightning/Neptune-PyTorch-Ligthning-basic.ipynb" target="_blank">
        <img width="200" height="200"src="https://colab.research.google.com/assets/colab-badge.svg"></img>
    </a>

.. |exp| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/e/PYTOR-68/charts" target="_blank">example experiment</a>

.. |script| raw:: html

    <a href="https://github.com/neptune-ai/neptune-examples/blob/master/integrations/pytorch-lightning/docs/Neptune-PyTorch-Ligthning-basic.py" target="_blank">GitHub</a>

.. |token| raw:: html

    <a href="https://docs.neptune.ai/security-and-privacy/api-tokens/how-to-find-and-set-neptune-api-token.html#how-to-find-and-set-neptune-api-token" target="_blank">securely set Neptune API token</a>

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

.. |Keras| raw:: html

    <a href="https://docs.neptune.ai/integrations/keras.html" target="_blank">Keras</a>

.. |LightGBM| raw:: html

    <a href="https://docs.neptune.ai/integrations/lightgbm.html" target="_blank">LightGBM</a>

.. |metrics| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/e/PYTOR-68/charts" target="_blank">metrics</a>

.. |params| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/e/PYTOR-68/parameters" target="_blank">logged parameters</a>

.. |hardware| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/e/PYTOR-68/monitoring" target="_blank">hardware usage statistics</a>

.. |metadata| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/e/PYTOR-68/details" target="_blank">metadata information</a>
