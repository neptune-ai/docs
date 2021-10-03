.. _integrations-pytorch-lightning:

Neptune-PyTorch Lightning Integration
=====================================

.. warning::
    This is ``LegacyLogger`` implementation that is under maintenance support only.
    No new updates will be made to this logger.

    Please visit
    `integration docs <https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch-lightning>`_
    to learn about the latest, fully supported version.

What will you get with this integration?
----------------------------------------
PyTorch Lightning is a lightweight PyTorch wrapper for high-performance AI research.
With Neptune integration you can:

* monitor model training live,
* log training, validation, and testing metrics, and visualize them in the Neptune UI,
* log hyperparameters,
* monitor hardware usage,
* log any additional metrics,
* log performance charts and images,
* save model checkpoints.

Installation
------------
Before you start, make sure that:

* You have Python 3.6+ in your system,
* You are already `registered user <https://neptune.ai/register>`_, so that you can log metadata to your `private projects <https://docs.neptune.ai/administration/workspace-project-and-user-management/projects>`_.
* You have your ``NEPTUNE_API_TOKEN`` set to the environment variable: `check this docs page <https://docs.neptune.ai/getting-started/installation#authentication-neptune-api-token>`_.

**Install neptune-contrib**

Depending on your operating system open a terminal or CMD and run this command.

.. code-block:: bash

    pip install 'neptune-contrib[monitoring]'

This logger works with ``1.0.7<=pytorch-lightning<=1.4``.

Quickstart
----------

Create NeptuneLogger
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python3

    from neptunecontrib.monitoring.pytorch_lightning import NeptuneLogger

    neptune_logger = NeptuneLogger(
                api_key='<YOUR_API_TOKEN>',
                project='<YOUR_WORKSPACE/YOUR_PROJECT>',
                name='lightning-run',  # Optional
            )

Pass your Neptune _Project_ name and API token to ``NeptuneLogger``.

.. note::

    See how to:

    * get `your full project name <https://docs.neptune.ai/getting-started/installation#authentication-neptune-api-token>`_
    * find and set your `Neptune API token <https://docs.neptune.ai/getting-started/installation#setting-the-project-name>`_

Pass neptune_logger to Trainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Pass ``neptune_logger`` instance to lightning ``Trainer`` to log model training metadata to Neptune:

.. code-block:: python3

    from pytorch_lightning import Trainer

    trainer = Trainer(max_epochs=10, logger=neptune_logger)

Run model training
^^^^^^^^^^^^^^^^^^
Pass your lightning ``Module`` and training ``Loader`` to ``Trainer`` and run ``.fit()``:

.. code-block:: python3

    trainer.fit(model, train_loader)

Explore Results
^^^^^^^^^^^^^^^
You just learned how to start logging PyTorch Lightning model training runs to Neptune, by using Neptune logger.

.. warning::
    This is ``LegacyLogger`` implementation that is under maintenance support only.
    No new updates will be made to this logger.

    Please visit
    `integration docs <https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch-lightning>`_
    to learn about the latest, fully supported version.

Use logger inside your lightning Module class
---------------------------------------------
You can use log Images, model checkpoints, and other ML metadata from inside your training and evaluation steps.

To do that you need to:

* access the ``Experiment`` object at ``self.logger.experiment``
* use one of the `logging methods <https://docs-legacy.neptune.ai/api-reference/neptune/experiments/index.html#neptune-experiments>`_ that ``Experiment`` object exposes.

.. code-block:: python3

    from neptune.new.types import File

    class LitModel(LightningModule):
        def training_step(self, batch, batch_idx):
            # log metrics
            acc = ...
            self.logger.experiment['train/acc'].log(acc)
            # log images
            img = ...
            self.logger.experiment['train/misclassified_images'].log(File.as_image(img))

        def any_lightning_module_function_or_hook(self):
            # log model checkpoint
            ...
            self.logger.experiment['checkpoints/epoch37'].upload('epoch=37.ckpt')
            # generic recipe
            metadata = ...
            self.logger.experiment['your/metadata/structure'].log(metadata)


.. note::
    You can log other model-building metadata like metrics, images, video, audio, interactive visualizations, and more. See `What can you log and display? <https://docs.neptune.ai/you-should-know/what-can-you-log-and-display>`_.

Log after training is finished
------------------------------
If you want to log objects after the training is finished, use ``close_after_fit=False``. You will then need to explicitly stop the logger after your logging is complete using ``neptune_logger.experiment.stop()``.

.. code-block:: python3

    from neptunecontrib.monitoring.pytorch_lightning import NeptuneLogger

    neptune_logger = NeptuneLogger(
        api_key='<YOUR_API_TOKEN>',
        project='<YOUR_WORKSPACE/YOUR_PROJECT>',
        close_after_fit=False,
    )
    trainer = Trainer(logger=neptune_logger)
    trainer.fit(model)

    # Log confusion matrix after training
    from neptune.new.types import File
    from scikitplot.metrics import plot_confusion_matrix
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 12))
    plot_confusion_matrix(y_true, y_pred, ax=ax)
    neptune_logger.experiment['test/confusion_matrix'].upload(File.as_image(fig))

    # Stop logging
    neptune_logger.experiment.stop()

Pass additional parameters to NeptuneLogger
-------------------------------------------
You can also pass ``kwargs`` to specify the ``Experiment`` in greater detail, like `tags` and `description`:

.. code-block:: python3

    neptune_logger = NeptuneLogger(
        api_key='<YOUR_API_TOKEN>',
        project='<YOUR_WORKSPACE/YOUR_PROJECT>',
        name='lightning-run',
        description='mlp quick run with pytorch-lightning',
        tags=['mlp', 'quick-run'],
    )
    trainer = Trainer(max_epochs=3, logger=neptune_logger)

.. warning::
    This is ``LegacyLogger`` implementation that is under maintenance support only.
    No new updates will be made to this logger.

    Please visit
    `integration docs <https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch-lightning>`_
    to learn about the latest, fully supported version.

External resources
------------------

* LegacyLogger `reference docs <https://neptune-contrib.readthedocs.io/>`_
