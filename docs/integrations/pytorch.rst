.. _integrations-pytorch:

Neptune-PyTorch Integration
===========================

Apart from pure PyTorch integration Neptune integrates with many libraries from the PyTorch Ecosystem.

- :ref:`PyTorch Lightning <integrations-pytorch-lightning>`
- :ref:`Fastai and Fastai2 <integrations-fastai>`
- :ref:`PyTorch Ignite <integrations-pytorch-ignite>`
- :ref:`Catalyst <integrations-catalyst>`
- :ref:`Skorch <integrations-skorch>`

|Colab and script buttons|

What will you get with this integration?
----------------------------------------

|pytorch-tour|

|Pytorch| is an open source machine learning framework commonly used for building deep neural network models.
Neptune helps with keeping track of model training metadata.

With Neptune + PyTorch integration you can:

- log hyperparameters
- see learning curves for losses and metrics during training
- see hardware consumption during training
- log stdout and stderr
- log training code and .git information
- log model weights
- log torch tensors as images

.. note::
    You can log many other experiment metadata like interactive charts, video, audio and more.
    See a :ref:`full list <what-you-can-log>`.

.. note::

    This integration is tested with ``torch==1.7.0``, ``neptune-client==0.4.126``.

Where to start?
---------------
To get started with this integration, follow the :ref:`quickstart <pytorch-quickstart>` below.
You can also skip the basics and take a look at how to log model weights and prediction images in the :ref:`advanced options <pytorch-advanced-options>` section.

If you want to try things out and focus only on the code you can either:

|Colab and script buttons|

.. _pytorch-quickstart:

Quickstart
----------
This quickstart will show you how to:

* Install the necessary neptune packages
* Connect Neptune to your Optuna hyperparameter tuning code and create the first experiment
* Log metrics, figures, and artifacts from your first Optuna sweep to Neptune, and
* Explore them in the Neptune UI.

.. _pytorch-before-you-start-basic:

Before you start
^^^^^^^^^^^^^^^^
You have ``Python 3.x`` and following libraries installed:

* ``neptune-client``. See :ref:`neptune-client installation guide <installation-neptune-client>`.
* ``torch``. See |pytorch-install|.

You also need minimal familiarity with torch. Have a look at the |torch-guide| to get started.

.. code-block:: bash

   pip install --quiet torch neptune-client

Step 1: Initialize Neptune
^^^^^^^^^^^^^^^^^^^^^^^^^^
Run the code below:

.. code-block:: python3

    import neptune

    neptune.init(api_token='ANONYMOUS', project_qualified_name='shared/pytorch-integration')

.. tip::

    You can also use your personal API token. Read more about how to :ref:`securely set the Neptune API token <how-to-setup-api-token>`.

Step 2: Create an Experiment and log parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Run the code below to create a Neptune experiment:

.. code-block:: python3

    neptune.create_experiment('pytorch-training')

This also creates a link to the experiment. Open the link in a new tab.
The charts will currently be empty, but keep the window open. You will be able to see live metrics once logging starts.

Step 3: Add metric logging into your training loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Log your loss after every batch by adding ``neptune.log_metric`` inside of the loop.

.. code-block:: python3

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        neptune.log_metric('batch_loss', loss)

You can log epoch metric and losses by calling ``neptune.log_metric`` at the epoch level.

Step 4: Run your training script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Run your script as you normally do:

.. code-block:: bash

    python train.py

Step 5: Monitor your PyTorch training in Neptune
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Now you can switch to the Neptune tab which you had opened previously to watch the training live!

Check out this |example experiment|.

|pytorch-basic-logs|

.. _pytorch-advanced-options:

Advanced Options
----------------

Log hardware consumption
^^^^^^^^^^^^^^^^^^^^^^^^
Neptune can automatically log your CPU and GPU consumption during training as well as stderr and stdout from your console.
To do that you just need to install psutil.

.. code-block:: bash

    pip install psutil

TODO screenshot

|example hardware|

Log hyperparameters
^^^^^^^^^^^^^^^^^^^
You can log training and model hyperparameters.
To do that just pass the parameter dictionary to ``neptune.create_experiment`` method:

.. code-block:: python3

    PARAMS = {'lr':}

    neptune.create_experiment('pytorch-training', params=PARAMS)

TODO screenshot

|example hyperparameters|

Log model weights
^^^^^^^^^^^^^^^^^
You can log model weights to Neptune both during and after training.

To do that just use a ``neptune.log_artifact`` method on the saved model file.

.. code-block:: python3

    torch.save(model.state_dict(), 'model_dict.ckpt')
    neptune.log_artifact('model_dict.ckpt')


TODO screenshot

|example model weights|

Log image predictions
^^^^^^^^^^^^^^^^^^^^^
You can log tensors as images to Neptune.

.. code-block:: python3

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)

        if i % 2000 == 1999:
            for output in outputs:
                neptune.log_image('image predictions`, output)

TODO screenshot

|example predictions|

How to ask for help?
--------------------
Please visit the :ref:`Getting help <getting-help>` page. Everything regarding support is there.

Other integrations you may like
-------------------------------
Here are other integrations with libraries from the PyTorch ecosystem:

- :ref:`PyTorch Lightning<integrations-pytorch-lightning>`
- :ref:`Fastai and Fastai2 <integrations-fastai>`
- :ref:`PyTorch Ignite <integrations-pytorch-ignite>`
- :ref:`Catalyst <integrations-catalyst>`
- :ref:`Skorch <integrations-skorch>`

You may also like these two integrations:

- :ref:`Optuna <integrations-optuna>`
- :ref:`Plotly <integrations-plotly>`

.. External links

.. |pytorch-integration| raw:: html

    <a href="https://ui.neptune.ai/shared/pytorch-integration/experiments" target="_blank">pytorch-integration</a>
