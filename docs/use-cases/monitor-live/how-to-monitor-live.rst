Monitor ML runs live wherever you are
=====================================

|run on colab button|

Introduction
------------

This guide will show you how to:

* Monitor training and evaluation metrics and losses live
* Monitor hardware resources during training
* Create a Neptune callback

By the end of it, you will run your first experiment and see it in Neptune!

Before you start
----------------

Make sure you meet the following prerequisites before starting:

- Have Python 3.x installed
- Have Tensorflow with Keras installed
- |Create Project in Neptune|
- |Configure Neptune API token on your system|

.. note::

    You can run this how-to on Google Colab with zero setup. Just click on the button above.

Step 1 - Install neptune-client
-------------------------------

Go to the command line of your operating system and run the installation command:

.. code:: bash

    pip install neptune-client

.. note::

    If you are using R or any other language |go read this|.

Step 1: Install psutil
----------------------

Step 2: Install psutil
----------------------

What's next
-----------

Now that you know how to create experiments and log metrics you can learn:

- |create a new project|
- See |how to log other objects and monitor training in Neptune|
- See |how to connect Neptune to your codebase|
- See |how to connect Neptune to your codebase|

.. External links

.. |Create a new project| raw:: html

    <a href="/teamwork-and-user-management/how-to/create-project.html" target="_blank">Create a new project</a>

.. |how to log other objects and monitor training in Neptune| raw:: html

    <a href="https://neptune.ai/blog/monitoring-machine-learning-experiments-guide" target="_blank">how to log other objects and monitor training in Neptune</a>

.. |how to connect Neptune to your codebase| raw:: html

    <a href="/getting-started/adding-neptune/step-by-step-connect-neptune.html" target="_blank">how to connect Neptune to your codebase</a>


.. |Check our integrations| raw:: html

    <a href="/integrations/index.html" target="_blank">Check our integrations</a>

.. |how to install it| raw:: html

    <a href="/getting-started/installation/install_client.html" target="_blank">how to install it</a>

.. |go read this| raw:: html

    <a href="/integrations/languages.html" target="_blank">go read this</a>

.. |run on colab button| raw:: html

    <a href="https://colab.research.google.com//github/neptune-ai/neptune-colab-examples/blob/master/Use-Neptune-API-to-log-your-first-experiment.ipynb" target="_blank">
        <img width="200" height="200"src="https://colab.research.google.com/assets/colab-badge.svg"></img>
    </a>
