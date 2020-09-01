How to organize ML experimentation: step by step guide
======================================================

|run on colab button|

Introduction
------------

This guide will show you how to:

- Keep track of code, data, environment and parameters
- Log results like evaluation metrics and model files
- Find experiments in the experiment dashboard with tags, parameter and metric filters
- Organize experiments in a dashboard view and save it for later

Before you start
----------------

Make sure you meet the following prerequisites before starting:

- Have Python 3.x installed
- Have Tensorflow 2.x with Keras installed
- |Have Neptune installed|
- |Create a project|
- |Configure Neptune API token on your system|

.. note::

    You can run this how-to on Google Colab with zero setup.

    Just click on the ``Open in Colab`` button on the top of the page.

Step 1: Create a basic training script
--------------------------------------

Step 3: Connect Neptune to your script
--------------------------------------

Step 4. Create an experiment and add parameter, code and environment tracking
-----------------------------------------------------------------------------------

1. Add parameters tracking

2. Add code and environment tracking

Step 5. Add tags to organize things
-----------------------------------

Step 6. Add logging of evaluation metrics
-----------------------------------------

Step 7. Add logging of model files
----------------------------------

Step 8. Run a few experiments with different parameters
-------------------------------------------------------

Step 9. Filter experiments by tag
---------------------------------

Step 10. Choose parameter and metric columns
--------------------------------------------

Step 11. Save the view of experiment space
------------------------------------------

What's next
-----------

Now that you know how to create experiments and log metrics you can learn:

- See |how to log other objects and monitor training in Neptune|
- See |how to connect Neptune to your codebase|
- |Check our integrations| with other frameworks

Full Neptune monitoring script
------------------------------

|run on colab button|

.. code:: python

    TODO

.. |Create a project| raw:: html

    <a href="/teamwork-and-user-management/how-to/create-project.html" target="_blank">Create a project in Neptune</a>

.. |Configure Neptune API token on your system| raw:: html

    <a href="/security-privacy/api-tokens/how-to-api-token.html" target="_blank">Configure Neptune API token on your system</a>

.. |Have Neptune installed| raw:: html

    <a href="/getting-started/installation/index.html">Have Neptune installed</a>

.. |run on colab button| raw:: html

    <a href="https://colab.research.google.com//github/neptune-ai/neptune-colab-examples/blob/master/Organize-ML-experiments.ipynb" target="_blank">
        <img width="200" height="200"src="https://colab.research.google.com/assets/colab-badge.svg"></img>
    </a>

.. |how to log other objects and monitor training in Neptune| raw:: html

    <a href="https://neptune.ai/blog/monitoring-machine-learning-experiments-guide" target="_blank">how to log other objects and monitor training in Neptune</a>

.. |how to connect Neptune to your codebase| raw:: html

    <a href="/getting-started/adding-neptune/step-by-step-connect-neptune.html" target="_blank">how to connect Neptune to your codebase</a>

.. |Check our integrations| raw:: html

    <a href="/integrations/index.html" target="_blank">Check our integrations</a>