.. _integrations-google-colab:

Neptune-Google Colab Integration
================================

|Video intro|

|Google Colab| is a temporary runtime environment. This means you lose all your data (unless saved externally) once you restart your kernel. 

This is where you can leverage Neptune. By running your experiments on Google Colab and tracking them with Neptune, you can log:

* parameters,
* metrics and losses,
* hardware consumption,
* model checkpoints and other artifacts,

to Neptune and refer to them even after the Colab kernel has died.

Introduction
------------

This guide will show you how to:

* Install ``neptune-client``
* Connect Neptune to your notebook and create the first experiment
* Log simple metrics to Neptune and explore them in the Neptune UI

Before you start
----------------

Make sure that you have an account with both |Google| and |Neptune|.

Quickstart
----------

|Run on Colab|

Step 1: Install Neptune client
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Go to your first cell in Google Colab and install ``neptune-client``:

.. code-block:: Bash

   pip install neptune-client

Step 2: Set Neptune API token
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|Video get pass|
  
#. Go to the Neptune web app and get your API token
   
#. Run the code below:
    
   .. code-block:: Python

      from getpass import getpass
      api_token = getpass('Enter your private Neptune API token: ')

#. Enter the token in the input box. This will save your token to ``api_token``.


Step 3: Initialize your Neptune Project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Run the code below:

.. code-block:: Python

    import neptune

    neptune.init(project_qualified_name = 'your_user_name/your_project_name'
                 , api_token = api_token)

Step 4: Run your training script with Neptune
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Run the code below:

.. code-block:: Python

    from numpy import random
    from time import sleep

    neptune.create_experiment()
        
    neptune.log_metric('single_metric', 0.62)
        
    for i in range(100):
        sleep(0.2) # to see logging live
        neptune.log_metric('random_training_metric', i*random.random())
        neptune.log_metric('other_random_training_metric', 0.5*i*random.random())
	
    neptune.stop()

Step 5: Check metrics on the Neptune UI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Follow the link shown to view your experiment progress and metrics in the Neptune UI.

|Explore experiment|

|Video explore experiments|

What's next
-----------

Now that you know how to integrate Neptune with Google Colab, you can check:

* :ref:`What can you log to experiments? <log-to-experiments>`
* :ref:`Downloading experiment data from Neptune <guides-download_data>`
* Other :ref:`Neptune integrations <integrations-index>`

.. External links

.. |Video intro| raw:: html

    <div style="position: relative; padding-bottom: 53.65126676602087%; height: 0;"><iframe src="https://www.loom.com/embed/a533ef47757a498a9fa944541b56d780" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>

.. |Video get pass| raw:: html

    <div style="position: relative; padding-bottom: 53.65126676602087%; height: 0;"><iframe src="https://www.loom.com/embed/a3681197abbe4f118896fc087c73932f" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>

.. |Video explore experiments| raw:: html

    <div style="position: relative; padding-bottom: 53.65126676602087%; height: 0;"><iframe src="https://www.loom.com/embed/5fedeee768c545c48694a804130d8bde" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>

.. |Google Colab| raw:: html

    <a href="https://colab.research.google.com/" target="_blank">Google Colab</a>

.. |Google| raw:: html
	
    <a href="https://support.google.com/accounts/answer/27441?hl=en" target="_blank">Google</a>

.. |Neptune| raw:: html

    <a href="https://neptune.ai/register" target="_blank">Neptune</a>
	
.. |Run on Colab| raw:: html

    <a href="https://colab.research.google.com/github/neptune-ai/neptune-colab-examples/blob/master/Neptune-API-Tour.ipynb" target="_blank">
        <img width="200" height="200"src="https://colab.research.google.com/assets/colab-badge.svg"></img>
    </a>

.. |Explore experiment| raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/BU20fhL6jBE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
