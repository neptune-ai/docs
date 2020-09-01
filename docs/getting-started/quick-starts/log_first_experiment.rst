Use Neptune API to Iog your first experiment
============================================

|run on colab button|

Introduction
------------

This guide will show you how to:

* Install neptune-client
* Connect Neptune to your script and create the first experiment
* Log metrics to Neptune and explore them in the UI

By the end of it, you will run your first experiment and see it in Neptune!

Before you start
----------------

Make sure you meet the following prerequisites before starting:

* Have Python 3.x installed

.. note::

    You can run this quick-start on Google Colab with zero setup. Just click on the button above.

Step 1 - Install neptune-client
-------------------------------

Go to the command line of your operating system and run the installation command:

.. code:: bash

    pip install neptune-client

.. note::

    If you are using R or any other language |go read this|.


Step 2 - Create a `quickstart.py`
---------------------------------

Create a python script called `quickstart.py` and copy the code below to it:

`quickstart.py`

.. code:: python

    # Connect your script to Neptune
    import neptune

    neptune.init(project_qualified_name='shared/onboarding',
                  api_token='ANONYMOUS',
                 )

    # Create experiment
    neptune.create_experiment()

    # Log metrics to experiment
    from time import sleep

    neptune.log_metric('single_metric', 0.62)

    for i in range(100):
        sleep(0.2) # to see logging live
        neptune.log_metric('random_training_metric', i * 0.6)
        neptune.log_metric('other_random_training_metric', i * 0.4)

|run on colab button|

.. note::

    Instead of logging data to the public project 'shared/onboarding' as an anonymous user 'neptuner' you can log it to your own project.

    1. Get your Neptune API token

       .. image:: ../../_static/images/others/get_token.gif
          :target: ../../_static/images/others/get_token.gif
          :alt: Get API token

    2. Pass the token to ``api_token`` argument of ``neptune.init()`` method: ``api_token=YOUR_API_TOKEN``
    3. Pass your username to the ``project_qualified_name`` argument of the ``neptune.init()`` method: ``project_qualified_name='YOUR_USERNAME/sandbox``.
       Keep ``/sandbox`` at the end, the ``sandbox`` project that was automatically created for you.

    For example:

    .. code:: python

        neptune.init(project_qualified_name='funky_steve/sandbox',
                     api_token='eyJhcGlfYW908fsdf23f940jiri0bn3085gh03riv03irn',
                    )


Step 3 - Run your script and explore results
--------------------------------------------

Now that you have your script ready you can run it and see results in Neptune.

Run your script from the terminal or Jupyter notebook

.. code:: bash

    python quickstart.py

Click on the link in the terminal or notebook or go directly to the Neptune app. 

See  metrics you logged in `Logs`, `Charts`, and hardware consumption in the `Monitoring` sections of the Neptune UI:

|Explore experiment|

Conclusion
----------

You’ve learned how to:

* Install neptune-client
* Connect Neptune to your python script and create an experiment
* Log metrics to Neptune
* Explore your metrics in ``Logs`` and ``Charts`` sections
* See hardware consumption during the experiment run

What's next
-----------

Now that you know how to create experiments and log metrics you can learn:

- |create a new project|
- See |how to log other objects and monitor training in Neptune|
- See |how to connect Neptune to your codebase|

.. External links

.. |how to log other objects and monitor training in Neptune| raw:: html

    <a href="https://neptune.ai/blog/monitoring-machine-learning-experiments-guide" target="_blank">how to log other objects and monitor training in Neptune</a>

.. |how to connect Neptune to your codebase| raw:: html

    <a href="/getting-started/adding-neptune/step-by-step-connect-neptune.html" target="_blank">how to connect Neptune to your codebase</a>

.. |run on colab button| raw:: html

    <a href="https://colab.research.google.com//github/neptune-ai/neptune-colab-examples/blob/master/Use-Neptune-API-to-log-your-first-experiment.ipynb" target="_blank">
        <img width="200" height="200"src="https://colab.research.google.com/assets/colab-badge.svg"></img>
    </a>

.. |Create a new project| raw:: html

    <a href="/teamwork-and-user-management/how-to/create-project.html" target="_blank">Create a new project</a>

.. |Get your Neptune API token| raw:: html

    <a href="/security/how-to/api-token.html" target="_blank">Get your Neptune API token</a>

.. |go read this| raw:: html

    <a href="/integrations/languages.html" target="_blank">go read this</a>

.. |Explore experiment| raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/BU20fhL6jBE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>