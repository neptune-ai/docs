.. _execution-python:

Python client
=============

.. note::

    Python client is the main way of logging things to Neptune and there are way more materials on how to use it.
    Go :ref:`here to read them <guides-logging-and-managing-experiment-results>`.

To log things to Neptune you just need to install neptune-client and add a snippet to your code.
See how to do that in 3 steps.

Step 1 Install Neptune client
-----------------------------

    .. code:: bash

        pip install neptune-client

Step 2 Add logging snippet to your scripts
------------------------------------------

    .. code:: python

        import neptune

        neptune.init(api_token='', # use your api token
                     project_qualified_name='') # use your project name

        neptune.create_experiment('my-experiment')

        neptune.log_metric('accuracy', 0.92)

Step 3 Run your experiment normally
------------------------------------

    .. code:: bash

        python my_script.py
