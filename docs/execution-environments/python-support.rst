.. _execution-python:

Python client
=============

You can run your experiments in any environment and log them to Neptune.
You just need to:

- install Neptune client

    .. code:: bash

        pip install neptune-client

- add logging snippet to your scripts

    .. code:: python

        import neptune

        neptune.init(api_token='', # use your api token
                     project_qualified_name='') # use your project name
        neptune.create_experiment('my-experiment')
        # training logic
        neptune.log_metric('accuracy', 0.92)

.. note::

    Since Neptune primarily supports Python there are way more materials on how to use it.
    Go :ref:`here to read them <guides-logging-and-managing-experiment-results>`.
