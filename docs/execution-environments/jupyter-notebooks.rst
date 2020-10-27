.. _execution-jupyter-notebooks:

Jupyter Lab and Jupyter Notebook
================================

You can run your experiments Jupyter notebooks and track them in Neptune.
You just need to:

- In one of the first cells install Neptune client

    .. code:: bash

        ! pip install neptune-client

- Create Neptune experiment

    .. code:: python

        import neptune

        neptune.init(api_token='', # use your api token
                     project_qualified_name='') # use your project name
        neptune.create_experiment('my-experiment')

To make sure that your API token is secure it is recommended to :ref:`pass it as an environment variable <how-to-setup-api-token>`.

- Log metrics or other object to Neptune (learn :ref:`what else you can log here <guides-logging-data-to-neptune>`).

    .. code:: python

        # training logic
        neptune.log_metric('accuracy', 0.92)

- Stop experiment

    .. code:: python

        neptune.stop()

.. note::

    Neptune supports keeping track of Jupyter Notebook checkpoints if you use neptune-notebooks extension.
    If you do that, your notebook checkpoints will get an auto-snapshot whenever you create an experiment.
    Go :ref:`here to read more about that <guides-keep-track-jupyter-notebooks>`.
