.. _execution-deepnote:

Deepnote
========

You can easily run experiments in |Deepnote notebooks| and log them to Neptune.
To do that you need to:

Step 1: Install neptune-client
------------------------------

    .. code:: bash

        ! pip install neptune-client==0.4.123

Step 2: Set up an environment variable for the API token
--------------------------------------------------------

Create a new environment variable integration in the left tab where you set the NEPTUNE_API_TOKEN.
Alternatively, you can initialize neptune with the API token directly with the snippet:

    .. code:: python

        # Alternative version to initialising Neptune
        neptune.init(project_qualified_name='<name_here>',
                     api_token='<token_here>')

See how to :ref:`get your Neptune API token here <how-to-setup-api-token>`.

Step 3: Replace the project name and log metrics into a Neptune dashboard
-------------------------------------------------------------------------

    .. code:: python

        import neptune

        # The init() function called this way assumes that
        # NEPTUNE_API_TOKEN environment variable is defined by the integration.

        neptune.init('<NEPTUNE_PROJECT_NAME>')
        neptune.create_experiment(name='minimal_example')

        # log some metrics

        for i in range(100):
            neptune.log_metric('loss', 0.95**i)

        neptune.log_metric('AUC', 0.96)

.. note::

    Check out a this |example Deepnote notebook| with Neptune logging.

.. external links

.. |example Deepnote notebook| raw:: html

    <a href="https://deepnote.com/publish/13d805d1-8b8e-4c2c-a02b-96c182db640d" target="_blank">example Deepnote notebook</a>

.. |Deepnote notebooks| raw:: html

    <a href="https://deepnote.com/" target="_blank">Deepnote notebooks</a>
