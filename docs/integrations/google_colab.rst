.. _integration-google-colab:

Neptune-Google Colab Integration
================================

You can run experiments on Google Colab and track them with Neptune.

Follow these steps:

1. Install Neptune client:

    Go to your first cell in Google Colab and install `neptune-client`:

    .. code-block:: Bash

        pip install neptune-client

2. Set Neptune API token:

    Go to the Neptune web app and `get your API token <../python-api/how-to/organize.html#find-my-neptune-api-token>`_. Set it to the environment variable `NEPTUNE_API_TOKEN`:

    .. code-block:: Bash

        env NEPTUNE_API_TOKEN='your_private_neptune_api_token=='

3. Delete this cell.

    .. warning::

        It is very important that you delete this cell, so as not to share your private token with anyone.

4. Now, run your training script with Neptune:

    .. code-block:: Python

        import neptune
        neptune.init('USER_NAME/PROJECT_NAME')

        with neptune.create_experiment():
            neptune.send_metric('auc', 0.92)