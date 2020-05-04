Google Colab
============

You can run experiments on Google Colab and track them with Neptune.


**Install Neptune client**

Go to your first cell in Google Colab and install `neptune-client`:

.. code-block:: Bash

    ! pip install neptune-client

**Set Neptune API token**

Go to Neptune web app and get your API token. Set it to the environment variable `NEPTUNE_API_TOKEN`:

.. code-block:: Bash

    % env NEPTUNE_API_TOKEN='your_private_neptune_api_token=='

Delete this cell.

.. warning::

    It is very important that you delete this cell not to share your private token with anyone.

**That's it. Run your training script with Neptune.**

.. code-block:: Python

    import neptune
    neptune.init('USER_NAME/PROJECT_NAME')

    with neptune.create_experiment():
        neptune.send_metric('auc', 0.92)