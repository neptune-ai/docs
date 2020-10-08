.. _api-token:

How to find and set Neptune API token
=====================================

Copy API token
--------------

``NEPTUNE_API_TOKEN`` is located under your user menu (top right side of the screen):

.. image:: ../../_static/images/security-and-privacy/api-tokens/get_token.gif
  :target: ../../_static/images/security-and-privacy/api-tokens/get_token.gif
  :alt: Get API token

Set to environment variable
---------------------------

Assign it to the bash environment variable:

Linux/IOS:

.. code:: bash

    export NEPTUNE_API_TOKEN='YOUR_LONG_API_TOKEN'

Windows:

.. code-block:: bat

    set NEPTUNE_API_TOKEN="YOUR_LONG_API_TOKEN"

or append this line to your ``~/.bashrc`` or ``~/.bash_profile`` files **(recommended)**.

.. warning:: Always keep your API token secret - it is like a password to the application. Appending the "export NEPTUNE_API_TOKEN='YOUR_LONG_API_TOKEN'" line to your ``~/.bashrc`` or ``~/.bash_profile`` file is the recommended method to ensure it remains secret.

