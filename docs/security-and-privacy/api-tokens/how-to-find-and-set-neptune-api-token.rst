.. _how-to-setup-api-token:

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

Linux/IOS
^^^^^^^^^

.. code:: bash

    export NEPTUNE_API_TOKEN='YOUR_LONG_API_TOKEN'

Windows
^^^^^^^

.. code-block:: bat

    set NEPTUNE_API_TOKEN="YOUR_LONG_API_TOKEN"

or append this line to your ``~/.bashrc`` or ``~/.bash_profile`` files **(recommended)**.

.. warning:: Always keep your API token secret - it is like a password to the application. Appending the "export NEPTUNE_API_TOKEN='YOUR_LONG_API_TOKEN'" line to your ``~/.bashrc`` or ``~/.bash_profile`` file is the recommended method to ensure it remains secret.

Pass API token in scripts
-------------------------

Python
^^^^^^

Once your API token is set to the ``NEPTUNE_API_TOKEN`` environment variable you can simply skip the ``api_token`` argument of :meth:`~neptune.init`

.. code:: python

    neptune.init(project_qualified_name='YOUR_PROJECT')

R
^
The suggested way to pass your ``api_token`` is to store your key in an environment variable and pass it using ``Sys.getenv('MY_NEPTUNE_KEY')``.

.. code:: R

    init_neptune(project_name = 'my_workspace/my_project',
                 api_token = Sys.getenv('NEPTUNE_API_TOKEN')
                 )

.. tip::

    You can set your environment variable ``NEPTUNE_API_TOKEN`` with your API token directly from R with:

    .. code:: R

        Sys.setenv('NEPTUNE_API_TOKEN'='eyJhcGlfYWRkcmVzcyI6Imh0dHBz')