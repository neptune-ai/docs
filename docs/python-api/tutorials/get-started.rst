Minimal example
===============

Below is the smallest possible example that follows the theme: *from zero to first Neptune experiment*.

Register
--------
Go here: https://neptune.ml/register to create free account.

Copy API token
--------------
``NEPTUNE_API_TOKEN`` is located under your User menu (top right side of the screen, like on the image below):

.. image:: ../../_static/images/tutorials/token.png
   :target: ../../_static/images/tutorials/token.png
   :alt: API token location

Assign it to the bash environment variable:

.. code:: bash

    export NEPTUNE_API_TOKEN='YOUR_LONG_API_TOKEN'

or append this line to your ``~/.bashrc`` or ``~/.bash_profile`` files **(recommended)**.

.. warning:: Always keep your API token secret - it is like password to the application. It is recommended to append "export NEPTUNE_API_TOKEN='YOUR_LONG_API_TOKEN'" line to your ``~/.bashrc`` or ``~/.bash_profile`` files.

Install neptune-client
----------------------
.. code:: bash

    pip install neptune-client

Install `psutil <https://psutil.readthedocs.io/en/latest/>`_ to see hardware monitoring charts
(please check psutil `documentation <https://psutil.readthedocs.io/en/latest/>`_ in case of installation problems):

.. code-block:: bash

    pip install psutil

Run Python script
-----------------
Save script below as ``minimal-example.py`` and run it like any other Python file: ``python minimal-example.py``.
You will see link to the experiment printed to the stdout.

.. tip::
    Make sure that you change ``USERNAME/sandbox`` (line 4 in the snippet below), to your username, that you picked at registration.

.. code:: Python

    import neptune
    import numpy as np

    # select project
    neptune.init('USERNAME/sandbox')

    # create experiment
    neptune.create_experiment(name='get-started-example-from-docs',
                              params={'n_iterations': 117})

    # send some metrics
    for i in range(1, 117):
        neptune.log_metric('iteration', i)
        neptune.log_metric('loss', 1/i**0.5)
        neptune.log_text('magic values', 'magic value {}'.format(0.95*i**2))

    neptune.set_property('model', 'lightGBM')

    # send some images
    for j in range(0, 5):
        array = np.random.rand(10, 10, 3)*255
        array = np.repeat(array, 30, 0)
        array = np.repeat(array, 30, 1)
        neptune.log_image('mosaics', array)

    neptune.stop()

Congrats! You just ran your first Neptune experiment and checked results online.

.. note:: What did you just learn? Few concepts:

    * how to run Neptune experiment
    * how to track it online
    * how to use basic Neptune client features, like *create_experiment()* and *send_metric()*
