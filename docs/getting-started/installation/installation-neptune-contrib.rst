Install neptune-contrib: Neptune extensions contributed by the community (3 min)
================================================================================

|neptune-contrib| is an open-source library with Neptune extensions created by the community.

Installing minimal contrib
--------------------------

To install it, open a terminal or CMD (depending on your operating system) and run this command:

.. code:: bash

    pip install neptune-contrib

Installing submodule dependencies
---------------------------------

Neptune contrib comes with a few helper submodules:

- neptunecontrib.api: extensions to the neptune api
- neptunecotnrib.bots: accessing experiment data through a bot
- neptunecontrib.hpo: Neptune utils for hyperparameter optimization
- neptunecontrib.monitoring: Neptune utils for various machine learning frameworks
- neptunecontrib.versioning: Data versioning utils
- neptunecontrib.viz: Vizualization utils

Some of the dependencies may not be installed with the minimal installation.

To install them you can add the suffix [SUBMODULE] to your pip command.

For example:

.. code:: bash

    pip install neptune-contrib[monitoring]

It will install all the minimal dependencies as well as the specific ``neptunecontrib.monitoring`` dependences.

Installing all dependencies
---------------------------

You can also install all the dependencies.

.. warning:: this may install a lot of things that you don't need.

.. code:: bash

    pip install neptune-contrib[all]


What is next?
-------------

- Check out |Quick starts|

.. |Quick starts| raw:: html

    <a href="/getting-started/quick-starts/index.html">Quick Starts</a>

.. |neptune-contrib| raw:: html

    <a href="https://github.com/neptune-ai/neptune-contrib" target="_blank">neptune-contrib</a>
