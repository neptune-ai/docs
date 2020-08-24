Installation SaaS
=================

How does Neptune SaaS work?
---------------------------

Neptune SaaS (Software as a Service) is a web application where you can log your experiment data, explore them, and share with other people.

In a nutshell:

We:

- maintain the cloud application on https://neptune.ai
- backup your experiment data
- manage experiment storage

You:

- create an account in |Neptune|
- install |neptune-client| library
- log data via |neptune-client| library
- log in to your |Neptune web app| to see your experiments

How do I install neptune-client library
-------------------------------------

Before you start
****************

Make sure you meet the following prerequisites before starting:

- Have Python 3.x installed

.. note:: if you are using R or any other language you need to have Python 3.x installed as well

Install neptune-client
**********************

Depending on your operating system run

Linux | Mac
###########

Open a terminal and run this command:

**pip**

.. code:: bash

    pip install neptune-client

**conda**

.. code:: bash

    conda install -c conda-forge neptune-client

Windows
#######

Open a CMD and run this command:

**pip**

.. code:: bash

    pip install neptune-client

**conda**

.. code:: bash

    conda install -c conda-forge neptune-client


Once you have successfully installed the package, add import neptune to your code to use neptune-client capabilities. See the following example.

What is next?
-------------

- Check out |Quick starts|

.. |Quick starts| raw:: html

    <a href="/getting-started/quick-starts.html">Quick Starts</a>

- TODO

.. |neptune-client| raw:: html

    <a href="https://github.com/neptune-ai/neptune-client" target="_blank">neptune-client</a>

.. |Neptune| raw:: html

    <a href="https://neptune.ai/" target="_blank">Neptune</a>

.. |Neptune web app| raw:: html

    <a href="https://ui.neptune.ai/" target="_blank">Neptune web app</a>


