.. _quick-starts-index:

Quick starts
============

.. toctree::
   :maxdepth: 1

   Hello World (1 min) <hello-world.rst>
   How to monitor experiments live (2 min) <how-to-monitor-live.rst>
   How to version and organize experiments (5 min) <how-to-organize-experiments.rst>
   How to version Jupyter notebooks (2 min) <how-to-version-notebooks.rst>

Using Neptune in 30 seconds
---------------------------

Step 1: Install client
**********************

.. code:: bash

    pip install neptune-client

Step 2: Create a Neptune experiment
***********************************

.. code:: python

    import neptune

    neptune.init(project_qualified_name='', api_token='') # add your credentials
    neptune.create_experiment()

Step 3: Log whatever you want
*****************************

.. code:: python

    neptune.log_metric('accuracy', 0.83)
