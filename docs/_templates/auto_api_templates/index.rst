API Reference
=============

The Neptune ecosystem consists of the following major libraries: 

|neptune-client|
~~~~~~~~~~~~~~~~
The main Python client. It has the following main classes:

.. csv-table::
   :header: "Class","Description"
   :widths: 10, 40
   :delim: #

   |Neptune| #A global object that provides the convenience of doing most of the logging using a single neptune global variable, similar to Numpy's ``import numpy as np`` statement. In Neptune, you write ``import neptune``.
   |Project| #This is the Neptune project to which you want to log things. You need to create it in the application. This is a place where you can create experiments. You can create new ones and update or download information from the existing one.
   |Experiment| #This is an object to which you log any piece of information you consider to be important during your run. Interaction with the experiment feels similar to interacting with a singleton dictionary object. Neptune gives you all the freedom - you simply log metrics, images, text, and everything else to particular names and those objects are sent to the application. You can have one or multiple experiments in one script. You can re-instantiate the experiments you have created in the past and update them.
   
Learn how to get started with logging and managing experiment data using Neptune :ref:`here <guides-logging-and-managing-experiment-results>`.

|neptune-contrib|
~~~~~~~~~~~~~~~~~
The library with community extensions. This is what you use to integrate Neptune with other frameworks. Check all the integrations Neptune supports :ref:`here <integrations-index>`.
	
|neptune-tensorboard|
~~~~~~~~~~~~~~~~~~~~~
Supports packages and functions for Neptune's integration with TensorBoard. Read how to integrate Neptune with TensorBoard :ref:`here <integrations-tensorboard>`.

.. toctree::
   :hidden:
   :titlesonly:

   /api-reference/neptune/index
   /api-reference/neptunecontrib/index
   /api-reference/neptune_tensorboard/index
   
.. External links

.. |neptune-client| raw:: html

    <a href="/api-reference/neptune/index.html" >neptune-client</a>

.. |Neptune| raw:: html

    <a href="/api-reference/neptune/index.html#functions" target="_blank">Neptune</a>

.. |Session| raw:: html

    <a href="/api-reference/neptune/sessions/index.html?highlight=neptune%20sessions%20session#neptune.sessions.Session" target="_blank">Session</a>

.. |Project| raw:: html

    <a href="/api-reference/neptune/projects/index.html#neptune.projects.Project" target="_blank">Project</a>

.. |Experiment| raw:: html

    <a href="/api-reference/neptune/experiments/index.html?highlight=neptune%20experiment#neptune.experiments.Experiment" target="_blank">Experiment</a>

.. |Notebook| raw:: html

    <a href="/api-reference/neptune/notebook/index.html?highlight=notebook#neptune.notebook.Notebook" target="_blank">Notebook</a>

.. |Git Info| raw:: html

    <a href="/api-reference/neptune/git_info/index.html#neptune.git_info.GitInfo" target="_blank">Git Info</a>

.. |neptune-contrib|  raw:: html

    <a href="/api-reference/neptunecontrib/index.html" >neptune-contrib</a>

.. |neptune-tensorboard| raw:: html

    <a href="/api-reference/neptune_tensorboard/index.html">neptune-tensorboard</a>
