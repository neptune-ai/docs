Neptune Python Library Reference
--------------------------------

.. contents::
    :local:
    :depth: 1
    :backlinks: top

Main Library Classes
====================

The following are the main classes in the `Neptune Python Library <introduction.html>`_:

.. csv-table::
   :header: "Class","Description"
   :widths: 10, 40
   :delim: #

   `Neptune <../neptune-client/docs/neptune.html>`_#A global object that provides the convenience of doing most of the logging using a single neptune global variable, similar to Numpy’s ``import numpy as np`` statement - in Neptune, write ``import neptune``.
   `Session <../neptune-client/docs/session.html>`_#When you are creating a Neptune session, you identify yourself with an API token so that the client knows which projects you have access to.
   `Project <../neptune-client/docs/project.html>`_#This is the Neptune project to which you want to log things. You need to create it in the application. This is a place where you can create experiments. You can create new ones and update or download information from the existing one.
   `Experiment <../neptune-client/docs/experiment.html>`_#This is an object to which you log any piece of information you consider to be important during your run. Interaction with the experiment feels similar to interacting with a Singleton dictionary object. Neptune gives you all the freedom: You simply log metrics, images, text and everything else to particular names and those objects are sent to the application. You can have one or multiple experiments in one script. You can reinstantiate the experiments you have created in the past and update them.
   `Notebook <../neptune-client/docs/notebook.html>`_#Contains all the information about a Neptune Jupyter Notebook.
   `Utils <../neptune-client/docs/utils.html>`_#Keeps information about the Git repository in an experiment.


Downloading Project and Experiment Data from Neptune
====================================================

A subset of the Neptune Python Library provides methods for downloading project and experiment data from Neptune.

For more information, see `Fetching Data From Neptune <fetch-data.html>`_.

