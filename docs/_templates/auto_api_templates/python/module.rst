{% if not obj.display %}
:orphan:

{% endif %}

:mod:`{{ obj.name }}`
======={{ "=" * obj.name|length }}

{% if 'neptune' == obj.name %}

.. image:: ../../../../_static/images/api_references/Neptune_Object_Hierarchy_V1.jpg
   :target: ../../../../_static/images/api_references/Neptune_Object_Hierarchy_V1.jpg
   :alt: Neptune object hierarchy
   
The following are the main classes in the Neptune client Python library:

.. csv-table::
   :header: "Class","Description"
   :widths: 10, 40
   :delim: #

   |Neptune| #A global object that provides the convenience of doing most of the logging using a single neptune global variable, similar to Numpy's ``import numpy as np`` statement. In Neptune, you write ``import neptune``.
   |Project| #This is the Neptune project to which you want to log things. You need to create it in the application. This is a place where you can create experiments. You can create new ones and update or download information from the existing one.
   |Experiment| #This is an object to which you log any piece of information you consider to be important during your run. Interaction with the experiment feels similar to interacting with a singleton dictionary object. Neptune gives you all the freedom - you simply log metrics, images, text, and everything else to particular names and those objects are sent to the application. You can have one or multiple experiments in one script. You can re-instantiate the experiments you have created in the past and update them.
   |Notebook| #Contains all the information about a Neptune Jupyter Notebook.
   |Git Info| #Keeps information about the Git repository in an experiment.

Learn how to get started with logging and managing experiment data using Neptune :ref:`here <guides-logging-and-managing-experiment-results>`.

{% endif %}

{% if 'neptunecontrib' == obj.name %}

This library contains community extensions. This is what you use to integrate Neptune with other frameworks. Check all the integrations Neptune supports :ref:`here <integrations-index>`.

{% endif %}

{% if 'neptune_tensorboard' == obj.name %}

This library supports packages and functions for Neptune's integration with TensorBoard. Read how to integrate Neptune with TensorBoard :ref:`here <integrations-tensorboard>`.

{% endif %}

{% if 'neptune.experiments' == obj.name %}

An Experiment is everything that you log to Neptune, beginning at ``neptune.create_experiment()`` and ending when script finishes or when you explicitly stop the experiment with ``neptune.stop`` (reference docs: :meth:`~neptune.experiments.Experiment.stop`).

Creating experiments is easy:

.. code-block:: python3

    # Set project
    neptune.init('my_workspace/my_project')

    # Create new experiment
    neptune.create_experiment()

You can now log various data to the experiment including metrics, losses, model weights, images, predictions and much more. Have a look at the complete list of :ref:`what you can log <what-you-can-log>` to the experiment.

Besides logging data, you can also :ref:`download experiment data <guides-download_data>` to you local machine or :ref:`update an existing experiment <update-existing-experiment>` even when it's closed.

.. note::

    ``neptune.log_metric('some_name', some_value)`` is for tracking all numeric values to Neptune (metric, loss, score, variances, etc.). Learn, what else can be tracked to experiment from :ref:`this list <what-you-can-log>`.

{% endif %}

{% if 'neptune.projects' == obj.name %}

A Project is a **collection of Experiments**, created by user (or users) assigned to the project.

You can log experiments to the project or fetch all experiments that satisfy some criteria.

.. code-block:: python3

    # Set project and get project object
    project = neptune.init('my_workspace/my_project')

    # Use project to create experiment
    project.create_experiment()

    # Use project to get experiments data from the project
    project.get_leaderboard(state=['succeeded'])

Learn more about :ref:`downloading data from Neptune <guides-download_data>`.

{% endif %}

.. py:module:: {{ obj.name }}

{% if obj.docstring %}
.. autoapi-nested-parse::

   {{ obj.docstring|prepare_docstring|indent(3) }}

{% endif %}

{% block subpackages %}
{% set visible_subpackages = obj.subpackages|selectattr("display")|list %}
{% if visible_subpackages %}
Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

{% for subpackage in visible_subpackages %}
   {{ subpackage.short_name }}/index.rst
{% endfor %}


{% endif %}
{% endblock %}
{% block submodules %}
{% set visible_submodules = obj.submodules|selectattr("display")|list %}
{% if visible_submodules %}
Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

{% for submodule in visible_submodules %}
   {{ submodule.short_name }}/index.rst
{% endfor %}


{% endif %}
{% endblock %}
{% block content %}
{% if obj.all is not none %}
{% set visible_children = obj.children|selectattr("short_name", "in", obj.all)|list %}
{% elif obj.type is equalto("package") %}
{% set visible_children = obj.children|selectattr("display")|list %}
{% else %}
{% set visible_children = obj.children|selectattr("display")|rejectattr("imported")|list %}
{% endif %}
{% if visible_children %}
{{ obj.type|title }} Contents
{{ "-" * obj.type|length }}---------

{% set visible_classes = visible_children|selectattr("type", "equalto", "class")|list %}
{% set visible_functions = visible_children|selectattr("type", "equalto", "function")|list %}
{% if "show-module-summary" in autoapi_options and (visible_classes or visible_functions) %}
{% block classes scoped %}
{% if visible_classes %}
Classes
~~~~~~~

.. autoapisummary::

{% for klass in visible_classes %}
   {{ klass.id }}
{% endfor %}


{% endif %}
{% endblock %}

{% block functions scoped %}
{% if visible_functions %}
Functions
~~~~~~~~~

.. autoapisummary::

{% for function in visible_functions %}
   {{ function.id }}
{% endfor %}


{% endif %}
{% endblock %}
{% endif %}
{% for obj_item in visible_children %}
{{ obj_item.rendered|indent(0) }}
{% endfor %}
{% endif %}
{% endblock %}

.. External links

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