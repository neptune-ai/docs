What is Neptune?
================
|Neptune| is a collaboration hub for data science / machine learning teams. that focuses on three areas:

* :ref:`Track <track>`: all metrics and outputs in your data science or machine learning project. It can be model training curves, visualizations, input data, calculated features and so on.
* :ref:`Organize <organize>`: automatically transform tracked data into a knowledge repository.
* :ref:`Collaborate <collaborate>`: share, compare and discuss your work across data science project.

.. image:: ./_static/images/overview/quick_overview.gif
   :target: ./_static/images/overview/quick_overview.gif
   :alt: image

Neptune is lightweight
----------------------
Neptune is built with the single design principle in mind: *being lightweight*. What does it mean in practice?

* easy user onboarding: if you know how to use ``print()`` you will learn how to use it in no time.
* 20-minute deployment: use SaaS, deploy on any cloud or own hardware (|contact us| to learn more).
* Neptune fits in any workflow, ranging from data exploration & analysis, decision science to machine learning and deep learning.
* Neptune works with common technologies in data science domain: Python 2 and 3, Jupyter Notebooks, R.
* Neptune integrates with other tools like |MLflow| and |TensorBoard|.

Neptune's focus: track, organize and collaborate
------------------------------------------------
.. _track:

Track
^^^^^
Track all objects in the data science or machine learning project. It can be model training curves, visualizations, input data, calculated features and so on. 
Snippet below, presents example integration with Python code.

.. code-block::

   import neptune

   neptune.init('shared/onboarding',
                api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiJiNzA2YmM4Zi03NmY5LTRjMmUtOTM5ZC00YmEwMzZmOTMyZTQifQ==')
   with neptune.create_experiment():
       neptune.append_tag('minimal-example')
       n = 117
       for i in range(1, n):
           neptune.send_metric('iteration', i)
           neptune.send_metric('loss', 1/i**0.5)
       neptune.set_property('n_iterations', n)

``api_token`` belongs to the public user |Neptuner|. So, when started you can see your experiment at the top of |experiments view|.

.. _organize:

Organize
^^^^^^^^
Organize structure of your project: 

* Code
* Notebooks
* Experiment results
* Model weights
* Meeting notes 
* Reports

Everything is in one place, accessible from the app or programmatically.
Neptune exposes :ref:`Query API <query-api>`, that allows users to access their Neptune data right from the Python code.

.. _collaborate:

Collaborate
^^^^^^^^^^^
Collaborate in the team:

* share your experiments,
* compare results,
* comment and communicate your work,
* use widgets and mentions to show your progress.

Speak Your language in our data-science focused interactive wiki!

.. image:: ./_static/images/overview/wiki.gif
   :target: ./_static/images/overview/wiki.gif
   :alt: image

Documentation contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Learn about Neptune

   learn-about-neptune/core-concepts.rst
   learn-about-neptune/team-management.rst
   learn-about-neptune/nql.rst
   learn-about-neptune/faq.rst

.. toctree::
   :maxdepth: 2
   :caption: Notebooks

   Introduction <notebooks/introduction.rst>
   Installation for Jupyter and JupyterLab <notebooks/installation.rst>
   Configuration <notebooks/configuration.rst>
   Integrations (AWS, SageMaker, Colab) <notebooks/integrations.rst>
   Troubleshoot <notebooks/troubleshoot.rst>

.. toctree::
   :maxdepth: 2
   :caption: Python Library

   python-api/introduction.rst
   python-api/api-reference.rst
   python-api/query-api.rst
   python-api/tutorials.rst
   python-api/how-to-guides.rst
   python-api/cheatsheet.rst

.. toctree::
   :maxdepth: 2
   :caption: Integrations

   Frameworks <integrations/frameworks.rst>
   R <integrations/r-support.rst>
   TensorBoard <integrations/tensorboard.rst>
   MLflow <integrations/mlflow.rst>
   Fast.ai <integrations/fast_ai.rst>
   PyTorchLightning <integrations/pytorch_lightning.rst>
   Catalyst <integrations/catalyst.rst>
   PyTorch Ignite <integrations/pytorch_ignite.rst>
   Neptune Contrib <integrations/neptune-contrib.rst>

.. External links

.. |Neptune| raw:: html

    <a href="https://neptune.ai/" target="_blank">Neptune</a>

.. |contact us| raw:: html

    <a href="mailto:contact@neptune.ai" target="_blank">contact us</a>

.. |MLflow| raw:: html

    <a href="https://mlflow.org/" target="_blank">MLflow</a>

.. |TensorBoard| raw:: html

    <a href="https://www.tensorflow.org/guide/summaries_and_tensorboard" target="_blank">TensorBoard</a>

.. |Neptuner| raw:: html

    <a href="https://ui.neptune.ai/o/shared/neptuner" target="_blank">Neptuner</a>

.. |experiments view| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/onboarding/experiments" target="_blank">experiments view</a>
