Integrations
============

.. image:: ../_static/images/overview/framework-logos.png
   :target: ../_static/images/overview/framework-logos.png
   :alt: selected integrations

Neptune comes with 25+ integrations with Python libraries popular in machine learning, deep learning and reinforcement learning.

How integrations work?
----------------------
Integrations are written using `Neptune-client <../python-api/introduction.html>`_ and provide a convenient way to jump-start working with Neptune and a library that you are using. There is no need to integrate your code manually using neptune-client (it's easy though).

Each integration, that is installation, scope and usage example are explained in detail in the documentation (see: `LightGBM <lightgbm.html>`_ for example).

Integrations are organized into the following categories:

* `Deep learning frameworks <deep_learning_frameworks.html>`_
* `Machine learning frameworks <machine_learning_frameworks.html>`_
* `Hyperparameter optimization libraries <hyperparams_opt_frameworks.html>`_
* `Visualization libraries <visualization_tools.html>`_
* `Messaging systems <messaging_systems.html>`_
* `Experiment tracking frameworks <experiment_tracking_frmwks.html>`_
* `Neptune Extensions Library <neptune-contrib.html>`_
* `Cloud providers <cloud_providers.html>`_

.. note::
    `Neptune-client <../python-api/introduction.html>`_ is our official way of logging experiments and notebooks to Neptune (all integrations use it). If you need more control or explicit logging, you can always use it in your projects as well.

My library is not here. What now?
---------------------------------
There are two common paths:

#. You can always use `neptune-client <../python-api/introduction.html>`_, our open source Python library for logging all kinds of data and metadata to `experiments <../learn-about-neptune/experiment_tracking.html>`_.
#. Contact us directly via mail (contact@neptune.ai), chat (that little thing in the lower right corner) or post |forum| on our forum to discuss what you need and how we can deliver it.

.. External links

.. |forum| raw:: html

    <a href="https://community.neptune.ai/c/feature-requests" target="_blank">feature request</a>


.. toctree::
   :maxdepth: 2

   Languages <languages.rst>
   Cloud providers <cloud_providers.rst>
   Deep learning frameworks <deep_learning_frameworks.rst>
   Machine learning frameworks <machine_learning_frameworks.rst>
   Hyperparmeter optimization frameworks <hyperparams_opt_frameworks.rst>
   Visualization tools <visualization_tools.rst>
   Explainability tools <explainability_tools.rst>
   Messaging systems <messaging_systems.rst>
   Experiment tracking frameworks <experiment_tracking_frmwks.rst>
   Neptune extensions library <neptune-contrib.rst>