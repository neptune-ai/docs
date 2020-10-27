.. _integrations-index:

Integrations
============

.. image:: ../_static/images/integrations/framework-logos.png
   :target: ../_static/images/integrations/framework-logos.png
   :alt: selected integrations

Neptune comes with 25+ integrations with Python libraries popular in machine learning, deep learning and reinforcement learning.

How integrations work?
----------------------
Integrations are written using |neptune-client| and provide a convenient way to jump-start working with Neptune and a library that you are using. There is no need to integrate your code manually using neptune-client (it's easy though).

Each integration, that is installation, scope and usage example are explained in detail in the documentation (see: :ref:`PyTorch Lightning <integrations-pytorch-lightning>` for example).

Integrations are organized into the following categories:

- :ref:`Deep learning frameworks <integrations-deep-learning-frameworks>`
- :ref:`Machine learning frameworks <integrations-machine-learning-frameworks>`
- :ref:`Hyperparameter optimization libraries <integrations-hyperparameter-optimization-frameworks>`
- :ref:`Visualization libraries <integrations-visualization-tools>`
- :ref:`Experiment tracking frameworks <integrations-experiment-tracking-frameworks>`
- :ref:`Other integrations <integrations-other-integrations>`

.. note::

     |neptune-client|  is our official way of logging experiments and notebooks to Neptune (all integrations use it). If you need more control or explicit logging, you can always use it in your projects as well.

My library is not here. What now?
---------------------------------
There are two common paths:

#. You can always use  |neptune-client| , our open source Python library for logging all kinds of data and metadata to experiments.
#. Contact us directly via mail (contact@neptune.ai), chat (that little thing in the lower right corner) or post |forum| on our forum to discuss what you need and how we can deliver it.

.. toctree::
   :hidden:
   :maxdepth: 2

   Deep learning frameworks <deep_learning_frameworks.rst>
   Machine learning frameworks <machine_learning_frameworks.rst>
   Hyperparmeter optimization frameworks <hyperparams_opt_frameworks.rst>
   Visualization libraries <visualization_tools.rst>
   Experiment tracking frameworks <experiment_tracking_frmwks.rst>
   Other integrations <other-integrations.rst>

.. External links

.. |forum| raw:: html

    <a href="https://community.neptune.ai/c/feature-requests" target="_blank">feature request</a>

.. |neptune-client| raw:: html

    <a href="/api-reference/neptune/index.html" target="_blank">neptune-client</a>