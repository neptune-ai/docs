Welcome to Neptune!
===================

Neptune is a light-weight experiment management tool that helps you keep track of your machine learning experiments.

Use Neptune to log hyperparameters and output metrics from your runs, then visualize and compare results. Automatically transform tracked data into a knowledge repository, then share and discuss your work with colleagues.

- Neptune fits in any workflow, ranging from data exploration and analysis, decision science to machine learning and deep learning.
- Neptune works with common technologies in the data science domain: Python, `Jupyter Notebooks <https://docs.neptune.ai/notebooks/introduction.html>`_, and `R <https://docs.neptune.ai/integrations/r-support.html>`_, to mention a few.
- It integrates with other tracking tools such as `MLflow <https://docs.neptune.ai/integrations/mlflow.html#>`_ and `TensorBoard <https://docs.neptune.ai/integrations/tensorboard.html#>`_ or `Sacred <https://neptune-contrib.readthedocs.io/examples/observer_sacred.html>`_ and many other machine learning and deep learning frameworks.
- It integrates seamlessly with your machine learning infrastructure, be it AWS, GCP, Kubernetes, Azure, or on-prem machines.
- `Neptune client <https://github.com/neptune-ai/neptune-client>`_ is an open source Python library that allows you to integrate your Python scripts with Neptune. Neptune client supports the following cases:

    - Creating and tracking experiments
    - Managing running experiment
    - Querying experiments and projects (search/download)

Get Started
===========

- New user? `Register <https://neptune.ai/register>`_ and climb aboard.
- Registered already? Log in `here <https://neptune.ai/login>`_, then click **Getting Started** and follow the onboarding instructions:

    .. image:: ./_static/images/core-concepts/getting_started_onboarding.png
        :target: ./_static/images/core-concepts/getting_started_onboarding.png
        :alt: Get Started Onboarding

- Take a look at Neptune Project starter code in our `sample project <https://ui.neptune.ai/o/USERNAME/org/example-project/wiki/2-Installation-and-minimal-example-cd2b3338-6629-40cc-966c-b455c62a90b3>`_.


Track, Organize, Collaborate
============================

.. image:: ./_static/images/overview/quick_overview.gif
   :target: ./_static/images/overview/quick_overview.gif
   :alt: image


The Neptune workflow comprises three iterative phases:

- **Track** all objects in the data science or machine learning project. It can be model training curves, visualizations, input data, calculated features and so on. The snippet below presents an example of integration with Python code.

    .. code-block:: python

        import neptune

        neptune.init('shared/onboarding', api_token='ANONYMOUS')
        neptune.create_experiment()

        neptune.append_tag('minimal-example')
        n = 117
        for i in range(1, n):
            neptune.send_metric('iteration', i)
            neptune.send_metric('loss', 1/i**0.5)
            neptune.set_property('n_iterations', n)

    .. note::
        The `api_token` belongs to the public user Neptuner. After running the code, your experiment will appear on the experiments `dashboard <https://ui.neptune.ai/shared/onboarding/experiments>`_.

    For more information, see `Experiment Tracking <learn-about-neptune/experiment_tracking.html>`_.

- **Organize** the structure of your project:

    - Code
    - Notebooks
    - Experiment results
    - Model weights
    - Meeting notes
    - Reports

    Everything is in one place, accessible from the application or programmatically. Neptune exposes a Query API, that allows users to access their Neptune data right from the Python code.

    For more information, see `Experiments View <learn-about-neptune/ui.html#experiments-view>`_.

- **Collaborate** in the team:

    - Share your experiments
    - Compare results
    - Comment and communicate your work
    - Use widgets and mentions to show your progress
    - Speak your language in our data-science focused interactive wiki!

        .. image:: ./_static/images/overview/wiki.gif
           :target: ./_static/images/overview/wiki.gif
           :alt: image

    For more information, see `Collaborating in Neptune <learn-about-neptune/collaborate.html>`_.


More Resources
==============

In addition to this documentation set, check out the following resources:

- `Hands-on tutorial <https://ui.neptune.ai/o/USERNAME/org/example-project/wiki/1-Intro-89a74d1e-c71d-4764-912a-63312c3e885c>`_: Covers installation, experiment tracking and comparison, data tracking, and Notebook use.
- Sample projects like a `comparison of binary classification metrics <https://ui.neptune.ai/neptune-ai/binary-classification-metrics/wiki/README-12ff3437-42e3-48c9-af34-957822849559>`_ applied to fraud detection, `research on hyperparameter optimization strategies <https://ui.neptune.ai/jakub-czakon/blog-hpo/wiki/Skopt-forest-51912822-7a61-42ad-87d1-108998739c73>`_, or a `step-by-step experiment tracking tutorial <https://ui.neptune.ai/USERNAME/example-project/wiki/1-Intro-89a74d1e-c71d-4764-912a-63312c3e885c>`_.
- `YouTube channel <https://www.youtube.com/channel/UCvOJU-ubyUqxGSDRN7xK4Ng>`_: Provides hands-on videos that showcase key Neptune features.
- `Blog <https://neptune.ai/blog>`_: Provides in-depth articles about best practices and trends in machine learning.
- `Neptune user community <https://spectrum.chat/neptune-community?tab=posts>`_: Meet other Neptune users and developers and start a discussion.
- `neptune-contrib <https://neptune-contrib.readthedocs.io/index.html>`_ : Built on top of neptune-client, this is an open-source collection of advanced utilities that make work with Neptune easier.
- `Product hunt <https://www.producthunt.com/posts/neptune-ai>`_: A review helps other people find our product.
- Presentations, talks, podcasts
- Technical support: Should you require further support, or have feature requests, reach out at contact@neptune.ai or click the chat icon in the bottom right corner of the Neptune UI.

Spread the Love
===============

Go ahead and mention us on social media!

- `Twitter <https://twitter.com/neptune_ai>`_: Tweet us. Our handle is @neptune.ai.
- Product feedback: File an issue or suggest a feature or improvement in our `GitHub repo <https://github.com/neptune-ai/neptune-client/issues>`_.

.. ----------------------
.. Documentation contents

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting Started with Neptune

   learn-about-neptune/ui.rst
   learn-about-neptune/experiment_tracking.rst
   learn-about-neptune/team-management.rst
   learn-about-neptune/collaborate.rst
   learn-about-neptune/nql.rst
   learn-about-neptune/deployment.rst
   learn-about-neptune/faq.rst

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Jupyter Notebooks in Neptune

   Using Jupyter Notebooks <notebooks/introduction.rst>
   Installation for Jupyter and JupyterLab <notebooks/installation.rst>
   Configuration <notebooks/configuration.rst>
   Troubleshoot <notebooks/troubleshoot.rst>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Python Library

   python-api/introduction.rst
   python-api/api-reference.rst
   python-api/query-api.rst
   python-api/sample_project.rst
   python-api/cheatsheet.rst

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Integrations

   Keras <integrations/keras.rst>
   PyTorch <integrations/pytorch.rst>
   LightGBM <integrations/lightgbm.rst>
   Matplotlib <integrations/matplotlib.rst>
   R <integrations/r-support.rst>
   Any language <integrations/any-language-support.rst>
   TensorBoard <integrations/tensorboard.rst>
   HiPlot <integrations/hiplot.rst>
   MLflow <integrations/mlflow.rst>
   Sacred <integrations/sacred.rst>
   Fast.ai <integrations/fast_ai.rst>
   PyTorchLightning <integrations/pytorch_lightning.rst>
   Catalyst <integrations/catalyst.rst>
   PyTorch Ignite <integrations/pytorch_ignite.rst>
   Skorch <integrations/skorch.rst>
   XGBoost <integrations/xgboost.rst>
   Scikit Optimize <integrations/skopt.rst>
   Optuna <integrations/optuna.rst>
   Telegram <integrations/telegram.rst>
   Neptune-enabled JupyterLab on AWS and AWS SageMaker <notebooks/integrations.rst>
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
