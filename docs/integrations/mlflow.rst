.. _integrations-mlflow:

Neptune-MLflow Integration
==========================

|Run on Colab|

What will you get with this integration?
----------------------------------------

|mlflow-tour|

|neptune-mlflow| is an open source project curated by Neptune team that enables |mlflow| experiment runs to be hosted on Neptune.

The integration lets you enjoy the best of both worlds: the tracking and reproducibility of MLflow with the organization and collaboration of Neptune.

.. note::

    This integration is tested with ``mlflow==1.12.1``
	
.. image:: ../_static/images/integrations/mlflow_neptuneml.png
   :target: ../_static/images/integrations/mlflow_neptuneml.png
   :alt: organize MLflow experiments in Neptune

.. _mlflow-quickstart:

Quickstart
----------

This quickstart will show you how to:

* Install the necessary neptune packages
* Sync your MLruns with Neptune.

|Run on Colab|

.. _mlflow-before-you-start-basic:

Before you start
^^^^^^^^^^^^^^^^
#. This integration needs you to have your Personal API token. You need a Neptune account to have this. Create one for free |neptune-register| if you haven't already

#. Ensure that you have ``Python 3.x`` and following libraries installed:

   * ``neptune-mlflow``
   * ``mlflow==1.12.1``. See |mlflow-install|.
   
   .. code-block:: bash
   	
      pip install --quiet mlflow neptune-mlflow

#. You also need minimal familiarity with Mlflow. Have a look at the |mlflow-guide| guide to get started.

Step 1: Set your ``NEPTUNE_API_TOKEN``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Linux/iOS:

.. code:: bash

    export NEPTUNE_API_TOKEN='YOUR_API_TOKEN'

Windows:

.. code-block:: bat

    set NEPTUNE_API_TOKEN="YOUR_API_TOKEN"

.. tip::

    Read more about how to :ref:`securely set the Neptune API token <how-to-setup-api-token>`.

Step 2: Sync your MLruns with Neptune
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Navigate to the MLflow project in your directory and run:

.. code-block:: bash

    neptune mlflow --project USER_NAME/PROJECT_NAME

Alternatively, you can point to the MLflow project directory:

.. code-block:: bash

    neptune mlflow /PATH/TO/MLflow_PROJECT --project USER_NAME/PROJECT_NAME

**That's it! You can now browse and collaborate on your MLflow runs in Neptune.**

Check out this |mlflow-integration|.

Organize and share your MLflow experiments
------------------------------------------

.. image:: ../_static/images/integrations/mlflow_1.png
   :target: ../_static/images/integrations/mlflow_1.png
   :alt: organize MLflow experiments in Neptune


.. image:: ../_static/images/integrations/mlflow_2.png
   :target: ../_static/images/integrations/mlflow_2.png
   :alt: share artifacts logged during MLflow run

.. External links

.. |Run on Colab| raw:: html

    <div class="run-on-colab">
        <button><a target="_blank"
                   href="https://colab.research.google.com//github/neptune-ai/neptune-examples/blob/master/integrations/mlflow/docs/Neptune-Mlflow.ipynb"><img
                width="50" height="50" style="margin-right:10px"
                src="https://neptune.ai/wp-content/uploads/colab_logo_120.png">Run in
            Google Colab</a></button>
        <button>
            <a target="_blank" href="https://github.com/neptune-ai/neptune-examples/blob/master/integrations/mlflow/docs/Neptune-MLflow.py">
                <img width="50" height="50" style="margin-right:10px"
                     src="https://neptune.ai/wp-content/uploads/GitHub-Mark-120px-plus.png">
                View source on GitHub
            </a>
        </button>
    </div>

.. |mlflow-tour| raw:: html

	<div style="position: relative; padding-bottom: 53.65126676602087%; height: 0;">
		<iframe src="https://www.loom.com/embed/a2caf18b8de148a9ad56e18ad0787cb2" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;">
		</iframe>
	</div>

.. |neptune-mlflow| raw:: html

    <a href="https://github.com/neptune-ai/neptune-mlflow" target="_blank">Neptune-MLflow</a>

.. |mlflow| raw:: html

    <a href="https://mlflow.org" target="_blank">MLflow</a>
	
.. |neptune-register| raw:: html

    <a href="https://neptune.ai/register" target="_blank">here</a>

.. |mlflow-install| raw:: html

    <a href="https://mlflow.org/docs/latest/quickstart.html#installing-mlflow" target="_blank">MLflow Installation Guide</a>

.. |mlflow-guide| raw:: html

    <a href="https://mlflow.org/docs/latest/quickstart.html" target="_blank">MLflow Quickstart</a>

.. |mlflow-integration| raw:: html

    <a href="https://ui.neptune.ai/jakub-czakon/mlflow-integration/experiments" target="_blank">example experiment</a>
