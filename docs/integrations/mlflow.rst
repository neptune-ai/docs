MLflow
======
.. image:: ../_static/images/others/mlflow_neptuneml.png
   :target: ../_static/images/others/mlflow_neptuneml.png
   :alt: organize MLflow experiments in Neptune

|neptune-mlflow| is an open source project curated by Neptune team, that integrates |mlflow| with Neptune to let you get the best of both worlds.
Enjoy tracking and reproducibility of MLflow with organization and collaboration of Neptune.

With |neptune-mlflow| you can have your **MLflow experiment runs hosted in Neptune**.

Check example project in Neptune: |mlflow-integration|.

Quick-start
-----------
**Installation**

.. code-block:: bash

    pip install neptune-mlflow

**Sync your MLruns with Neptune**

Navigate to your MLflow project in your directory and run:

.. code-block:: bash

    neptune mlflow --project USER_NAME/PROJECT_NAME

Alternatively you can point to MLflow project directory:

.. code-block:: bash

    neptune mlflow /PATH/TO/MLflow_PROJECT --project USER_NAME/PROJECT_NAME

.. note:: That's it! You can now browse and collaborate on your MLflow runs in Neptune.

**Organize your mlflow experiments**

.. image:: ../_static/images/mlflow/mlflow_1.png
   :target: ../_static/images/mlflow/mlflow_1.png
   :alt: organize MLflow experiments in Neptune

**Share you work with others**

.. image:: ../_static/images/mlflow/mlflow_2.png
   :target: ../_static/images/mlflow/mlflow_2.png
   :alt: share artifacts logged during MLflow run

Examples
--------
.. toctree::
   :maxdepth: 1

   Sync and compare runs <mlflow/mlflow_compare_runs.rst>
   Sync runs and share model weights <mlflow/mlflow_sync_runs_and_share_model.rst>

Support
-------
If you find yourself in any trouble drop an issue or talk to us directly on the |support-chat|.

.. External links

.. |neptune-mlflow| raw:: html

    <a href="https://github.com/neptune-ai/neptune-mlflow" target="_blank">Neptune-mlflow</a>

.. |mlflow| raw:: html

    <a href="https://mlflow.org" target="_blank">MLflow</a>

.. |mlflow-integration| raw:: html

    <a href="https://ui.neptune.ai/jakub-czakon/mlflow-integration/experiments" target="_blank">MLflow integration</a>

.. |support-chat| raw:: html

    <a href="https://spectrum.chat/neptune-community" target="_blank">support chat</a>
