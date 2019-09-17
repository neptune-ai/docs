MLflow
======
|neptune-mlflow| is an open source project curated by Neptune team, that integrates |mlflow| with Neptune to let you get the best of both worlds. Enjoy tracking and reproducibility of MLflow with organization and collaboration of Neptune.

With |neptune-mlflow| you can have your mlflow experiment runs hosted in Neptune.

Resources
---------
* Project on GitHub: |neptune-mlflow|
* Documentation: `Neptune integration with MLflow <https://neptune-mlflow.readthedocs.io/en/latest>`_
* Example project in Neptune: |mlflow-integration|.

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


.. External links

.. |neptune-mlflow| raw:: html

    <a href="https://github.com/neptune-ml/neptune-mlflow" target="_blank">Neptune-mlflow</a>

.. |mlflow| raw:: html

    <a href="https://mlflow.org" target="_blank">MLflow</a>

.. |mlflow-integration| raw:: html

    <a href="https://ui.neptune.ml/jakub-czakon/mlflow-integration/experiments" target="_blank">MLflow integration</a>
