MLflow
======
`Neptune-mlflow <https://github.com/neptune-ml/neptune-mlflow>`_ is an open source project curated by Neptune team, that integrates `MLflow <https://mlflow.org>`_ with Neptune to let you get the best of both worlds. Enjoy tracking and reproducibility of MLflow with organization and collaboration of Neptune.

With `neptune-mlflow <https://github.com/neptune-ml/neptune-mlflow>`_ you can have your mlflow experiment runs hosted in Neptune.

Resources
---------
* Project on GitHub: `neptune-mlflow <https://github.com/neptune-ml/neptune-mlflow>`_
* Documentation: `Neptune integration with MLflow <https://neptune-mlflow.readthedocs.io/en/latest>`_
* Example project in Neptune: `mlflow-integration <https://ui.neptune.ml/jakub-czakon/mlflow-integration/experiments>`_.

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

What next?
----------
#. Go to project `documentation <https://neptune-mlflow.readthedocs.io/en/latest>`_ to learn more.
#. Check our `blog <https://medium.com/neptune-ml>`_, where Neptune team publishes posts about variety of data science-related topics.

TensorBoard
===========
`Neptune-tensorboard <https://github.com/neptune-ml/neptune-tensorboard>`_ is an open source project curated by Neptune team, that integrates `TensorBoard <https://www.tensorflow.org/guide/summaries_and_tensorboard>`_ with Neptune to let you get the best of both worlds.

With `neptune-tensorboard <https://github.com/neptune-ml/neptune-tensorboard>`_ you can have your TensorBoard visualizations hosted in Neptune.

Resources
---------
* Project on GitHub: `neptune-tensorboard <https://github.com/neptune-ml/neptune-tensorboard>`_
* Documentation: `TensorBoard integration with Neptune <https://neptune-tensorboard.readthedocs.io/en/latest/>`_
* Example project in Neptune: `tensorboard-integration <https://ui.neptune.ml/jakub-czakon/tensorboard-integration/experiments>`_.

Quick-start
-----------
**Installation**

.. code-block:: bash

    pip install neptune-tensorboard

**Sync TensorBoard logdir with Neptune**

Point Neptune to your TensorBoard logs directory:

.. code-block:: bash

    neptune tensorboard /PATH/TO/TensorBoard_logdir --project USER_NAME/PROJECT_NAME

.. note:: That's it! You can now browse and collaborate on your TensorBoard runs in Neptune.

What next?
----------
#. Go to project `documentation <https://neptune-mlflow.readthedocs.io/en/latest>`_ to learn more.
#. Check our `blog <https://medium.com/neptune-ml>`_, where Neptune team publishes posts about variety of data science-related topics.

Neptune Contrib
===============
`Neptune-contrib <https://github.com/neptune-ml/neptune-contrib>`_ is an open source project curated by Neptune team. It is collection of framework integrations, functions and other productivity helpers that makes your work with Neptune faster and more efficient.

Resources
---------
* Project on GitHub: `neptune-contrib <https://github.com/neptune-ml/neptune-contrib>`_
* Project `docs <https://neptune-contrib.readthedocs.io>`_

Example tools:

* `LightGBM training monitor <https://neptune-contrib.readthedocs.io/examples/monitor_lgbm.html>`_
* `fast.ai training monitor <https://neptune-contrib.readthedocs.io/examples/monitor_fastai.html>`_
* Automatic `project progress visualizations <https://neptune-contrib.readthedocs.io/examples/project_progress.html>`_
* Hyper-parameter `comparison routines <https://neptune-contrib.readthedocs.io/examples/explore_hyperparams_skopt.html>`_


Quick-start
-----------
**Installation**

.. code-block:: bash

    pip install neptune-contrib

What next?
----------
#. It is highly recommended to skim through neptune-contrib `docs <https://neptune-contrib.readthedocs.io>`_. It contains number of examples and tutorials that show you how to use it.
#. Check our `blog <https://medium.com/neptune-ml>`_, where Neptune team publishes posts about variety of data science-related topics.
