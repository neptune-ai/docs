TensorBoard
===========
|neptune-tensorboard| is an open source project curated by Neptune team, that integrates |tensorboard| with Neptune to let you get the best of both worlds.

With |neptune-tensorboard| you can have your TensorBoard visualizations hosted in Neptune.

Resources
---------
* Project on GitHub: |neptune-tensorboard|
* Documentation: `TensorBoard integration with Neptune <https://neptune-tensorboard.readthedocs.io/en/latest/>`_
* Example project in Neptune: |tensorboard-integration|.

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

.. External links

.. |neptune-tensorboard| raw:: html

    <a href="https://github.com/neptune-ml/neptune-tensorboard" target="_blank">Neptune-TensorBoard</a>


.. |tensorboard| raw:: html

    <a href="https://www.tensorflow.org/guide/summaries_and_tensorboard" target="_blank">TensorBoard</a>


.. |tensorboard-integration| raw:: html

    <a href="https://ui.neptune.ml/jakub-czakon/tensorboard-integration/experiments" target="_blank">TensorBoard project</a>
