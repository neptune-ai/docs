TensorBoard
===========

.. image:: ../_static/images/tensorboard/tensorboard_neptuneml.png
   :target: ../_static/images/tensorboard/tensorboard_neptuneml.png
   :alt: organize TensorBoard experiments in Neptune

|neptune-tensorboard| is an open source project curated by Neptune team, that integrates |tensorboard| with Neptune to let you get the best of both worlds.

With |neptune-tensorboard| you can have your TensorBoard visualizations hosted in Neptune.

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

You can now organize your TensorBoard experiments:

.. image:: ../_static/images/tensorboard/tensorboard_1.png
   :target: ../_static/images/tensorboard/tensorboard_1.png
   :alt: organize TensorBoard experiments in Neptune

and compare your TensorBoard runs,

.. image:: ../_static/images/tensorboard/tensorboard_2.png
   :target: ../_static/images/tensorboard/tensorboard_2.png
   :alt: compare TensorBoard runs in Neptune

and share your work with others by sending an |experiment-link|.

.. toctree::
   :maxdepth: 1
   :caption: Examples

   Sync and compare TensorBoard runs <tensorboard/tensorboard_compare.rst>
   Integrate TensorBoard logging with Neptune <tensorboard/tensorboard_integrate.rst>

Contribute
----------
Neptune-tensorboard is an open source project hosted on |GitHub|.
If you find yourself in any trouble drop an issue or talk to us directly on the |support-chat|.

Resources
---------
* Project on GitHub: |neptune-tensorboard|
* Example project in Neptune: |tensorboard-integration|.

.. External links

.. |GitHub| raw:: html

    <a href="https://github.com/neptune-ml/neptune-tensorboard" target="_blank">GitHub</a>


.. |support-chat| raw:: html

    <a href="https://spectrum.chat/neptune-community" target="_blank">support chat</a>


.. |neptune-tensorboard| raw:: html

    <a href="https://github.com/neptune-ml/neptune-tensorboard" target="_blank">Neptune-TensorBoard</a>


.. |tensorboard| raw:: html

    <a href="https://www.tensorflow.org/guide/summaries_and_tensorboard" target="_blank">TensorBoard</a>


.. |tensorboard-integration| raw:: html

    <a href="https://ui.neptune.ml/jakub-czakon/tensorboard-integration/experiments" target="_blank">TensorBoard project</a>


.. |experiment-link| raw:: html

    <a href="https://ui.neptune.ml/jakub-czakon/tensorboard-integration/compare?shortId=%5B%22TEN-41%22%2C%22TEN-40%22%2C%22TEN-39%22%2C%22TEN-38%22%2C%22TEN-37%22%2C%22TEN-36%22%2C%22TEN-35%22%2C%22TEN-34%22%2C%22TEN-33%22%2C%22TEN-32%22%5D" target="_blank">experiment link</a>
