API Reference
=============

It contains API reference for all libraries in the Neptune ecosystem:

- |neptune-client|: the main Python client
- |neptune-contrib|: library with community extensions
- |neptune-tensorboard|: Neptune integration with TensorBoard
- |neptune-mlflow|: Neptune integration with MLflow

Packages:
---------

.. toctree::
   :titlesonly:

   {% for page in pages %}
   {% if page.top_level_object and page.display %}
   {{ page.include_path }}
   {% endif %}
   {% endfor %}

.. |neptune-client| raw:: html

    <a href="https://github.com/neptune-ai/neptune-client" target="_blank">neptune-client</a>

.. |neptune-contrib|  raw:: html

    <a href="https://github.com/neptune-ai/neptune-contrib" target="_blank">neptune-client</a>

.. |neptune-tensorboard| raw:: html

    <a href="https://github.com/neptune-ai/neptune-tensorboard" target="_blank">neptune-client</a>

.. |neptune-mlflow|  raw:: html

    <a href="https://github.com/neptune-ai/neptune-mlflow" target="_blank">neptune-client</a>
