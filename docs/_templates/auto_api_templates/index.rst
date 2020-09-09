API Reference
=============

It contains API reference for the following libraries from the Neptune ecosystem:

- |neptune-client|: the main Python client
- |neptune-contrib|: library with community extensions
- |neptune-tensorboard|: Neptune integration with TensorBoard

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

    <a href="/api-reference/neptune/index.html" >neptune-client</a>

.. |neptune-contrib|  raw:: html

    <a href="/api-reference/neptunecontrib/index.html" >neptune-contrib</a>

.. |neptune-tensorboard| raw:: html

    <a href="/api-reference/neptune_tensorboard/index.html">neptune-tensorboard</a>
