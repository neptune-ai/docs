Installation for Jupyer and JupyerLab
=====================================

Jupyer
------
Install `neptune-notebooks extension <https://github.com/neptune-ml/neptune-notebooks>`_:

.. code-block:: bash

   pip install neptune-notebooks

Then enable extension for your Jupyter:

.. code-block:: bash

   jupyter nbextension enable --py neptune-notebooks

Remember to install Neptune-client, if you did not do so already:

.. code-block:: bash

   pip install neptune-client

JupyerLab
---------
Install `neptune-notebooks <https://www.npmjs.com/package/neptune-notebooks>`_ in your JupyerLab. In terminal run:

.. code-block:: bash

    jupyter labextension install neptune-notebooks

Remember to install Neptune-client, if you did not do so already:

.. code-block:: bash

   pip install neptune-client
