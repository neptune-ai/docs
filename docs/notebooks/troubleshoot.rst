Troubleshoot
============
Typical problems with notebooks, together with their solutions are listed here.

.. contents::
    :local:
    :depth: 1
    :backlinks: top

I can't see 'configure' button. What to do?
-------------------------------------------
Make sure to install notebook extension:

.. code-block:: bash

   pip install neptune-notebooks

then enable extension for your Jupyter:

.. code-block:: bash

   jupyter nbextension enable --py neptune-notebooks

Don't forget to install Neptune client: 

.. code-block:: bash

   pip install neptune-client

How to enable notebook extension in my jupyter?
-----------------------------------------------
Enable extension for your jupyter:

.. code-block:: bash

   jupyter nbextension enable --py neptune-notebooks

I do not know where my notebook was uploaded. How to check it?
--------------------------------------------------------------
#. Click on the **n** button in your jupyter menu.
#. Click on **Checkpoint**.
#. Bottom drop-down is your current project.

.. _token-location:

Where is *NEPTUNE_API_TOKEN*?
-----------------------------
#. Log in to `neptune <https://neptune.ai/login>`_.
#. Click on your avatar (top-right part of the screen) and select **Get API Token**

.. image:: ../_static/images/notebooks/token.png
   :target: ../_static/images/notebooks/token.png
   :alt: image

My integration does not work, but it worked well previously. What do to?
------------------------------------------------------------------------
Most likely, you restarted kernel. Here is a solution:

#. Go to configuration (**n** button).
#. Click **Integrate**.

.. image:: ../_static/images/notebooks/integration_01.png
   :target: ../_static/images/notebooks/integration_01.png
   :alt: image
