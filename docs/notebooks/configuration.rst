Configuration
=============
Simple configuration wizard lets you to setup your work with Neptune.

**Step 1**

Click **Configure** button (if you cannot see it, just hit **F5**).

.. image:: ../_static/images/notebooks/buttons_01_1.png
   :target: ../_static/images/notebooks/buttons_01_1.png
   :alt: image

**Step 2**

You see **Configure connection to Neptune**. Paste API Token first, then select project:

* *API Token* - take one from your account. :ref:`Check this <token-location>`, if you do not know where *API Token* is located. This field is validated automatically.
* *Select project* - pick one from the drop down list.
* Click **Create notebook**.

.. warning:: Your *API Token* is private and unique. Never share it. It's like sharing password.

.. image:: ../_static/images/notebooks/checkpoint_01.png
   :target: ../_static/images/notebooks/checkpoint_01.png
   :alt: image

.. image:: ../_static/images/notebooks/checkpoint_02.png
   :target: ../_static/images/notebooks/checkpoint_02.png
   :alt: image

.. image:: ../_static/images/notebooks/checkpoint_03.png
   :target: ../_static/images/notebooks/checkpoint_03.png
   :alt: image

.. note:: Success! You just created new notebook with first checkpoint in the project you selected.

**Step 3**

* You can use *this link* at the top, to go directly to this notebook in Neptune.
* Click **Integrate** button to run environment variables export. It allows Neptune to checkpoint your work correctly. Moreover, you will be able to run Neptune experiments directly from notebook.

.. image:: ../_static/images/notebooks/integration_01.png
   :target: ../_static/images/notebooks/integration_01.png
   :alt: image

.. note:: Re-run **Integrate** each time you reset or restart your kernel.
