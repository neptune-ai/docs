What is Neptune?
================

`Neptune <https://neptune.ml/>`_ is data science collaboration hub. With Neptune, teams can work together efficiently and keep all aspects of their workflow in a single place. Whether it is source code, model training curves or meeting notes, Neptune got you covered.

Neptune is lightweight
----------------------
Neptune team builds the platform with single design principle is mind: *lightweightness*. What it means?

* fast&easy new user onboarding (check yourself: :doc:`get-started <tutorials/tutorial-1>`).
* 20 minutes deployment: use SaaS or deploy on any cloud or on your hardware.
* Neptune fits in any workflow ranging from data exploration & analysis, decision science to machine learning and deep learning.
* Neptune support your techstack: Notebooks, Python scripts, R.
* Open Source extensions (use if you need)

Neptune's focus is put on three aspects of data science / machine learning project: :ref:`track <track>`, :ref:`organize <organize>` and :ref:`collaborate <collaborate>`.

.. _track:

Track
-----
Track all objects in the data science or machine learning project. It can be model training curves, visualizations, input data, calculated features and so on. Snipped below, presents example integration with Python code.

.. code-block:: python

   import neptune

   neptune.init('shared/onboarding')
   with neptune.create_experiment():
       neptune.append_tag('minimal-example')
       n = 117
       for i in range(1, n):
           neptune.send_metric('iteration', i)
           neptune.send_metric('loss', 1/i**0.5)
       neptune.set_property('n_iterations', n)

.. _organize:

Organize
--------
Organize structure of your project: meeting notes, reports, notebooks, experiment - everything is in one place.

.. _collaborate:

Collaborate
-----------
Collaborate in the team share, compare, comment and communicate your work.

Documentation contents
----------------------

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   Get started <tutorials/tutorial-1.rst>

.. toctree::
   :maxdepth: 1
   :caption: How-to guides

   how-to/track.rst
   how-to/organize.rst

.. toctree::
   :maxdepth: 1
   :caption: Supported languages

   Python API <python-api.rst>
   R support <r-support.rst>

.. toctree::
   :maxdepth: 1
   :caption: Notebooks in Neptune

   Introduction <notebooks/introduction.rst>
   Installation and configuration <notebooks/install-configure.rst>
   Troubleshoot <notebooks/troubleshoot.rst>

.. toctree::
   :maxdepth: 1
   :caption: Integrations

   Integrations <integrations.rst>
