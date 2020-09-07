Updating existing experiment
============================

You can update experiments even after they finished running.
To do that you need to.

Step 1: Fetch Neptune project
-----------------------------

``neptune.init()`` method returns a project object so you can access it by running:

.. code:: python

    import neptune

    project = neptune.init('shared/onboarding')

The project contains all the experiments that are also objects that you can fetch.

Step 2: Fetch Experiment
------------------------

Use ``project.get_experiment()`` method and specify your experiment ID.
For example:

.. code:: python

    experiment = project.get_experiments(id='ON-238')[0]

``project.get_experiment()`` returns a list of experiments. In this case we just have one but still need to access it.

Step 3: Update experiment
-------------------------

You can use **all** the normal ``experiment`` logging methods like:

- ``.log_metric``
- ``.log_image``
- ``.log_artifact``

For example I'll update the experiment with a new metric 'external_test_auc':

.. code:: python

    experiment.log_metric('external_test_auc', 0.82)

And you can go to the UI and see your updated experiment.

.. warning::

    Some things are not logged when you update the existing experiment.

    Those are:

    - hardware consumption, stderr, stdout logs in the ``Monitoring`` section
    - code in the ``Source code`` section



