.. _update-existing-experiment:

Updating existing experiment
============================
[loom-placeholder]

You can update experiments even after they finished running. This enables updating experiment with new data and visualizations even after closing experiment and makes multi-stage training convenient.

.. _update-existing-experiment-basics:

Basics of updating
------------------
Why updating existing experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Updating existing experiment can be handy, in several situations:

* You want to enrich closed experiment with more metrics or visualizations.
* You finished model training earlier and now want to continue training from that moment. You want to keep logging new metrics for that update. Actually, you can even make multiple iterations of the procedure: ``resume experiment -> log more data``. Have a look at the :ref:`simple case <update-existing-experiment-basics-simple-case>` for details.

.. _update-existing-experiment-basics-simple-case:

Update existing experiment - simple case
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To update existing experiment with new data, you need to perform just two steps.

Get experiment to update
""""""""""""""""""""""""
Retrieve the :class:`~neptune.experiments.Experiment` object of the experiment you want to update.

.. code-block:: python3

    import neptune

    # Get project
    project = neptune.init('my_workspace/my_project')

    # Get experiment object for appropriate experiment, here 'SHOW-2066'
    my_exp = project.get_experiments(id='SHOW-2066')[0]

Few explanations:

* ``project`` is :class:`~neptune.projects.Project` instance, that we use to retrieve desired experiment.
* Use :meth:`~neptune.projects.Project.get_experiments` to get desired experiment. Fetching by ``id`` is just one option.
* :meth:`~neptune.projects.Project.get_experiments` returns list of :class:`~neptune.experiments.Experiment` objects. In this case list has just single element - experiment with ``id='SHOW-2066'``. We want to get this single element directly, hence ``[0]`` at the end of the line.

Experiment with ``id='SHOW-2066'`` is now ready to be updated. Use ``my_exp`` to continue logging to it.

Log more data
"""""""""""""
With ``my_exp`` at hand, you can use it to continue logging to the experiment with ``id='SHOW-2066'``.

.. code-block:: python3

    from neptunecontrib.api.table import log_chart

    my_exp.log_metric(...)
    my_exp.log_image(...)
    my_exp.log_text(...)

    my_exp.append_tag('updated')

    log_chart('matplotlib-interactive', fig, my_exp)

Really nothing special here. Technique is the same as described in section about :ref:`logging by using experiment object <logging-advanced-using-experiment-object-explicitly>`.

Example
"""""""
Experiment with ``id='SHOW-2066'`` was recorded then updated: |original-exp|. All th sources are logged:

* |original| - in the "Source code" section.
* |update| - logged as file and rendered nicely in the "Artifacts" section.

|example-update|

.. _update-existing-experiment-what-you-can-cannot:

What you can/cannot update
--------------------------


.. _update-existing-experiment-step-by-step:

How to update step by step
--------------------------

Troubleshooting
---------------


























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

.. External links

.. |original| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-2066/source-code?path=.&file=update-experiment-1.py" target="_blank">original experiment sources</a>

.. |update| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-2066/artifacts?file=update-experiment-2.py" target="_blank">update sources</a>

.. |original-exp| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-2066/charts" target="_blank">example</a>

.. Buttons

.. |example-update| raw:: html

    <div class="see-in-neptune">
        <button><a target="_blank"
                   href="https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-2066/charts">
                <img width="50" height="50" style="margin-right:10px"
                     src="https://gist.githubusercontent.com/kamil-kaczmarek/7ac1e54c3b28a38346c4217dd08a7850/raw/8880e99a434cd91613aefb315ff5904ec0516a20/neptune-ai-blue-vertical.png">See example in Neptune</a>
        </button>
    </div>
