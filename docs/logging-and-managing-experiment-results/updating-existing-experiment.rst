.. _update-existing-experiment:

Updating existing experiment
============================
[loom-placeholder]

You can update experiments even after they finished running. This let you update experiment with new data or visualizations even after closing experiment and makes multi-stage training convenient.

.. _update-existing-experiment-basics:

Basics of updating
------------------
Why updating existing experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Updating existing experiment can be handy, in several situations:

* You want to enrich closed experiment with more metrics or visualizations.
* You finished model training and closed the experiment earlier, but now you want to continue training from that moment. Actually, you can even make multiple iterations of the procedure: ``resume experiment -> log more data``. Have a look at the :ref:`simple example <update-existing-experiment-basics-simple-example>` below for details.

.. _update-existing-experiment-basics-simple-example:

Simple example
^^^^^^^^^^^^^^
In this example you will see how to upload more data to the existing experiment that was previously closed. Example result below, shows updated experiment with more data-points logged to the ``'mse'`` metric and ``'pretty-random-metric'`` added.

+--------------------------------------------------------------------------------------------------------------------+
| .. image:: ../_static/images/logging-and-managing-experiment-results/updating-experiment/update-charts-before.png  |
|    :target: ../_static/images/logging-and-managing-experiment-results/updating-experiment/update-charts-before.png |
|    :alt: Charts in original experiment                                                                             |
+====================================================================================================================+
| Charts in original experiment                                                                                      |
+--------------------------------------------------------------------------------------------------------------------+

+-------------------------------------------------------------------------------------------------------------------+
| .. image:: ../_static/images/logging-and-managing-experiment-results/updating-experiment/update-charts-after.png  |
|    :target: ../_static/images/logging-and-managing-experiment-results/updating-experiment/update-charts-after.png |
|    :alt: Charts in updated experiment                                                                             |
+===================================================================================================================+
| Charts in updated experiment                                                                                      |
+-------------------------------------------------------------------------------------------------------------------+


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

Example Code
""""""""""""
Experiment with ``id='SHOW-2066'`` was recorded then updated: |original-exp|. All the sources are logged:

* |original| - in the "Source code" section.
* |update| - logged as file and rendered nicely in the "Artifacts" section.

|example-update|

.. _update-existing-experiment-what-you-can-cannot:

What you can/cannot update
--------------------------
You can freely use all :class:`~neptune.experiments.Experiment` methods for logging more data:

* :meth:`~neptune.experiments.Experiment.log_metric`
* :meth:`~neptune.experiments.Experiment.log_artifact`
* :meth:`~neptune.experiments.Experiment.log_image`
* :meth:`~neptune.experiments.Experiment.log_text`

All other methods like :meth:`~neptune.experiments.Experiment.set_property`, :meth:`~neptune.experiments.Experiment.append_tag` or :meth:`~neptune.experiments.Experiment.download_artifacts` will work just fine.

However, updating experiment come with some limitation, notably:

* you cannot update |parameters| and |source-code|, but you can upload sources as artifact, using :meth:`~neptune.experiments.Experiment.log_artifact`.
* |hardware-consumption| for the update will not be tracked.
* ``stdout`` and ``stderr`` are not logged during update.
* experiment status (failed/succeeded/aborted) will not be updated.

.. _update-existing-experiment-step-by-step:


.. External links

.. |original| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-2066/source-code?path=.&file=update-experiment-1.py" target="_blank">original experiment sources</a>

.. |update| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-2066/artifacts?file=update-experiment-2.py" target="_blank">update sources</a>

.. |original-exp| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-2066/charts" target="_blank">here it is</a>

.. |parameters| raw:: html

    <a href="https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-325/parameters" target="_blank">parameters</a>

.. |hardware-consumption| raw:: html

    <a href="https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-325/monitoring" target="_blank">hardware consumption</a>

.. |source-code| raw:: html

    <a href="https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-325/source-code" target="_blank">source code</a>








.. Buttons

.. |example-update| raw:: html

    <div class="see-in-neptune">
        <button><a target="_blank"
                   href="https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-2066/charts">
                <img width="50" height="50" style="margin-right:10px"
                     src="https://gist.githubusercontent.com/kamil-kaczmarek/7ac1e54c3b28a38346c4217dd08a7850/raw/8880e99a434cd91613aefb315ff5904ec0516a20/neptune-ai-blue-vertical.png">See example in Neptune</a>
        </button>
    </div>
