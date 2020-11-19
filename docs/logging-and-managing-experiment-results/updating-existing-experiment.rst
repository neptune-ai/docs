.. _update-existing-experiment:

Updating existing experiment
============================
|video-update|

You can update experiments even after they finished running. This lets you add new data or visualizations to the previously closed experiment and makes multi-stage training convenient.

.. _update-existing-experiment-basics:

Why you may want to update an existing experiment?
--------------------------------------------------
Updating existing experiment can come in handy in several situations:

* You want to add metrics or visualizations to the closed experiment.
* You finished model training and closed the experiment earlier, but now you want to continue training from that moment. Actually, you can even make multiple iterations of the procedure: ``resume experiment -> log more data``. Have a look at the simple example below for details.

.. _update-existing-experiment-basics-simple-example:

How to update existing experiment?
----------------------------------
To update the experiment you need to get the project where this experiment is. Then you need to get the :class:`~neptune.experiments.Experiment` object of the experiment you want to update.

.. code-block:: python3

    import neptune

    # Get project
    project = neptune.init('my_workspace/my_project')

    # Get experiment object for appropriate experiment, here 'SHOW-2066'
    my_exp = project.get_experiments(id='SHOW-2066')[0]

Experiment with ``id='SHOW-2066'`` is now ready to be updated. Use ``my_exp`` to continue logging to it.

Note that with :meth:`~neptune.projects.Project.get_experiments` you can get experiments by ``id``, ``state``, ``owner``, ``tag`` and ``min_running_time``.

.. code-block:: python3

    from neptunecontrib.api.table import log_chart

    # Log metrics, images, text
    my_exp.log_metric(...)
    my_exp.log_image(...)
    my_exp.log_text(...)

    # Append tag
    my_exp.append_tag('updated')

    # Log new chart
    log_chart('matplotlib-interactive', fig, my_exp)

Technique is the same as described in section about :ref:`logging by using experiment object <logging-advanced-using-experiment-object-explicitly>`.

.. note::

    You can retrieve an experiment and log more data to it multiple times.

Example below shows updated experiment with more data-points logged to the ``'mse'`` metric and ``'pretty-random-metric'`` added.

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

|example-update|

.. _update-existing-experiment-what-you-can-cannot:

What you can/cannot update
--------------------------
You can freely use all :class:`~neptune.experiments.Experiment` methods. These include methods for logging more data:

* :meth:`~neptune.experiments.Experiment.log_metric`
* :meth:`~neptune.experiments.Experiment.log_artifact`
* :meth:`~neptune.experiments.Experiment.log_image`
* :meth:`~neptune.experiments.Experiment.log_text`
* :meth:`~neptune.experiments.Experiment.set_property`
* :meth:`~neptune.experiments.Experiment.append_tag`

Moreover, you can use all logging methods from ``neptunecontrib``, that is:

* :meth:`~neptunecontrib.api.audio.log_audio`
* :meth:`~neptunecontrib.api.chart.log_chart`
* :meth:`~neptunecontrib.api.video.log_video`
* :meth:`~neptunecontrib.api.table.log_table`
* :meth:`~neptunecontrib.api.html.log_html`
* :meth:`~neptunecontrib.api.explainers.log_explainer`

.. note::

    Learn more about :ref:`logging options <what-you-can-log>` to see why and how to use each method.

However, updating experiment comes with some limitations. Specifically:

* you cannot update |parameters| and |source-code|, but you can upload sources as artifact, using :meth:`~neptune.experiments.Experiment.log_artifact`.
* |hardware-consumption| for the update will not be tracked.
* ``stdout`` and ``stderr`` are not logged during update.
* experiment status (failed/succeeded/aborted) will not be updated.

.. _update-existing-experiment-step-by-step:


.. External links

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

.. Videos

.. |video-update| raw:: html

    <div style="position: relative; padding-bottom: 56.872037914691944%; height: 0;"><iframe src="https://www.loom.com/embed/d2bb1e74c74a4892a68b0bc9dc0a0f11" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>