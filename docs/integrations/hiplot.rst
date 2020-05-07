Neptune-HiPlot Integration
==========================

This integration lets you analyze multiple experiments in Neptune using |HiPlot| visualization. HiPlot is a lightweight interactive visualization tool published by
the Facebook AI group.

.. image:: ../_static/images/hiplot/example_hiplot_0.png
   :target: ../_static/images/hiplot/example_hiplot_0.png
   :alt: parallel plot header

Parallel coordinates plot is a powerful tool that allows AI researchers to analyze correlations
and patterns between experiment metrics, parameters and properties.

Parallel plots are especially useful when inspecting hyperparameter optimization jobs
that usually consists of hundreds of experiments. Neptune allows you to very easily generate such plots in a Jupyter Notebook
or Python script.

Requirements
------------
This feature is implemented as a part of |neptune-contrib|.
Make sure that you have all dependencies installed:

* neptune-client
* neptune-contrib[viz]
* hiplot

Use this command to install them:

.. code-block:: bash

    pip install neptune-client neptune-contrib[viz] hiplot

Generate parallel coordinates plot
----------------------------------

.. note::
    Make sure you have your project set: ``neptune.init('USERNAME/example-project')``

.. code-block:: python3

    import neptune

    from neptunecontrib.viz.parallel_coordinates_plot import make_parallel_coordinates_plot

    neptune.init('USERNAME/example-project')

    make_parallel_coordinates_plot(html_file_path='my_visual.html',
                                   metrics= ['epoch_accuracy', 'epoch_loss', 'eval_accuracy', 'eval_loss'],
                                   params = ['activation', 'batch_size', 'dense_units', 'dropout', 'learning_rate', 'optimizer'],
                                   tag='optuna')

.. image:: ../_static/images/hiplot/example_hiplot_1.png
   :target: ../_static/images/hiplot/example_hiplot_1.png
   :alt: parallel plot overview

Customize visualization to your needs
-------------------------------------

Perform the following steps:

#. Set axes order.
#. Drop the unused axes.
#. Apply coloring to the axis.
#. Sort by clicking on the axis.
#. Select range in the axis and slide.

.. image:: ../_static/images/hiplot/example_hiplot_1.gif
   :target: ../_static/images/hiplot/example_hiplot_1.gif
   :alt: parallel plot customization options

Inspect experiments lineage
---------------------------

Perform the following steps:

#. Right-click on the axis name.
#. Use options 'Set as X axis' and 'Set as Y axis' (in the menu XY group at the bottom).

When both are selected, you will see the lineage plot below the parallel coordinates plot.

.. image:: ../_static/images/hiplot/example_hiplot_2.gif
    :target: ../_static/images/hiplot/example_hiplot_2.gif
    :alt: experiments lineage

Check example notebooks in Neptune
----------------------------------
#. |credit-default-prediction|
#. |example-project|

These notebooks are tracked in Neptune public projects. Feel free to play with the plots - they are interactive.

Learn more
----------
Check integration |documentation| for more details.

.. External links

.. |Neptune| raw:: html

    <a href="https://neptune.ai/" target="_blank">Neptune</a>

.. |HiPlot| raw:: html

    <a href="https://facebookresearch.github.io/hiplot/index.html" target="_blank">HiPlot</a>

.. |neptune-contrib| raw:: html

    <a href="https://github.com/neptune-ai/neptune-contrib" target="_blank">neptune-contrib</a>

.. |documentation| raw:: html

    <a href="https://neptune-contrib.readthedocs.io/user_guide/viz/parallel_coordinates_plot.html" target="_blank">documentation</a>

.. |example-project| raw:: html

    <a href="https://ui.neptune.ai/o/USERNAME/org/example-project/n/parallel-plot-cb5394cc-edce-41e3-9a25-7970865c66ad/59377976-6651-40ed-b3c3-eb0fa5aa79bc" target="_blank">example-project</a>

.. |credit-default-prediction| raw:: html

    <a href="https://ui.neptune.ai/neptune-ai/credit-default-prediction/n/parallel-plot-04e5c379-0837-42ff-a11c-a8861ca4a408/c486644a-a356-4317-b397-6cdae86b7575" target="_blank">credit-default-prediction</a>
