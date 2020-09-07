

:mod:`neptunecontrib.monitoring.utils`
======================================

.. py:module:: neptunecontrib.monitoring.utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   neptunecontrib.monitoring.utils.axes2fig
   neptunecontrib.monitoring.utils.send_figure
   neptunecontrib.monitoring.utils.pickle_and_send_artifact


.. function:: axes2fig(axes, fig=None)

   Converts ndarray of matplotlib object to matplotlib figure.

   Scikit-optimize plotting functions return ndarray of axes. This can be tricky
   to work with so you can use this function to convert it to the standard figure format.

   :param axes: Array of matplotlib axes objects.
   :type axes: `numpy.ndarray`
   :param fig: Matplotlib figure on which you may want to plot
               your axes. Default None.
   :type fig: 'matplotlib.figure.Figure'

   :returns: Matplotlib figure with axes objects as subplots.
   :rtype: 'matplotlib.figure.Figure'

   .. rubric:: Examples

   Assuming you have a `scipy.optimize.OptimizeResult` object you want to plot::

       from skopt.plots import plot_evaluations
       eval_plot = plot_evaluations(result, bins=20)
       >>> type(eval_plot)
           numpy.ndarray

       from neptunecontrib.viz.utils import axes2fig
       fig = axes2fig(eval_plot)
       >>> fig
           matplotlib.figure.Figure


.. function:: send_figure(fig, channel_name='figures', experiment=None)


.. function:: pickle_and_send_artifact(obj, filename, experiment=None)



.. External links

.. |Neptune| raw:: html

    <a href="/api-reference/neptune/index.html#functions" target="_blank">Neptune</a>

.. |Session| raw:: html

    <a href="/api-reference/neptune/sessions/index.html?highlight=neptune%20sessions%20session#neptune.sessions.Session" target="_blank">Session</a>

.. |Project| raw:: html

    <a href="/api-reference/neptune/projects/index.html#neptune.projects.Project" target="_blank">Project</a>

.. |Experiment| raw:: html

    <a href="/api-reference/neptune/experiments/index.html?highlight=neptune%20experiment#neptune.experiments.Experiment" target="_blank">Experiment</a>

.. |Notebook| raw:: html

    <a href="/api-reference/neptune/notebook/index.html?highlight=notebook#neptune.notebook.Notebook" target="_blank">Notebook</a>

.. |Git Info| raw:: html

    <a href="/api-reference/neptune/git_info/index.html#neptune.git_info.GitInfo" target="_blank">Git Info</a>