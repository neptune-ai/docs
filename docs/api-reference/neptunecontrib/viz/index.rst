:mod:`neptunecontrib.viz`
=========================

.. py:module:: neptunecontrib.viz


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   experiments/index.rst
   parallel_coordinates_plot/index.rst
   projects/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   neptunecontrib.viz.channel_curve_compare
   neptunecontrib.viz.make_parallel_coordinates_plot
   neptunecontrib.viz.project_progress


.. function:: channel_curve_compare(experiment_df, width=800, heights=(50, 400), line_size=5, legend_mark_size=100)

   Creates an interactive curve comparison chart for a list of experiments.

   It lets you tick or untick experiments that you want to compare by clicking on the legend (shift+click for multi),
   you can select the x range which you want to investigate by selecting it on the top chart and you
   get shown the actual values on mousehover.

   The chart is build on top of the Altair which in turn is build on top of Vega-Lite and Vega.
   That means you can use the objects produces by this script (converting it first to json by .to_json() method)
   in your html webpage without any problem.

   :param experiment_df: Dataframe containing ['id','x','CHANNEL_NAME'].
                         It can be obtained from a list of experiments by using the
                         `neptunelib.api.concat_experiments_on_channel` function. If the len of the dataframe exceeds 5000 it will
                         cause the MaxRowsError. Read the Note to learn why and how to disable it.
   :type experiment_df: 'pandas.DataFrame'
   :param width: width of the chart. Default is 800.
   :type width: int
   :param heights: heights of the subcharts. The first value controls the top chart, the second
                   controls the bottom chart. Default is (50,400).
   :type heights: tuple
   :param line_size: size of the lines. Default is 5.
   :type line_size: int
   :param legend_mark_size: size of the marks in legend. Default is 100.
   :type legend_mark_size: int

   :returns: Altair chart object which will be automatically rendered in the notebook. You can
             also run the `.to_json()` method on it to convert it to the Vega-Lite json format.
   :rtype: `altair.Chart`

   .. rubric:: Examples

   Instantiate a session::

       from neptunelib.api.session import Session
       session = Session()

   Fetch a project and a list of experiments::

       project = session.get_projects('neptune-ai')['neptune-ai/Salt-Detection']
       experiments = project.get_experiments(state=['aborted'], owner=['neyo'], min_running_time=100000)

   Construct a channel value dataframe::

       from neptunelib.api.utils import concat_experiments_on_channel
       compare_df = concat_experiments_on_channel(experiments,'unet_0 epoch_val iout loss')

   Plot interactive chart in notebook::

       from neptunelib.viz.experiments import channel_curve_compare
       channel_curve_compare(compare_df)

   .. note::

      Because Vega-Lite visualizations keep all the chart data in the HTML the visualizations can consume huge
      amounts of memory if not handled properly. That is why, by default the hard limit of 5000 rows is set to
      the len of dataframe. That being said, you can disable it by adding the following line in the notebook or code::
      
          import altair as alt
          alt.data_transformers.enable('default', max_rows=None)


.. function:: make_parallel_coordinates_plot(html_file_path=None, metrics=False, text_logs=False, params=True, properties=False, experiment_id=None, state=None, owner=None, tag=None, min_running_time=None)

   Visualize experiments on the parallel coordinates plot.

   This function, when executed in Notebook, displays interactive parallel coordinates plot in the cell's output.
   Another option is to save visualization to the standalone html file.
   You can also inspect the lineage of experiments.

   **See** `example <https://neptune-contrib.readthedocs.io/examples/hiplot_visualizations.html>`_
   **for the full use case.**

   Axes are ordered as follows: first axis is neptune ``experiment id``,
   second is experiment ``owner``,
   then ``params`` and ``properties`` in alphabetical order.
   Finally, ``metrics`` on the right side (alphabetical order as well).

   This visualization it built using `HiPlot <https://facebookresearch.github.io/hiplot/index.html>`_.
   It is a library published by the Facebook AI group.
   Learn more about the `parallel coordinates plot <https://en.wikipedia.org/wiki/Parallel_coordinates>`_.

   .. tip::

      Use ``metrics``, ``params`` and ``properties`` arguments to select what data you want to see as axes.
      
      Use ``experiment_id``, ``state``, ``owner``, ``tag``, ``min_running_time`` arguments to filter experiments
      included in a plot. Only experiments matching all the criteria will be returned.

   .. note:: Make sure you have your project set: ``neptune.init('USERNAME/example-project')``

   :param html_file_path: | Saves visualization as a standalone html file. No external dependencies needed.
   :type html_file_path: :obj:`str`, optional, default is ``None``
   :param metrics: | Metrics to display on the plot (as axes).
                   | If `True`, then display all metrics.
                   | If `False`, then exclude all metrics.
   :type metrics: :obj:`bool` or :obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``False``
   :param text_logs: | Text logs to display on the plot (as axes).
                     | If `True`, then display all text logs.
                     | If `False`, then exclude all text logs.
   :type text_logs: :obj:`bool` or :obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``False``
   :param params: | Parameters to display on the plot (as axes).
                  | If `True`, then display all parameters.
                  | If `False`, then exclude all parameters.
   :type params: :obj:`bool` or :obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``True``
   :param properties: | Properties to display on the plot (as axes).
                      | If `True`, then display all properties.
                      | If `False`, then exclude all properties.
   :type properties: :obj:`bool` or :obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``False``
   :param experiment_id: | An experiment id like ``'SAN-1'`` or list of ids like ``['SAN-1', 'SAN-2']``.
                         | Matching any element of the list is sufficient to pass criterion.
   :type experiment_id: :obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``None``
   :param state: | An experiment state like ``'succeeded'`` or list of states like ``['succeeded', 'running']``.
                 | Possible values: ``'running'``, ``'succeeded'``, ``'failed'``, ``'aborted'``.
                 | Matching any element of the list is sufficient to pass criterion.
   :type state: :obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``None``
   :param owner:
                 | *Username* of the experiment owner (User who created experiment is an owner) like ``'josh'``
                   or list of owners like ``['frederic', 'josh']``.
                 | Matching any element of the list is sufficient to pass criterion.
   :type owner: :obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``None``
   :param tag: | An experiment tag like ``'lightGBM'`` or list of tags like ``['pytorch', 'cycleLR']``.
               | Only experiments that have all specified tags will match this criterion.
   :type tag: :obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``None``
   :param min_running_time: Minimum running time of an experiment in seconds, like ``2000``.
   :type min_running_time: :obj:`int`, optional, default is ``None``

   :returns: :obj:`ExperimentDisplayed`, object that can be used to get a ``list`` of ``Datapoint`` objects,
             like this: ``ExperimentDisplayed.get_selected()``. This is only implemented for Jupyter notebook. Check
             `HiPlot docs
             <https://facebookresearch.github.io/hiplot/py_reference.html?highlight=display#hiplot.Experiment.display>`_.

   .. rubric:: Examples

   .. code:: python3

       # Make sure you have your project set:
       neptune.init('USERNAME/example-project')

       # (example 1) visualization for all experiments in project
       make_parallel_coordinates_plot()

       # (example 2) visualization for experiment with tag 'optuna' and saving to html file.
       make_parallel_coordinates_plot(html_file_path='visualizations.html', tag='optuna')

       # (example 3) visualization with all params, two metrics for experiment with tag 'optuna'
       make_parallel_coordinates_plot(tag='optuna', metrics=['epoch_accuracy', 'eval_accuracy'])

       # (example 4) visualization with all params and two metrics. All experiments created by john.
       make_parallel_coordinates_plot(metrics=['epoch_accuracy', 'eval_accuracy'], owner='john')


.. function:: project_progress(progress_df, width=800, heights=(50, 400), line_size=5, text_size=15, opacity=0.3)

   Creates an interactive project progress exploration chart.

   It lets you choose the resources you want to see ('experiment_count_day' or 'running_time_day'), you
   can see the metric/id/tags for every experiment on mouseover, you can select the x range which you want to
   investigate by selecting it on the top chart and you get shown the actual values on mousehover.

   The chart is build on top of the Altair which in turn is build on top of Vega-Lite and Vega.
   That means you can use the objects produces by this script (converting it first to json by .to_json() method)
   in your html webpage without any problem.

   :param progress_df: Dataframe containing ['id', 'metric', 'metric_best', 'running_time',
                       'running_time_day', 'experiment_count_day', 'owner', 'tags', 'timestamp', 'timestamp_day'].
                       It can be obtained from a list of experiments by using the
                       `neptunecontrib.api.extract_project_progress_info` function.
                       If the len of the dataframe exceeds 5000 it will cause the MaxRowsError.
                       Read the Note to learn why and how to disable it.
   :type progress_df: 'pandas.DataFrame'
   :param width: width of the chart. Default is 800.
   :type width: int
   :param heights: heights of the subcharts. The first value controls the top chart, the second
                   controls the bottom chart. Default is (50,400).
   :type heights: tuple
   :param line_size: size of the lines. Default is 5.
   :type line_size: int
   :param text_size: size of the text containing metric/id/tags in the middle.
   :type text_size: int
   :param opacity: opacity of the resource bars in the background. Default is 0.3.
   :type opacity: float

   :returns: Altair chart object which will be automatically rendered in the notebook. You can
             also run the `.to_json()` method on it to convert it to the Vega-Lite json format.
   :rtype: `altair.Chart`

   .. rubric:: Examples

   Instantiate a session::

       from neptunelib.api.session import Session
       session = Session()

   Fetch a project and the experiment view of that project::

       project = session.get_projects('neptune-ai')['neptune-ai/Salt-Detection']
       leaderboard = project.get_leaderboard()

   Create a progress info dataframe::

       from neptunecontrib.api.utils import extract_project_progress_info
       progress_df = extract_project_progress_info(leadearboard,
                                                   metric_colname='channel_IOUT',
                                                   time_colname='finished')

   Plot interactive chart in notebook::

       from neptunecontrib.viz.projects import project_progress
       project_progress(progress_df)

   .. note::

      Because Vega-Lite visualizations keep all the chart data in the HTML the visualizations can consume huge
      amounts of memory if not handled properly. That is why, by default the hard limit of 5000 rows is set to
      the len of dataframe. That being said, you can disable it by adding the following line in the notebook or code::
      
          import altair as alt
          alt.data_transformers.enable('default', max_rows=None)


