:mod:`neptunecontrib.viz.experiments`
=====================================

.. py:module:: neptunecontrib.viz.experiments


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   neptunecontrib.viz.experiments.channel_curve_compare


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


