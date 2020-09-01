:mod:`neptunecontrib.api.chart`
===============================

.. py:module:: neptunecontrib.api.chart


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   neptunecontrib.api.chart.log_chart


.. function:: log_chart(name, chart, experiment=None)

   Logs charts from matplotlib, plotly, bokeh, and altair to neptune.

   Plotly, Bokeh, and Altair charts are converted to interactive HTML objects and then uploaded to Neptune
   as an artifact with path charts/{name}.html.

   Matplotlib figures are converted optionally. If plotly is installed, matplotlib figures are converted
   to plotly figures and then converted to interactive HTML and uploaded to Neptune as an artifact with
   path charts/{name}.html. If plotly is not installed, matplotlib figures are converted to PNG images
   and uploaded to Neptune as an artifact with path charts/{name}.png

   :param name: | Name of the chart (without extension) that will be used as a part of artifact's destination.
   :type name: :obj:`str`
   :param chart:
                 | Figure from `matplotlib` or `plotly`. If you want to use global figure from `matplotlib`, you
                   can also pass reference to `matplotlib.pyplot` module.
   :type chart: :obj:`matplotlib` or :obj:`plotly` Figure
   :param experiment:
                      | For advanced users only. Pass Neptune
                        `Experiment <https://docs.neptune.ai/neptune-client/docs/experiment.html#neptune.experiments.Experiment>`_
                        object if you want to control to which experiment data is logged.
                      | If ``None``, log to currently active, and most recent experiment.
   :type experiment: :obj:`neptune.experiments.Experiment`, optional, default is ``None``

   .. rubric:: Examples

   Start an experiment::

       import neptune

       neptune.init(api_token='ANONYMOUS',
                    project_qualified_name='shared/showroom')
       neptune.create_experiment(name='experiment_with_charts')

   Create matplotlib figure and log it to Neptune::

       import matplotlib.pyplot as plt

       fig = plt.figure()
       x = [21,22,23,4,5,6,77,8,9,10,31,32,33,34,35,36,37,18,49,50,100]
       plt.hist(x, bins=5)
       plt.show()

       from neptunecontrib.api import log_chart

       log_chart('matplotlib_figure', fig)

   Create Plotly chart and log it to Neptune::

       import plotly.express as px

       df = px.data.tips()
       fig = px.histogram(df, x="total_bill", y="tip", color="sex", marginal="rug",
                          hover_data=df.columns)
       fig.show()

       from neptunecontrib.api import log_chart

       log_chart('plotly_figure', fig)

   Create Altair chart and log it to Neptune::

       import altair as alt
       from vega_datasets import data

       source = data.cars()

       chart = alt.Chart(source).mark_circle(size=60).encode(
                       x='Horsepower',
                       y='Miles_per_Gallon',
                       color='Origin',
                       tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
       ).interactive()

       from neptunecontrib.api import log_chart

       log_chart('altair_chart', chart)

   Create Bokeh figure and log it to Neptune::

       from bokeh.plotting import figure

       p = figure(plot_width=400, plot_height=400)

       # add a circle renderer with a size, color, and alpha
       p.circle([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=20, color="navy", alpha=0.5)

       from neptunecontrib.api import log_chart

       log_chart('bokeh_figure', p)

   .. note::

      Check out how the logged charts look in Neptune:
      `example experiment
      <https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-973/artifacts?path=charts%2F&file=bokeh_figure.html>`_


