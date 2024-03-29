.. _integrations-plotly:

Neptune-plotly Integration
==========================

.. warning::
    This is the documentation of the legacy client which is under the maintenance support only.
    No new updates will be made to this documentation and legacy client.

    It is **highly recommended** to go the to `new documentation <https://docs.neptune.ai/>`_ for the latest documentation and full support of the new, improved Neptune.

    `Read new documentation <https://docs.neptune.ai/>`_

This integration lets you log interactive charts generated in |plotly|, like confusion matrix or distribution, in Neptune.


.. image:: ../_static/images/integrations/plotly.png
   :target: ../_static/images/integrations/plotly.png
   :alt: plotly neptune.ai integration


Follow these steps:


0. Create an experiment:

   .. code-block::

        import neptune

        neptune.init(api_token='ANONYMOUS',project_qualified_name='shared/showroom')
        neptune.create_experiment()

1. Create and log plotly figures into Neptune:

   .. code-block::

        import plotly.express as px

        df = px.data.tips()
        plotly_fig = px.histogram(df, x="total_bill", y="tip", color="sex", marginal="rug",
                           hover_data=df.columns)

   .. code-block::

        from neptunecontrib.api import log_chart

        log_chart(name='plotly_figure', chart=plotly_fig)

2. Explore the results in the Neptune dashboard:

Check out |this experiment| in the app.

.. image:: ../_static/images/integrations/plotly.gif
   :target: ../_static/images/integrations/plotly.gif
   :alt: image

.. External Links

.. |plotly| raw:: html

    <a href="https://plotly.com/" target="_blank">plotly</a>

.. |this experiment| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-978/artifacts?path=charts%2F&file=plotly_figure.html" target="_blank">this experiment</a>
