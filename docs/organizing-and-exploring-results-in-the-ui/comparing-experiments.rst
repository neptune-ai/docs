.. _guides-compare-experiments-ui:

Compare experiments
===================

|Youtube Video compare|

Neptune lets you compare up to 10 experiments using in-depth analysis in the specialized view.
Simply select experiments and click **Compare**:


   .. image:: ../_static/images/organizing-and-exploring-results-in-the-ui/experiment-dashboard/compare_experiments_select.png
      :target: ../_static/images/organizing-and-exploring-results-in-the-ui/experiment-dashboard/compare_experiments_select.png
      :alt: Compare experiments table

.. note::

    You can share your experiment comparison by sending a link. |Like this one|.

- **Overlaid charts**: In the comparison view, all metrics with the same name are placed on a single chart with one curve per experiment. The customizable legend lets you select additional metrics and/or parameters to display. When hovering with the mouse over a particular area, the values for the selected metrics are displayed below:

   .. image:: ../_static/images/organizing-and-exploring-results-in-the-ui/experiment-dashboard/charts_legend_mouseover.png
      :target: ../_static/images/organizing-and-exploring-results-in-the-ui/experiment-dashboard/charts_legend_mouseover.png
      :alt: Charts legend
      :width: 600

- **Interactive comparison table**: Below the charts, details of the experiments being compared are shown in table form.

    Each column represents one experiment and each row represents a single property and the data associated with it.

    This table has a few useful features:

    - You can tick "show diff only" to show only the rows where the metrics, parameters and other experiment properties are different.
    - You can tick "show cell changes" to highlight where the metrics (or other properties) went up or down
    - You can change the reference experiment (to which the "show cell changes" refers to) by pinning a different experiment.

   .. image:: ../_static/images/organizing-and-exploring-results-in-the-ui/experiment-dashboard/compare_experiments.png
      :target: ../_static/images/organizing-and-exploring-results-in-the-ui/experiment-dashboard/compare_experiments.png
      :alt: Compare experiments table
      :width: 600

.. External videos

.. |Youtube Video compare| raw:: html

    <iframe width="720" height="420" src="https://www.youtube.com/embed/DEBkjqsaMrc" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

.. |Like this one| raw:: html

    <a href="https://ui.neptune.ai/o/neptune-ai/org/credit-default-prediction/compare?shortId=%5B%22CRED-93%22%2C%22CRED-92%22%2C%22CRED-91%22%2C%22CRED-89%22%2C%22CRED-85%22%2C%22CRED-80%22%2C%22CRED-70%22%5D&viewId=a261e2d2-a558-468e-bf16-9fc9d0394abc" target="_blank">Like this one</a>

