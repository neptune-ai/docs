Working with the Neptune UI
===========================

Neptune is a browser-enabled app that lets you visualize and browse experiments.


   .. image:: /_static/images/core-concepts/experiment_general_view.png
      :target: /_static/images/core-concepts/experiment_general_view.png
      :alt: Experiments view

.. contents::
    :local:
    :depth: 1
    :backlinks: top

Supported Browsers
------------------

Neptune supports Chrome and Firefox, on all major operating systems.

Log in to the App
-----------------
1. In a web browser, navigate to ``ui.neptune.ai`` or, in the case of an on-prem deployment, to your Neptune instance address.
2. In the window that appears, click **Login** in the upper right corner.

3. In the Sign-in dialog that appears, either:

   - Click the Facebook, Google, or GitHub icon and sign in.

   - Type in your user name and password, then click **Sign in**.

   .. image:: /_static/images/core-concepts/login.png
      :target: /_static/images/core-concepts/login.png
      :alt: Login screen
      :width: 350

Experiments View
----------------


The **Experiments** tab  displays all the experiments in a specific Project in table form.

Filter experiments
^^^^^^^^^^^^^^^^^^
You can perform the simplest filtering by typing into the search fields:

   .. image:: /_static/images/core-concepts/search_fields.png
      :target: /_static/images/core-concepts/search_fields.png
      :alt: Experiment search


You can also use the `Neptune Query Language <nql.html>`_ to filter experiments for more advanced criteria.

Organize experiments
^^^^^^^^^^^^^^^^^^^^

There are several ways to organize your experiments:

- **Use tags**: You can create tag(s), which you assign to experiments. Later, you can quickly filter by these tags.

   .. image:: /_static/images/core-concepts/tag_chooser.png
      :target: /_static/images/core-concepts/tag_chooser.png
      :alt: Tag chooser
      :width: 250

- **Customize views**: You can create a custom view of the dashboard. For example, you can filter rows by parameter or metric values and select a subset of useful columns. Then save the view and quickly return to it by selecting it from the list of views.

   .. image:: /_static/images/core-concepts/view_list.png
      :target: /_static/images/core-concepts/view_list.png
      :alt: View list
      :width: 400

- **Choose columns**: You can configure several data types logged to Neptune so that they are displayed as columns in the dashboard. They are metrics, text logs, properties and parameters. However, **all** data can be seen in the **single** experiment view.

- **Group experiments**: You can group experiments by one or more column(s). The dashboard displays the selected columns, allowing you to make in-group and across-groups analysis of the experiments. Each group is represented by the first experiment that appears according to the sorting order. After opening it, each group shows at most 10 experiments - all experiments can be viewed by clicking **Show all**.

   .. image:: /_static/images/core-concepts/group_by.png
      :target: /_static/images/core-concepts/group_by.png
      :alt: Group columns


Compare experiments
^^^^^^^^^^^^^^^^^^^
Neptune lets you compare up to 10 experiments using in-depth analysis in the specialized view.
Simply select experiments and click **Compare**:


   .. image:: /_static/images/core-concepts/compare_experiments_select.png
      :target: /_static/images/core-concepts/compare_experiments_select.png
      :alt: Compare experiments table


- **Overlaid charts**: In the comparison view, all metrics with the same name are placed on a single chart with one curve per experiment. The customizable legend lets you select additional metrics and/or parameters to display. When hovering with the mouse over a particular area, the values for the selected metrics are displayed below:

   .. image:: /_static/images/core-concepts/charts_legend_mouseover.png
      :target: /_static/images/core-concepts/charts_legend_mouseover.png
      :alt: Charts legend
      :width: 600

- **Interactive comparison table**: Below the charts, details of the experiments being compared are shown in table form. Each column represents one experiment and each row represents a single property and the data associated with it.

   .. image:: /_static/images/core-concepts/compare_experiments.png
      :target: /_static/images/core-concepts/compare_experiments.png
      :alt: Compare experiments table
      :width: 600


Single Experiment View
----------------------
Click a line in the experiments table to see details of that experiment.


   .. image:: /_static/images/core-concepts/single_experiment.png
      :target: /_static/images/core-concepts/single_experiment.png
      :alt: Single experiment

Inside the experiment, there are tabs in the left sidebar. Each displays specific content that is logged to Neptune for the specific experiment. Each tab has a unique URL.

- **Charts**: All metrics (numeric type of data) are visualized as charts.
- **Logs**: Logs are data that can be logged or tracked to the experiment. There are multiple types: Numeric, text, and images.
- **Monitoring**: Displays information about utilization of the memory, CPU and GPU, as well as stdout and stderr.
- **Artifacts**: Displays files uploaded to the experiment.
- **Source code**: Displays sources uploaded to the experiment.
- **Parameters**: Displays parameters uploaded to the experiment during experiment creation.
- **Details**: Displays additional metadata information.

For more information about the experiment data that is logged, see `Logging to Neptune <../learn-about-neptune/experiment_tracking.html#logging-to-neptune>`_.

Notebooks
---------

Neptune lets you track Jupyter Notebooks.

For more information, see `Using Jupyter Notebooks in Neptune <../notebooks/introduction.html>`_.

Wiki
----

An interactive Wiki with collaborative editing features lets multiple users edit the Wiki pages and create reports with teammates in real time.

For more information, see `Working with the Project Wiki <../learn-about-neptune/collaborate.html#working-with-the-project-wiki>`_.
