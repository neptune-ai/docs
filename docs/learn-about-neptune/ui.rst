Working with the Neptune UI
===========================

The Neptune client is a browser-enabled app that lets you visualize and browse experiments.


   .. image:: /_static/images/core-concepts/experiment_general_view.png
      :target: /_static/images/core-concepts/experiment_general_view.png
      :alt: Experiments view


Supported browsers
------------------

Neptune supports Chrome and Firefox, on all major operating systems.

Log in to the app
-----------------
1. In a web browser, navigate to ``ui.neptune.ai`` or, in the case of an on-prem deployment, to your Neptune instance address.
2. Click **Login**.

.. image:: /_static/images/core-concepts/login.png
   :target: /_static/images/core-concepts/login.png
   :alt: Login screen
   :width: 350


3. Type in your credentials and click **Sign in**.

Experiments View
----------------


The **Experiments** tab  displays all the experiments in a specific Project in table form.

Filter Experiments
^^^^^^^^^^^^^^^^^^
You can perform the simplest filtering by typing into the search fields:

   .. image:: /_static/images/core-concepts/search_fields.png
      :target: /_static/images/core-concepts/search_fields.png
      :alt: Experiment search


You can also use the `Neptune Query Language <nql.html>`_ to filter experiments for more advanced criteria.

Organize Experiments
^^^^^^^^^^^^^^^^^^^^

There are several ways to organize your experiments:

- **Tags**: You can create tag(s), which you assign to experiments. Later, you can quickly filter by these tags.

   .. image:: /_static/images/core-concepts/tag_chooser.png
      :target: /_static/images/core-concepts/tag_chooser.png
      :alt: Tag chooser
      :width: 250

- **Custom views**: You can create a custom view of the dashboard, for example, by filtering selected rows, columns and columns order, sorting and width. Then you can save the view and quickly return to it by selecting the desired view from the list of views.

   .. image:: /_static/images/core-concepts/view_list.png
      :target: /_static/images/core-concepts/view_list.png
      :alt: View list
      :width: 400

- **Choosing columns**: Columns in the dashboard are configurable. All types of data logged to Neptune can be seen as columns in the dashboard, for example parameters, metrics or tags.

- **Grouping**: You can group experiments by one or more column(s). The dashboard displays the selected columns, allowing you to make in-group and across-groups analysis of the experiments.

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


- **Overlaid charts**: In the comparison view, all metrics with the same name are placed on a single chart with one curve per experiment. The rich and customizable legend lets you display additional information.

   .. image:: /_static/images/core-concepts/charts_legend.png
      :target: /_static/images/core-concepts/charts_legend.png
      :alt: Charts legend
      :width: 200

- **Interactive comparison table**: Below the charts, details of the experiments being compared are shown in table form. Each column represents one experiment and each row represents a single property and the data associated with it.

   .. image:: /_static/images/core-concepts/compare_experiments.png
      :target: /_static/images/core-concepts/compare_experiments.png
      :alt: Compare experiments table


Single experiment view
----------------------
Click a line in the experiments table to see details of that experiment.


   .. image:: /_static/images/core-concepts/single_experiment.png
      :target: /_static/images/core-concepts/single_experiment.png
      :alt: Single experiment

Inside the experiment, there are tabs in the left sidebar. Each displays specific content that is logged to Neptune for the specific experiment. Each tab has a unique URL.

- **Charts**: All metrics (numeric type of data) are visualized as charts. You can build your own subset of charts. Once created, they are available for all experiments.

- **Logs**: Logs are data that can be logged or tracked to the experiment. There are multiple types:

   - **Numeric**: Float or int type.
   - **Text**: String type.
   - **Image**: Images (image files, numpy array, matplotlib)

- **Monitoring**: Displays information about hardware utilization.

   - **Hardware utilization**: If psutil is installed, you can see utilization of the memory, CPU and GPU (utilization and memory).

   - **Terminal outputs**: Both stdout and stderr are logged.

- **Artifacts**: Displays files uploaded to the experiment.

- **Source code**: Displays sources uploaded to the experiment.

- **Parameters**: Displays parameters uploaded to the experiment (during experiment creation).

- **Details**: Displays additional metadata information:

   - **Metadata**: Additional information like experiment owner, creation and completion date, tags, description and more.
   - **Neptune metadata**: Neptune client version.
   - **Source summary**: Meta description of the source code.
   - **Git reference**: If you use Git version control, then extra information about Git is displayed (commit message, commit author, and more).
   - **Properties**: Experiment properties are displayed here (if set during experiment execution).