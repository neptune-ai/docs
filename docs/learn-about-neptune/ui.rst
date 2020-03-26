Working with the Neptune UI
===========================

The Neptune client is a browser-enabled app that lets you visualize and browse experiments.

SCREEN CAP

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
   :width: 500


3. Type in your credentials and click **Sign in**.

Explore the Dashboard
---------------------

SCREEN CAP

The dashboard displays all your experiments in table form.

Filtering - Query Language
You can filter experiments by typing filter query. It allows you to make complex filtering on experiments.

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

- **Choosing columns**: Dashboard’s columns are configurable. All types of data logged to Neptune can be seen as columns in the dashboard, for example parameters, metrics or tags.


Choosing rows a.k.a. Filtering
Neptune’s “Filtering - Query Language” can be used to filter experiments that are saved as a view.

- **Grouping**:

You can group experiments by some column. Then, the dashboard displays these columns, allowing you to make in-group and across-groups analysis of the experiments.

Comparison tools
^^^^^^^^^^^^^^^^
Neptune lets you compare between 2-10 experiments using in-depth analysis in the specialized view. Simply select experiments and compare.

Overlaid charts
In the comparison view, all metrics with the same name are placed on a single chart with one curve per experiment. Rich and customizable legend lets you see additional information.

Interactive comparison table
Experiments in comparison are displayed below charts in the comparison table. Each column is one experiment and rows are all properties and data associated with them.
Single experiment
Charts
All metrics (numeric type of data) are visualized as charts.
Chart sets
You can build your own subset of charts. Once created they are available for all experiments.
Logs
Logs are data that can be logged/tracked to the experiment. They have multiple types.
Numeric
Float/Int type of log.
Text
String type of log
Image
Images (image files, numpy array, matplotlib)
Monitoring
Displays information about hardware utilization
Hardware utilization
If psutil is installed, then you can see utilization of the memory, CPU and GPU (utilization and memory)
Terminal Outputs
Both stdout and stderr are logged.
Artifacts
Display files uploaded to the experiment.
Source code
Displays sources uploaded to the experiment.
Parameters
Display parameters uploaded to the experiment (during experiment creation).
Details
Display additional metadata information
Metadata
Additional information like experiment owner, creation and completion date, tags, description and more.
Neptune metadata
Neptune client version
Source summary
Meta description of the source code.
Git reference
If you use git version control, then extra info about git is displayed (commit message, commit author, and more)
Properties
Experiment properties are displayed here (if set during experiment execution).
