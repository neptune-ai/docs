Using Experiments Dashboard
===========================

|Youtube Video dashboard|

Neptune is a browser-enabled app that lets you visualize and browse experiments.


   .. image:: /_static/images/core-concepts/experiment_general_view.png
      :target: /_static/images/core-concepts/experiment_general_view.png
      :alt: Experiments view

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


.. |Youtube Video dashboard| raw:: html

    <iframe width="720" height="420" src="https://www.youtube.com/embed/QppF5CR_J1E" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

