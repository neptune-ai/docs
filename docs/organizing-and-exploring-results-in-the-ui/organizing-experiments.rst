.. _guides-experiment-dashboard:

Organizing Experiments in a Dashboard
=====================================

|Youtube Video dashboard|

Neptune is a browser-enabled app that lets you visualize and browse experiments.


   .. image:: ../_static/images/organizing-and-exploring-results-in-the-ui/experiment-dashboard/experiment_general_view.png
      :target: ../_static/images/organizing-and-exploring-results-in-the-ui/experiment-dashboard/experiment_general_view.png
      :alt: Experiments view

The **Experiments** space displays all the experiments in a specific Project in table form.

There are several ways to organize your experiments.

Using tags
----------

You can create tag(s), which you assign to experiments. Later, you can quickly filter by these tags.

   .. image:: ../_static/images/organizing-and-exploring-results-in-the-ui/experiment-dashboard/tag_chooser.png
      :target: ../_static/images/organizing-and-exploring-results-in-the-ui/experiment-dashboard/tag_chooser.png
      :alt: Tag chooser
      :width: 250

Using dashboard views
---------------------

You can create a custom view of the dashboard. For example, you can filter rows by parameter or metric values and select a subset of useful columns.

Then you can save that view and quickly return to it by selecting it from the list of views.

   .. image:: ../_static/images/organizing-and-exploring-results-in-the-ui/experiment-dashboard/view_list.png
      :target: ../_static/images/organizing-and-exploring-results-in-the-ui/experiment-dashboard/view_list.png
      :alt: View list
      :width: 400

Customizing columns
-------------------

You can configure several data types logged to Neptune so that they are displayed as columns in the dashboard. They are metrics, text logs, properties and parameters. However, **all** data can be seen in the **single** experiment view.

.. note::

    Neptune automatically proposes columns based on what is different between experiments. This helps you see what changed quickly.

Grouping experiments
--------------------

You can group experiments by one or more column(s). The dashboard displays the selected columns, allowing you to make in-group and across-groups analysis of the experiments. Each group is represented by the first experiment that appears according to the sorting order. After opening it, each group shows at most 10 experiments - all experiments can be viewed by clicking **Show all**.

   .. image:: ../_static/images/organizing-and-exploring-results-in-the-ui/experiment-dashboard/group_by.png
      :target: ../_static/images/organizing-and-exploring-results-in-the-ui/experiment-dashboard/group_by.png
      :alt: Group columns

.. External links

.. |Youtube Video dashboard| raw:: html

    <iframe width="720" height="420" src="https://www.youtube.com/embed/QppF5CR_J1E" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

