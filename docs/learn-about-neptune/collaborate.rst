Collaborating in Neptune
========================

As an experiment tracking hub, Neptune provides several features for enabling knowledge sharing and collaboration among members
of your data science team.

To share a view on a project or any of its parts, you simply copy and paste the URL to it.

.. contents::
    :local:
    :depth: 1
    :backlinks: top

About Access to Organizations and Projects
------------------------------------------

- If the project is private, access to the shared content is restricted to users who have access to the project.
- If the project is public, anyone who has access to the Internet has access to the shared content.

For more information about user permissions in organizations and projects, see :ref:`user roles <core-concepts_user-roles>`.

Link Structure
--------------

Links to organizations in Neptune.ai are in the following format:
neptune.ai/*ORGANIZATION_NAME*/*PROJECT_NAME*

There are three parts:

- The Neptune domain: https://ui.neptune.ai

- /*ORGANIZATION_NAME*

  - In the case of a team project, the organization name is used.
  - In the case of a single-user project, the username is used.

- /*PROJECT_NAME*

**Examples**

- https://ui.neptune.ai/USERNAME/example-project is an example of a team project.
- https://ui.neptune.ai/jakub-czakon/r-integration is an example of a single-user project.


.. note:: You can also fetch these experiment details programmatically. For more information, see `Query API <../python-api/query-api.html>`_.

Additions to the project URL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The URL requires additional subdirectories to access a specific experiment.

**Example**

https://ui.neptune.ai/USERNAME/org/example-project/e/HELLO-191

Inside the experiment, there are tabs on the left side: Charts, Logs, Monitoring, Artifacts, Source code, Parameters, Details.
Each presents specific content logged to Neptune for the particular experiment.

Each tab has a specific URL, for example:

- Charts: https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-191/charts
- Logs: https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-191/logs
- Monitoring: https://ui.neptune.ai/USERNAME/example-project/e/HELLO-21/monitoring
- Artifacts: https://ui.neptune.ai/USERNAME/example-project/e/HELLO-191/artifacts
- Source code: https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-191/source-code
- Parameters: https://ui.neptune.ai/USERNAME/example-project/e/HELLO-191/parameters
- Details: https://ui.neptune.ai/USERNAME/example-project/e/HELLO-191/details


Sharing Links with Teammates
----------------------------
Neptune lets you share views of experiments, Notebooks, and projects by sharing the URL to a specific view.

.. contents::
    :local:
    :depth: 1
    :backlinks: top


There are two ways to get the URL you need for sharing:

- Clicking the **Copy** button.

   Copy the current URL to the clipboard by clicking any **Copy** button that appears. After copying the link, you can paste it, as needed, in an email message, message or other medium.

   **Example**

    .. image:: ../_static/images/core-concepts/metadata_copy.png
        :target: ../_static/images/core-concepts/metadata_copy.png
        :alt: Copy URL

- Copying the URL from the address bar.

Details of the URL composition appear in `Link Structure <collaborate.html#link-structure>`_, above.

Experiment details
^^^^^^^^^^^^^^^^^^

You can share short experiment IDs. An experiment ID is a unique identifier within the project.

In the **Experiment Details** tab, under the **Metadata** section, click the **Copy** button in the ID field.

**Example**

https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-21/details

Experiment charts and other resources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can share experiment charts and other resources.

Inside the experiment, there are tabs on the left side: Charts, Logs, Monitoring, Artifacts, Source code, Parameters, Details.
Each displays specific content that is logged to Neptune for the specific experiment.
`Each tab has a unique URL <collaborate.html#additions-to-the-project-url>`_.

Projects
^^^^^^^^
You can share projects.

**Example**

https://ui.neptune.ai/o/USERNAME/org/example-project/experiments?viewId=d7f80ebe-5bfe-4d12-97c1-2b1e6184a2ed


Experiment comparisons
^^^^^^^^^^^^^^^^^^^^^^

When you compare experiments in the UI, Neptune assigns it a unique URL. Share the URL to show your teammates the exact comparison
you made.

**Example**

https://ui.neptune.ai/o/USERNAME/org/example-project/compare?shortId=%5B%22HELLO-191%22%2C%22HELLO-197%22%2C%22HELLO-176%22%2C%22HELLO-177%22%2C%22HELLO-123%22%5D&viewId=6013ecbc-416d-4e5c-973e-871e5e9010e9



    .. image:: ../_static/images/core-concepts/compare_experiments.png
        :target: ../_static/images/core-concepts/compare_experiments.png
        :alt: Compare experiments



Notebook checkpoints
^^^^^^^^^^^^^^^^^^^^

Every time you make a Notebook checkpoint, Neptune assigns it a unique ID.
Similarly to other views in Neptune, you can share a link to the particular Notebook checkpoint.

**Example**

https://ui.neptune.ai/o/USERNAME/org/example-project/n/HPO-analysis-with-HiPlot-82bf08ed-c442-4d62-8f41-bc39fcc6c272/d1d4ad24-25f5-4286-974c-c0b08450d5e1

1. Click **Share** in the checkpoint actions menu:

    .. image:: ../_static/images/core-concepts/notebook_checkpoint.png
        :target: ../_static/images/core-concepts/notebook_checkpoint.png
        :alt: Notebook checkpoint
        :width: 400

2. Click **Copy**.

Notebook comparisons
^^^^^^^^^^^^^^^^^^^^
You can share a Notebook comparison with your teammates.

The Notebook comparison feature lets you compares two checkpoints site-by-site, like source code. The comparison has a unique link, as well.
You can copy the link in either of the following ways:

- Copy the link from the address bar in the browser.
- Click the **Share** button in the Notebook comparison view, then, in the dialog that appears, click **Copy**.

**Example**

https://ui.neptune.ai/o/USERNAME/org/example-project/compare-notebooks?sourceNotebookId=d311a774-7235-4f25-96eb-a5750eb6a1dc&sourceCheckpointId=289b0afa-41ba-4dbe-a9be-40ae8f03711a&targetNotebookId=d311a774-7235-4f25-96eb-a5750eb6a1dc&targetCheckpointId=eb59b83e-836e-4378-a326-1401dd499848


    .. image:: ../_static/images/core-concepts/notebook_comparison.png
        :target: ../_static/images/core-concepts/notebook_comparison.png
        :alt: Notebook comparison
        :width: 900

Working with the Project Wiki
-----------------------------

Each Neptune project has a built-in Wiki. The Wiki is a collabortive space for developing and sharing reports, insights, and remarks
about the project's progress, experiments and data exploration Notebooks.

Create a Wiki page
^^^^^^^^^^^^^^^^^^

1. Enter the relevant project.
2. Click the **Wiki** tab.
3. Click the **+** button.

    .. image:: ../_static/images/core-concepts/new_wiki_page.png
        :target: ../_static/images/core-concepts/new_wiki_page.png
        :alt: Create new Wiki page
        :width: 200

4. Type in the name of the new page.
5. Click **Save**.

Insert a comment
^^^^^^^^^^^^^^^^
When you hover on any content in the Wiki page, the **Comment** icon appears on the right side. Click it to make a comment.

    .. image:: ../_static/images/core-concepts/new_wiki_comment.png
        :target: ../_static/images/core-concepts/new_wiki_comment.png
        :alt: New Wiki comment
        :width: 800


Alternatively, select existing text and click the comment button in the context menu that appears:


    .. image:: ../_static/images/core-concepts/new_wiki_comment_menu.png
        :target: ../_static/images/core-concepts/new_wiki_comment_menu.png
        :alt: New Wiki comment
        :width: 600


Share a Wiki page
^^^^^^^^^^^^^^^^^

1. Enter the Wiki page you want to share.
2. Click the **Share** button:

    .. image:: ../_static/images/core-concepts/share_wiki_page.png
        :target: ../_static/images/core-concepts/share_wiki_page.png
        :alt: Share Wiki page
        :width: 600

3. In the dialog that appears, click **Copy** to send the link to the clipboard. From there, paste it as needed.

Formatting text, adding links and more
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you select existing text in a Wiki page, a contextual menu appears, presenting actions for formatting the selected text:

    .. image:: ../_static/images/core-concepts/wiki_context_menu.png
        :target: ../_static/images/core-concepts/wiki_context_menu.png
        :alt: Format Wiki text
        :width: 450


Formatting headings and adding mentions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
While in a Wiki page, you can display a menu for formatting a heading or adding a mention of a person or experiment:

   .. image:: ../_static/images/core-concepts/wiki_commands.png
        :target: ../_static/images/core-concepts/wiki_commands.png
        :alt: Wiki commands
        :width: 450

There are two ways to display the menu. Either:

- Type  **/**

  or

- Hover with the mouse until the **+** icon appears. Then click it.


Collaborative editing
^^^^^^^^^^^^^^^^^^^^^

You can edit Wiki pages collaboratively (think Google Docs).
In the figure shown here, four users are editing the Wiki page simultaneously. 
Their avatars are displayed next to the title page. 
Note that each user has one color associated with them. 


    .. image:: ../_static/images/core-concepts/wiki_collaborative_editing.png
        :target: ../_static/images/core-concepts/wiki_collaborative_editing.png
        :alt: Copy URL

In the example, the user who has a green color, has highlighted the word “Project”, 
so that it is also highlighted in green. 
Another example, is the blue user -- her mouse cursor is next to the word “progress”.


Avatar details
""""""""""""""

Hover on an avatar to see user details:

    .. image:: ../_static/images/core-concepts/avatar_highlight.png
        :target: ../_static/images/core-concepts/avatar_highlight.png
        :alt: Avatar details
        :width: 250

Highlighted text
""""""""""""""""

Hover on the highlighted text to see who highlighted it:

    .. image:: ../_static/images/core-concepts/editor_details.png
        :target: ../_static/images/core-concepts/editor_details.png
        :alt: Editor details
        :width: 250