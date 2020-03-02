Collaborating in Neptune
========================

As an experiment tracking hub, Neptune provides several features for enabling knowledge sharing and collaboration among members 
of your data science team.

The sections below describe these.

In multiple places in Neptune, there is “copy” icon: , click on it to copy the content.

.. contents::
    :local:
    :depth: 1
    :backlinks: top

About Access to Organizations and Projects
------------------------------------------

- If the project is private, access to the shared content is restricted to users who have access to the project.
- If the project is public, anyone who has access to the Internet has access to the shared content.

For more information about user permissions in organizations and projects, see :ref:`user roles <core-concepts_user-roles>`.

Sharing with Teammates
----------------------
Neptune lets you share details of experiments, Notebooks, and projects.

.. contents::
    :local:
    :depth: 1
    :backlinks: top


Experiment details
^^^^^^^^^^^^^^^^^^

You can share experiment details with your teammates.

Each link to the experiment is unique, so you can send it using your favorite communication medium. 
Example: https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-21. This link will guide you to the experiment charts.

Share experiment short ID
In the experiment details tab, there is a metadata section that allows users to copy experiment short ID. 
It is a unique experiment identifier within the project.

Example https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-21/details

Experiment charts and other resources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can share experiment charts and other resources with your teammates.

Inside the experiment, there are tabs on the left side: Charts, Logs, Monitoring, Artifacts, Source code, Parameters, Details. 
Each present specific content logged to Neptune for the experiment in question. Each tab has a separate link, for example:

- Charts: https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-191/charts
- Logs: https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-191/logs
- Monitoring: https://ui.neptune.ai/USERNAME/example-project/e/HELLO-21/monitoring
- Artifacts: https://ui.neptune.ai/USERNAME/example-project/e/HELLO-191/artifacts
- Source code: https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-191/source-code
- Parameters: https://ui.neptune.ai/USERNAME/example-project/e/HELLO-191/parameters
- Details: https://ui.neptune.ai/USERNAME/example-project/e/HELLO-191/details

Note:
You can also fetch these experiment details programmatically. For more information, see Query API.

Projects
^^^^^^^^
You can share projects with your teammates.

Links to projects are unique, they consist of two pieces: https://ui.neptune.ai then project name for example: https://ui.neptune.ai/USERNAME/example-project or https://ui.neptune.ai/jakub-czakon/r-integration.
Share experiments view
Views on the experiment table are associated with a unique id in the URL. It means that you can share a link to a particular view, for example: https://ui.neptune.ai/o/USERNAME/org/example-project/experiments?viewId=d7f80ebe-5bfe-4d12-97c1-2b1e6184a2ed
Just to remind you, you switch between experiment views using drop-down menu, depicted below:

Experiment comparisons
^^^^^^^^^^^^^^^^^^^^^^

You can share experiment comparisons with your teammates.

When you compare experiments on the compare you, note that the URL in your browser contains experiments IDs. In this way, 
you can point your team members to some comparison view that has specific experiments.

Example compare view is like this.

It is available under this link:
https://ui.neptune.ai/o/USERNAME/org/example-project/compare?shortId=%5B%22HELLO-191%22%2C%22HELLO-197%22%2C%22HELLO-176%22%2C%22HELLO-177%22%2C%22HELLO-123%22%5D&viewId=6013ecbc-416d-4e5c-973e-871e5e9010e9


Notebook checkpoints
^^^^^^^^^^^^^^^^^^^^
You can share a Notebook checkpoint with your teammates.

Every time you make a Notebook checkpoint, Neptune assigns it a unique ID. 
Similarly to other views in Neptune, you can share a link to the particular Notebook checkpoint.
Example link: https://ui.neptune.ai/o/USERNAME/org/example-project/n/HPO-analysis-with-HiPlot-82bf08ed-c442-4d62-8f41-bc39fcc6c272/d1d4ad24-25f5-4286-974c-c0b08450d5e1
You can either copy the link from the address bar in the browser or click **Share** in the checkpoint actions menu. See below:

Then you are ready to click “copy” icon:

Notebook comparisons
^^^^^^^^^^^^^^^^^^^^
You can share a Notebook comparison with your teammates.

The Notebook comparison feature lets you compares two checkpoints site-by-site, like source code. The comparison has a unique link, as well. 
You can copy the link in either of the following ways:
- Copy the link from the address bar in the browser
- Click the **Share** icon in the Notebook comparison view, then, in the dialog that pops up, click the copy icon.


Example link: https://ui.neptune.ai/o/USERNAME/org/example-project/compare-notebooks?sourceNotebookId=d311a774-7235-4f25-96eb-a5750eb6a1dc&sourceCheckpointId=289b0afa-41ba-4dbe-a9be-40ae8f03711a&targetNotebookId=d311a774-7235-4f25-96eb-a5750eb6a1dc&targetCheckpointId=eb59b83e-836e-4378-a326-1401dd499848


Using the Project Wiki
----------------------

Each Neptune project has a built-in Wiki. The Wiki is a great place for developing and sharing reports, insights, and remarks 
about the project's progress, experiments and data exploration Notebooks.

Create page
Click “+” icon

Now, write down the name of the page and click “Save” to create a page.

Make comment
When you hover on any content in the Wiki page you see the “Comment” icon on the right side. Click on it to make a comment. See below:

Alternatively, you can select text and use the context menu, like depicted below:

Share page
Click on the “Share” button to share the link to the page:


Now, you can click “copy” button:

Actions
When you select some text, you can see contextual actions for text formatting:


Mentions
On the Wiki page the slash “/” triggers menu with widgets that you can add to the page contents:

Alternatively, you can access this menu by clicking on the “” icon:

Collaborative editing
You can edit wiki pages collaboratively (think Google Docs).
In the example below, there are four users editing the Wiki page simultaneously. Their avatars are displayed next to the title page. Note that each user has one color associated. For example, a user who has a green color associated, and she highlighted “Project” word, so that it is also highlighted in green. Another example, is blue user: her prompt is next to the word “progress”. See below:

Avatars’ details
Hover on the avatars to see user details:

Highlighted text
Hover on the highlighted text to see who highlighted it: