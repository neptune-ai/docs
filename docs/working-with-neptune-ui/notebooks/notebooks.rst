Notebook UI
===========

|Youtube Video|

The `Notebooks tab <https://ui.neptune.ai/shared/onboarding/notebooks>`_ in the Neptune UI provides a table of all the Notebooks in the current project.

.. image:: ../../_static/images/notebooks/nb-view-11.png
    :target: ../../_static/images/notebooks/nb-view-11.png
    :alt: image


This view lets you see what your team members are working on, review details and checkpoints associated with a Notebook, as well as share or download a Notebook and compare two or more Notebooks.

The Notebook data is arranged in the following columns:

* Name
* Owner
* Latest checkpoint
* Description

In addition, for each Notebook, there are buttons for downloading the Notebook, comparing it with another Notebook, or for sharing a link to it.

A **Compare** button at the top right displays a Notebooks Comparison pane. See `Compare Notebooks <introduction.html#id3>`_.


Notebook contents
~~~~~~~~~~~~~~~~~
Once you select a Notebook, you can see all its contents, that is: code and markdown cells, outputs and execution count.

There are two tabs on the right:

- **Details**: Here are shown the ID, size, creation date, latest checkpoint, owner, description and associated experiments of the selected Notebook.
- **Checkpoints**: Here are listed all the checkpoints of the Notebook. Click a checkpoint to see the details in the main pane. From this tab, you can also access the experiments that are associated with the checkpoint.

You can also view snapshots of the work with the Notebook, as well as download, share or compare this checkpoint with another checkpoint.

.. image:: ../../_static/images/notebooks/nb-view-22.png
    :target: ../../_static/images/notebooks/nb-view-22.png
    :alt: image

Uploading and Downloading Notebook Checkpoints
----------------------------------------------


Notebooks are stored as files on your computer.

Each Notebook file (.ipynb) is a JSON containing everything that the user can see in a Notebook and some metadata.

Neptune uses metadata to associate particular files with Notebook entities on Neptune servers. That means that after a Notebook
is uploaded to Neptune, the file on disk is changed to include the ID of the entity on the Neptune server.

**Name changes**

If you copy a Notebook file (let’s call it "Notebook A") and
edit it with the intention of creating something completely separate from Notebook A,
the association with Notebook A on the Neptune server remains. If the name of the Notebook changes from "Notebook A",
you will be warned.


**Global accessibility**

When you download a Notebook checkpoint, the ID in the metadata is preserved, so that when, after some work,
you click **Upload**, Neptune knows that this may be another checkpoint in a particular Notebook.
You can do some work, upload some intermediate snapshot, go to another computer
(or another SageMaker instance, and so on), download the Notebook and keep on working on it.

The capability is comparable to Google Docs in that there’s a place where you store your work and you can access

it easily from wherever you choose.

**Collaboration**

Depending on their roles, members of a project can view and download all Notebooks (and their checkpoints) in the project.

- Viewers can download Notebooks.
- Contributors and Owners can also upload them.

When uploading a new Notebook, a user becomes the owner of this Notebook. Only the owner of a Notebook can upload
new checkpoints of this Notebook.

Uploading a Notebook
~~~~~~~~~~~~~~~~~~~~

You can upload Notebook checkpoints from Jupyter to Neptune.

**To upload the current Notebook as a checkpoint**:

1. Click **Upload**.

    .. image:: ../../_static/images/notebooks/upload_dialog.png
        :target: ../../_static/images/notebooks/upload_dialog.png
        :width: 450
        :alt: Upload Notebook dialog

2. In the dialog that is displayed, select a project from the list.
3. (Optional) Type in a checkpoint name and description.
4. Click **Upload checkpoint**.

A confirmation message is displayed. You can click the link in the message to open the Notebook in Neptune.

Downloading a Notebook
~~~~~~~~~~~~~~~~~~~~~~

You can download a specific Notebook checkpoint from Neptune to Jupyter.

**To download a Notebook checkpoint**:

1. Click **Download**.

    .. image:: ../../_static/images/notebooks/download_dialog.png
        :target: ../../_static/images/notebooks/download_dialog.png
        :width: 450
        :alt: Download Notebook dialog

2. In the dialog that is displayed, select the following from the respective lists:

  - Project
  - Notebook
  - Checkpoint


3. Click **Download**.

.. |Youtube Video| raw:: html

    <iframe width="720" height="420" src="https://www.youtube.com/embed/8qmz2yIndOw" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
