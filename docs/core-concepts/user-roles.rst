User roles
==========

Roles in organization
---------------------
If you have an organization (check how to :ref:`create one <how-to_team-management_create-organization>`),
you can invite people to it. Two roles are available:

.. _core-concepts_user-roles_organization-owner:

Admin
^^^^^
* Has edit access to organization settings, that is *billing* and *people*.
* On the *billing* panel can edit payment options and plans and access to invoice data.
* On the *people* panel can add add / remove people from organization.
* By default admin is an *Owner* of all projects (editable option).

.. figure:: ../_static/images/core-concepts/org-settings.png
   :target: ../_static/images/core-concepts/org-settings.png
   :alt: organization settings button

   You can access organization settings by clicking blue button "Settings".

Member
^^^^^^
* No access to organization settings
* For projects created in organization: member can be added to such project by project Owner.

Roles in project
----------------

.. _core-concepts_user-roles_project-owner:

Owner
^^^^^
* Edit access to all experiments and notebooks
* Edit access to project settings
* Can remove project
* Project creator is by default project owner

Contributor
^^^^^^^^^^^
* Edit access to own experiments and notebooks
* Can run experiments
* Can add notebooks and make checkpoints
* In the settings tab, have view-only access to people in project
* View access to other Users experiments and notebooks
* Can leave project

Viewer
^^^^^^
* View-only access to experiments, notebooks and Wiki
* Cannot run experiments, make notebook checkpoints
* No access to project settings
