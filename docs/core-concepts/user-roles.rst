User roles
==========

.. _core-concepts_user-roles:

Roles in organization
---------------------
If you have an organization (check how to :ref:`create one <how-to_team-management_create-organization>`),
you can invite people to it. Two roles are available:

.. _core-concepts_user-roles_organization-admin:

Admin
^^^^^
* Has edit access to organization settings, that is *billing* and *people*.
* On the *billing* panel can edit payment options and plans and access to invoice data.
* On the *people* panel can add / remove people from organization.
* By default admin is an *Owner* of all projects (editable option).

.. figure:: ../_static/images/core-concepts/org-settings.png
   :target: ../_static/images/core-concepts/org-settings.png
   :alt: organization settings button

   You can access organization settings by clicking blue button "Settings".

.. note::

    Organization must have at least one Admin, so last Admin cannot be removed from the organization.

Member
^^^^^^
* Has no access to organization settings.
* For projects created in organization: member can be assigned to such project by project Owner.

Roles in project
----------------

.. _core-concepts_user-roles_project-owner:

Owner
^^^^^
* Has edit access to all experiments and notebooks.
* Has edit access to project settings.
* Can remove project.
* Project creator is by default project owner.

Contributor
^^^^^^^^^^^
* Has edit access to own experiments and notebooks.
* Has view access to all experiments and notebooks.
* Can run experiments.
* Can add notebooks and make checkpoints.
* In the settings tab, have view-only access to people in project.
* Can leave project.

Viewer
^^^^^^
* Has view-only access to experiments, notebooks and Wiki.
* Cannot run experiments, make notebook checkpoints.
* Has no access to project settings.
