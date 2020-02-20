Core Concepts
=============

This page familiarizes you with the Neptune platform by describing some of its core elements.

.. contents::
    :local:
    :depth: 1
    :backlinks: top

Projects
--------

A Neptune project is a collection of experiments.

In addition to experiments, a project includes `Notebooks <../notebooks/introduction.html>`_ and a Wiki.

Project settings are accessible from the **Settings** tab in the Project view.

Project types
^^^^^^^^^^^^^

A Project is either private or public.

You can set a project’s public/private visibility in the **Properties** tab under **Settings**.

.. _core-concepts_project-types:

Private projects
""""""""""""""""
Only people who have been added to a private project can see it.
The Project :ref:`Owner <core-concepts_user-roles_project-owner>` can manage who has access to the project in the Settings tab.


Public projects
"""""""""""""""
Public projects are freely available to view by anyone who has access to the Internet.
Examples are: |credit-default-prediction| and |binary-classification-metrics|.

.. External links

.. |credit-default-prediction| raw:: html

    <a href="https://ui.neptune.ai/neptune-ai/credit-default-prediction" target="_blank">Credit default prediction</a>


.. |binary-classification-metrics| raw:: html

    <a href="https://ui.neptune.ai/neptune-ai/binary-classification-metrics" target="_blank">Binary classification metrics</a>


Learn more
^^^^^^^^^^

- `Create a project <team-management.html#create-a-project>`_
- `Add users to a project <team-management.html#add-users-to-a-project>`_

----------------------------------------------------------------------------------------------

Organizations
-------------

An Organization is a way to centrally manage projects, users and subscription.

Organization types
^^^^^^^^^^^^^^^^^^

.. _core-concepts_organization-types:

Neptune has two organization types: Individual and Team

Individual organizations
""""""""""""""""""""""""
* Each user is assigned an individual organization with ``username`` as the organization name.
* The user is the only member of this organization but may :ref:`invite collaborators <how-to_team-management_invite-to-project>` to projects.
* The user can create an unlimited number of projects in the individual organization.

Team organizations
""""""""""""""""""
* Team organizations comes in handy when an entire team needs to be managed centrally.
* Once :ref:`created <how-to_team-management_create-organization>`, a team organization can be managed by :ref:`organization Admin <core-concepts_user-roles_organization-admin>`. This includes users and subscription.
* Only users who joined team organizations can browse its content, subject to their assigned :ref:`roles <core-concepts_user-roles>` in the organization or project.


Learn more
^^^^^^^^^^

- `Create an organization <team-management.html#create-an-organization>`_
- `Invite users to an organization <team-management.html#invite-users-to-an-organization>`_

----------------------------------------------------------------------------------------------

User Roles
----------
.. _core-concepts_user-roles:

Subject to their roles, users belong to organizations and collaborate on projects.
The roles in an organization are different from the roles in a project.

Roles in an organization
^^^^^^^^^^^^^^^^^^^^^^^^
`Have you already created an organization? <team-management.html#how-to-team-management-create-organization>`_

If so, you can invite people to join it. You can assign the members one of two roles: Admin or Member.

.. _core-concepts_user-roles_organization-admin:

**Admin**

.. note::

    An organization must have at least one Admin.

By default, an Admin is the Owner of all projects.

Admins have edit permissions for organization settings, which they can access by
clicking **Settings** for the relevant organization.

.. image:: ../_static/images/core-concepts/org-settings.png
   :target: ../_static/images/core-concepts/org-settings.png
   :alt: Organization settings button
   :width: 250

Settings include people and subscription:

.. Also: in the pix below, there are more tabs
.. Is it Billing or Subscription?

* In the **Subscription** tab, Admins can edit payment options and plans and access invoice data.
* In the **People** tab, Admins can add people to an organization or remove them.

**Member**

Regular members have no access to organization settings. For projects created in an organization, a member can be
assigned by the project Owner.

Roles in a project
^^^^^^^^^^^^^^^^^^

Members of projects can be one of three types: Owner, Contributor or Viewer.

.. _core-concepts_user-roles_project-owner:

**Owner**

* Has edit access to all experiments and Notebooks.
* Has edit access to project settings.
* Can remove projects.
* A project creator is by default the project Owner.

**Contributor**

* Has edit access to all experiments and Notebooks.
* Has edit access to project settings.
* Can run experiments.
* Can add Notebooks and make checkpoints.
* In the Settings tab, has view-only access to people in a project.
* Can leave a project.

**Viewer**

* Has view-only access to experiments, Notebooks and wikis.
* Cannot run experiments or make Notebook checkpoints.
* Has no access to project settings.