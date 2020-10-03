Workspaces
==========

Workspace Types
---------------

.. _core-concepts_workspace-types:

A *workspace* is a way to centrally manage projects, users and subscriptions.

Neptune has two workspace types: *individual* and *team*.

Individual
^^^^^^^^^^

* Each user is assigned an individual workspace with their ``username`` as the workspace name.
* The user is the only member of this workspace but may :ref:`invite collaborators <how-to_team-management_invite-to-project>` to projects.
* The user can create an unlimited number of projects in their individual workspace.

.. _administration-team-workspace:

Team
^^^^

* A team workspace comes in handy when an entire team needs to be managed centrally.
* Once :ref:`created <how-to_team-management_create-workspace>`, a team workspace can be managed by the :ref:`workspace Admin <administration-user-roles>`. The admin can manage users and subscription settings.
* Only users who joined team workspace can browse its content, subject to the assigned :ref:`role <administration-user-roles>` in the workspace or project.

Learn more about :ref:`project types <administration-project-types>` and :ref:`user roles <administration-user-roles>`.

**More info:**

.. toctree::
   :maxdepth: 1

    Create a workspace <create-workspace.rst>
    Add people to workspace <invite-to-workspace.rst>

