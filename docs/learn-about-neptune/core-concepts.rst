Core concepts
=============

Basic things
------------
Experiment
Experiments table
Notebook
Checkpoint
Wiki and Wiki page

User roles
----------
.. _core-concepts_user-roles:

Roles in organization
^^^^^^^^^^^^^^^^^^^^^
If you have an organization (check how to :ref:`create one <how-to_team-management_create-organization>`),
you can invite people to it. Two roles are available:

.. _core-concepts_user-roles_organization-admin:

**Admin**

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

**Member**

* Has no access to organization settings.
* For projects created in organization: member can be assigned to such project by project Owner.

Roles in project
^^^^^^^^^^^^^^^^
.. _core-concepts_user-roles_project-owner:

**Owner**

* Has edit access to all experiments and notebooks.
* Has edit access to project settings.
* Can remove project.
* Project creator is by default project owner.

**Contributor**

* Has edit access to own experiments and notebooks.
* Has view access to all experiments and notebooks.
* Can run experiments.
* Can add notebooks and make checkpoints.
* In the settings tab, have view-only access to people in project.
* Can leave project.

**Viewer**

* Has view-only access to experiments, notebooks and Wiki.
* Cannot run experiments, make notebook checkpoints.
* Has no access to project settings.

====

Organization types
------------------

.. _core-concepts_organization-types:

Organization is a way to centrally manage projects, users (and billing). Neptune has two organization types.

Individual
^^^^^^^^^^
* Each user is assigned individual organization with ``username`` as an organization name.
* User is the only member of this organization but may :ref:`invite collaborators <how-to_team-management_invite-to-project>` to projects.
* User can create unlimited number of projects in the individual organization.

Team
^^^^
* Team organization comes handy when entire team needs to be managed centrally.
* Once :ref:`created <how-to_team-management_create-organization>`, team organization can be managed by :ref:`organization Admin <core-concepts_user-roles_organization-admin>`. This include users and billing.
* Only Users who joined team organization can browse its content, subject to assigned :ref:`role <core-concepts_user-roles>` in the organization or project.

Learn more about :ref:`project types <core-concepts_project-types>` and :ref:`user roles <core-concepts_user-roles>`.

====

Project types
-------------
.. _core-concepts_project-types:

Private
^^^^^^^
Only people added to the project can see it. Project :ref:`owner <core-concepts_user-roles_project-owner>` can manage who has access to the project in the settings view.

Example view, where project Owner can manage project members

.. image:: ../_static/images/core-concepts/invite-to-project.png
   :target: ../_static/images/core-concepts/invite-to-project.png
   :alt: Invite user to the project

Public
^^^^^^
Public project is freely available to view by everyone who has access to the Internet.
Examples are: |credit-default-prediction| and |binary-classification-metrics|.

.. External links

.. |credit-default-prediction| raw:: html

    <a href="https://ui.neptune.ml/neptune-ml/credit-default-prediction" target="_blank">Credit default prediction</a>


.. |binary-classification-metrics| raw:: html

    <a href="https://ui.neptune.ml/neptune-ml/binary-classification-metrics" target="_blank">Binary classification metrics</a>
