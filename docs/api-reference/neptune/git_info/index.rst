

:mod:`neptune.git_info`
=======================

.. py:module:: neptune.git_info


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   neptune.git_info.GitInfo



.. py:class:: GitInfo(commit_id, message='', author_name='', author_email='', commit_date='', repository_dirty=True, active_branch='', remote_urls=None)

   Bases: :class:`object`

   Class that keeps information about a git repository in experiment.

   When :meth:`~neptune.projects.Project.create_experiment` is invoked, instance of this class is created to
   store information about git repository.
   This information is later presented in the experiment details tab in the Neptune web application.

   :param commit_id: commit id sha.
   :type commit_id: :obj:`str`
   :param message: commit message.
   :type message: :obj:`str`, optional, default is ``""``
   :param author_name: commit author username.
   :type author_name: :obj:`str`, optional, default is ``""``
   :param author_email: commit author email.
   :type author_email: :obj:`str`, optional, default is ``""``
   :param commit_date: commit datetime.
   :type commit_date: :obj:`datetime.datetime`, optional, default is ``""``
   :param repository_dirty: ``True``, if the repository has uncommitted changes, ``False`` otherwise.
   :type repository_dirty: :obj:`bool`, optional, default is ``True``

   .. method:: __eq__(self, o)

      Return self==value.


   .. method:: __ne__(self, o)

      Return self!=value.


   .. method:: __str__(self)

      Return str(self).


   .. method:: __repr__(self)

      Return repr(self).




.. External links

.. |Neptune| raw:: html

    <a href="/api-reference/neptune/index.html#functions" target="_blank">Neptune</a>

.. |Session| raw:: html

    <a href="/api-reference/neptune/sessions/index.html?highlight=neptune%20sessions%20session#neptune.sessions.Session" target="_blank">Session</a>

.. |Project| raw:: html

    <a href="/api-reference/neptune/projects/index.html#neptune.projects.Project" target="_blank">Project</a>

.. |Experiment| raw:: html

    <a href="/api-reference/neptune/experiments/index.html?highlight=neptune%20experiment#neptune.experiments.Experiment" target="_blank">Experiment</a>

.. |Notebook| raw:: html

    <a href="/api-reference/neptune/notebook/index.html?highlight=notebook#neptune.notebook.Notebook" target="_blank">Notebook</a>

.. |Git Info| raw:: html

    <a href="/api-reference/neptune/git_info/index.html#neptune.git_info.GitInfo" target="_blank">Git Info</a>