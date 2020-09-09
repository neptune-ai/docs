

:mod:`neptune.notebook`
=======================

.. py:module:: neptune.notebook


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   neptune.notebook.Notebook



.. py:class:: Notebook(backend, project, _id, owner)

   Bases: :class:`object`

   It contains all the information about a Neptune Notebook

   :param backend: A Backend object
   :type backend: :class:`~neptune.Backend`
   :param project: Project object
   :type project: :class:`~neptune.projects.Project`
   :param _id: Notebook uuid
   :type _id: :obj:`str`
   :param owner: Creator of the notebook is the Notebook owner
   :type owner: :obj:`str`

   .. rubric:: Examples

   .. code:: python3

       # Create a notebook in Neptune.
       notebook = project.create_notebook('data_exploration.ipynb')

   .. attribute:: id
      

      

   .. attribute:: owner
      

      

   .. method:: add_checkpoint(self, file_path)

      Uploads new checkpoint of the notebook to Neptune

      :param file_path: File path containing notebook contents
      :type file_path: :obj:`str`

      .. rubric:: Example

      .. code:: python3

          # Create a notebook.
          notebook = project.create_notebook('file.ipynb')

          # Change content in your notebook & save it

          # Upload new checkpoint
          notebook.add_checkpoint('file.ipynb')


   .. method:: get_path(self)

      Returns the path used to upload the current checkpoint of this notebook

      :returns: path of the current checkpoint
      :rtype: :obj:`str`


   .. method:: get_name(self)

      Returns the name used to upload the current checkpoint of this notebook

      :returns: the name of current checkpoint
      :rtype: :obj:`str`




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