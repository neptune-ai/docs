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



