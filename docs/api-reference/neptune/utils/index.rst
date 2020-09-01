:mod:`neptune.utils`
====================

.. py:module:: neptune.utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   neptune.utils.map_values
   neptune.utils.map_keys
   neptune.utils.as_list
   neptune.utils.validate_notebook_path
   neptune.utils.align_channels_on_x
   neptune.utils.get_channel_name_stems
   neptune.utils.merge_dataframes
   neptune.utils.is_float
   neptune.utils.is_nan_or_inf
   neptune.utils.file_contains
   neptune.utils.in_docker
   neptune.utils.is_notebook
   neptune.utils._split_df_by_stems
   neptune.utils.discover_git_repo_location
   neptune.utils.update_session_proxies
   neptune.utils.get_git_info
   neptune.utils.with_api_exceptions_handler
   neptune.utils.glob
   neptune.utils.is_ipython


.. data:: _logger
   

   

.. data:: IS_WINDOWS
   

   

.. function:: map_values(f_value, dictionary)


.. function:: map_keys(f_key, dictionary)


.. function:: as_list(value)


.. function:: validate_notebook_path(path)


.. function:: align_channels_on_x(dataframe)


.. function:: get_channel_name_stems(columns)


.. function:: merge_dataframes(dataframes, on, how='outer')


.. function:: is_float(value)


.. function:: is_nan_or_inf(value)


.. function:: file_contains(filename, text)


.. function:: in_docker()


.. function:: is_notebook()


.. function:: _split_df_by_stems(df)


.. function:: discover_git_repo_location()


.. function:: update_session_proxies(session, proxies)


.. function:: get_git_info(repo_path=None)

   Retrieve information about git repository.

   If attempt fails, ``None`` will be returned.

   :param repo_path: | Path to the repository from which extract information about git.
                     | If ``None`` is passed, calling ``get_git_info`` is equivalent to calling
                       ``git.Repo(search_parent_directories=True)``.
                       Check `GitPython <https://gitpython.readthedocs.io/en/stable/reference.html#git.repo.base.Repo>`_
                       docs for more information.
   :type repo_path: :obj:`str`, optional, default is ``None``

   :returns: :class:`~neptune.git_info.GitInfo` - An object representing information about git repository.

   .. rubric:: Examples

   .. code:: python3

       # Get git info from the current directory
       git_info = get_git_info('.')


.. function:: with_api_exceptions_handler(func)


.. function:: glob(pathname)


.. function:: is_ipython()


