

:mod:`neptune.projects`
=======================

.. py:module:: neptune.projects


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   neptune.projects.Project



.. data:: _logger
   

   

.. py:class:: Project(backend, internal_id, namespace, name)

   Bases: :class:`object`

   A class for storing information and managing Neptune project.

   :param backend: A Backend object.
   :type backend: :class:`~neptune.Backend`, required
   :param internal_id: UUID of the project.
   :type internal_id: :obj:`str`, required
   :param namespace: It can either be your workspace or user name.
   :type namespace: :obj:`str`, required
   :param name: project name.
   :type name: :obj:`str`, required

   .. note:: ``namespace`` and ``name`` joined together with ``/`` form ``project_qualified_name``.

   .. attribute:: full_id
      

      Project qualified name as :obj:`str`, for example `john/sandbox`.


   .. method:: get_members(self)

      Retrieve a list of project members.

      :returns: :obj:`list` of :obj:`str` - A list of usernames of project members.

      .. rubric:: Examples

      .. code:: python3

          project = session.get_projects('neptune-ai')['neptune-ai/Salt-Detection']
          project.get_members()


   .. method:: get_experiments(self, id=None, state=None, owner=None, tag=None, min_running_time=None)

      Retrieve list of experiments matching the specified criteria.

      All parameters are optional, each of them specifies a single criterion.
      Only experiments matching all of the criteria will be returned.

      :param id: | An experiment id like ``'SAN-1'`` or list of ids like ``['SAN-1', 'SAN-2']``.
                 | Matching any element of the list is sufficient to pass criterion.
      :type id: :obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``None``
      :param state: | An experiment state like ``'succeeded'`` or list of states like ``['succeeded', 'running']``.
                    | Possible values: ``'running'``, ``'succeeded'``, ``'failed'``, ``'aborted'``.
                    | Matching any element of the list is sufficient to pass criterion.
      :type state: :obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``None``
      :param owner:
                    | *Username* of the experiment owner (User who created experiment is an owner) like ``'josh'``
                      or list of owners like ``['frederic', 'josh']``.
                    | Matching any element of the list is sufficient to pass criterion.
      :type owner: :obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``None``
      :param tag: | An experiment tag like ``'lightGBM'`` or list of tags like ``['pytorch', 'cycleLR']``.
                  | Only experiments that have all specified tags will match this criterion.
      :type tag: :obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``None``
      :param min_running_time: Minimum running time of an experiment in seconds, like ``2000``.
      :type min_running_time: :obj:`int`, optional, default is ``None``

      :returns: :obj:`list` of :class:`~neptune.experiments.Experiment` objects.

      .. rubric:: Examples

      .. code:: python3

          # Fetch a project
          project = session.get_projects('neptune-ai')['neptune-ai/Salt-Detection']

          # Get list of experiments
          project.get_experiments(state=['aborted'], owner=['neyo'], min_running_time=100000)

          # Example output:
          # [Experiment(SAL-1609),
          #  Experiment(SAL-1765),
          #  Experiment(SAL-1941),
          #  Experiment(SAL-1960),
          #  Experiment(SAL-2025)]


   .. method:: get_leaderboard(self, id=None, state=None, owner=None, tag=None, min_running_time=None)

      Fetch Neptune experiments view as pandas ``DataFrame``.

      **returned DataFrame**

      | In the returned ``DataFrame`` each *row* is an experiment and *columns* represent all system properties,
        numeric and text logs, parameters and properties in these experiments.
      | Note that, returned ``DataFrame`` does not contain all columns across the entire project.
      | Some columns may be empty, since experiments may define various logs, properties, etc.
      | For each log at most one (the last one) value is returned per experiment.
      | Text values are trimmed to 255 characters.

      **about parameters**

      All parameters are optional, each of them specifies a single criterion.
      Only experiments matching all of the criteria will be returned.

      :param id: | An experiment id like ``'SAN-1'`` or list of ids like ``['SAN-1', 'SAN-2']``.
                 | Matching any element of the list is sufficient to pass criterion.
      :type id: :obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``None``
      :param state: | An experiment state like ``'succeeded'`` or list of states like ``['succeeded', 'running']``.
                    | Possible values: ``'running'``, ``'succeeded'``, ``'failed'``, ``'aborted'``.
                    | Matching any element of the list is sufficient to pass criterion.
      :type state: :obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``None``
      :param owner:
                    | *Username* of the experiment owner (User who created experiment is an owner) like ``'josh'``
                      or list of owners like ``['frederic', 'josh']``.
                    | Matching any element of the list is sufficient to pass criterion.
      :type owner: :obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``None``
      :param tag: | An experiment tag like ``'lightGBM'`` or list of tags like ``['pytorch', 'cycleLR']``.
                  | Only experiments that have all specified tags will match this criterion.
      :type tag: :obj:`str` or :obj:`list` of :obj:`str`, optional, default is ``None``
      :param min_running_time: Minimum running time of an experiment in seconds, like ``2000``.
      :type min_running_time: :obj:`int`, optional, default is ``None``

      :returns: :obj:`pandas.DataFrame` - Fetched Neptune experiments view.

      .. rubric:: Examples

      .. code:: python3

          # Fetch a project.
          project = session.get_projects('neptune-ai')['neptune-ai/Salt-Detection']

          # Get DataFrame that resembles experiment view.
          project.get_leaderboard(state=['aborted'], owner=['neyo'], min_running_time=100000)


   .. method:: create_experiment(self, name=None, description=None, params=None, properties=None, tags=None, upload_source_files=None, abort_callback=None, logger=None, upload_stdout=True, upload_stderr=True, send_hardware_metrics=True, run_monitoring_thread=True, handle_uncaught_exceptions=True, git_info=None, hostname=None, notebook_id=None, notebook_path=None)

      Create and start Neptune experiment.

      Create experiment, set its status to `running` and append it to the top of the experiments view.
      All parameters are optional, hence minimal invocation: ``neptune.create_experiment()``.

      :param name: Editable name of the experiment.
                   Name is displayed in the experiment's `Details` (`Metadata` section)
                   and in `experiments view` as a column.
      :type name: :obj:`str`, optional, default is ``'Untitled'``
      :param description: Editable description of the experiment.
                          Description is displayed in the experiment's `Details` (`Metadata` section)
                          and can be displayed in the `experiments view` as a column.
      :type description: :obj:`str`, optional, default is ``''``
      :param params: Parameters of the experiment.
                     After experiment creation ``params`` are read-only
                     (see: :meth:`~neptune.experiments.Experiment.get_parameters`).
                     Parameters are displayed in the experiment's `Details` (`Parameters` section)
                     and each key-value pair can be viewed in `experiments view` as a column.
      :type params: :obj:`dict`, optional, default is ``{}``
      :param properties: Properties of the experiment.
                         They are editable after experiment is created.
                         Properties are displayed in the experiment's `Details` (`Properties` section)
                         and each key-value pair can be viewed in `experiments view` as a column.
      :type properties: :obj:`dict`, optional, default is ``{}``
      :param tags: Must be list of :obj:`str`. Tags of the experiment.
                   They are editable after experiment is created
                   (see: :meth:`~neptune.experiments.Experiment.append_tag`
                   and :meth:`~neptune.experiments.Experiment.remove_tag`).
                   Tags are displayed in the experiment's `Details` (`Metadata` section)
                   and can be viewed in `experiments view` as a column.
      :type tags: :obj:`list`, optional, default is ``[]``
      :param upload_source_files: List of source files to be uploaded. Must be list of :obj:`str` or single :obj:`str`.
                                  Uploaded sources are displayed in the experiment's `Source code` tab.

                                  | If ``None`` is passed, Python file from which experiment was created will be uploaded.
                                  | Pass empty list (``[]``) to upload no files.
                                  | Unix style pathname pattern expansion is supported. For example, you can pass ``'*.py'`` to upload
                                    all python source files from the current directory.
                                    For Python 3.5 or later, paths of uploaded files on server are resolved as relative to the
                                  | calculated common root of all uploaded source  files. For older Python versions, paths on server are
                                  | resolved always as relative to the current directory.
                                    For recursion lookup use ``'**/*.py'`` (for Python 3.5 and later).
                                    For more information see `glob library <https://docs.python.org/3/library/glob.html>`_.
      :type upload_source_files: :obj:`list` or :obj:`str`, optional, default is ``None``
      :param abort_callback: Callback that defines how `abort experiment` action in the Web application should work.
                             Actual behavior depends on your setup:

                                 * (default) If ``abort_callback=None`` and `psutil <https://psutil.readthedocs.io/en/latest/>`_
                                   is installed, then current process and it's children are aborted by sending `SIGTERM`.
                                   If, after grace period, processes are not terminated, `SIGKILL` is sent.
                                 * If ``abort_callback=None`` and `psutil <https://psutil.readthedocs.io/en/latest/>`_
                                   is **not** installed, then `abort experiment` action just marks experiment as *aborted*
                                   in the Web application. No action is performed on the current process.
                                 * If ``abort_callback=callable``, then ``callable`` is executed when `abort experiment` action
                                   in the Web application is triggered.
      :type abort_callback: :obj:`callable`, optional, default is ``None``
      :param logger: If Python's `Logger <https://docs.python.org/3/library/logging.html#logging.Logger>`_
                     is passed, new experiment's `text log`
                     (see: :meth:`~neptune.experiments.Experiment.log_text`) with name `"logger"` is created.
                     Each time `Python logger` logs new data, it is automatically sent to the `"logger"` in experiment.
                     As a results all data from `Python logger` are in the `Logs` tab in the experiment.
      :type logger: :obj:`logging.Logger` or `None`, optional, default is ``None``
      :param upload_stdout: Whether to send stdout to experiment's *Monitoring*.
      :type upload_stdout: :obj:`Boolean`, optional, default is ``True``
      :param upload_stderr: Whether to send stderr to experiment's *Monitoring*.
      :type upload_stderr: :obj:`Boolean`, optional, default is ``True``
      :param send_hardware_metrics: Whether to send hardware monitoring logs (CPU, GPU, Memory utilization) to experiment's *Monitoring*.
      :type send_hardware_metrics: :obj:`Boolean`, optional, default is ``True``
      :param run_monitoring_thread: Whether to run thread that pings Neptune server in order to determine if experiment is responsive.
      :type run_monitoring_thread: :obj:`Boolean`, optional, default is ``True``
      :param handle_uncaught_exceptions:
                                         Two options ``True`` and ``False`` are possible:

                                             * If set to ``True`` and uncaught exception occurs, then Neptune automatically place
                                               `Traceback` in the experiment's `Details` and change experiment status to `Failed`.
                                             * If set to ``False`` and uncaught exception occurs, then no action is performed
                                               in the Web application. As a consequence, experiment's status is `running` or `not responding`.
      :type handle_uncaught_exceptions: :obj:`Boolean`, optional, default is ``True``
      :param git_info:
                       | Instance of the class :class:`~neptune.git_info.GitInfo` that provides information about
                         the git repository from which experiment was started.
                       | If ``None`` is passed,
                         system attempts to automatically extract information about git repository in the following way:

                             * System looks for `.git` file in the current directory and, if not found,
                               goes up recursively until `.git` file will be found
                               (see: :meth:`~neptune.utils.get_git_info`).
                             * If there is no git repository,
                               then no information about git is displayed in experiment details in Neptune web application.
      :type git_info: :class:`~neptune.git_info.GitInfo`, optional, default is ``None``
      :param hostname: If ``None``, neptune automatically get `hostname` information.
                       User can also set `hostname` directly by passing :obj:`str`.
      :type hostname: :obj:`str`, optional, default is ``None``

      :returns: :class:`~neptune.experiments.Experiment` object that is used to manage experiment and log data to it.

      :raises ExperimentValidationError: When provided arguments are invalid.
      :raises ExperimentLimitReached: When experiment limit in the project has been reached.

      .. rubric:: Examples

      .. code:: python3

          # minimal invoke
          neptune.create_experiment()

          # explicitly return experiment object
          experiment = neptune.create_experiment()

          # create experiment with name and two parameters
          neptune.create_experiment(name='first-pytorch-ever',
                                    params={'lr': 0.0005,
                                            'dropout': 0.2})

          # create experiment with name and description, and no sources files uploaded
          neptune.create_experiment(name='neural-net-mnist',
                                    description='neural net trained on MNIST',
                                    upload_source_files=[])

          # Send all py files in cwd (excluding hidden files with names beginning with a dot)
          neptune.create_experiment(upload_source_files='*.py')

          # Send all py files from all subdirectories (excluding hidden files with names beginning with a dot)
          # Supported on Python 3.5 and later.
          neptune.create_experiment(upload_source_files='**/*.py')

          # Send all files and directories in cwd (excluding hidden files with names beginning with a dot)
          neptune.create_experiment(upload_source_files='*')

          # Send all files and directories in cwd including hidden files
          neptune.create_experiment(upload_source_files=['*', '.*'])

          # Send files with names being a single character followed by '.py' extension.
          neptune.create_experiment(upload_source_files='?.py')

          # larger example
          neptune.create_experiment(name='first-pytorch-ever',
                                    params={'lr': 0.0005,
                                            'dropout': 0.2},
                                    properties={'key1': 'value1',
                                                'key2': 17,
                                                'key3': 'other-value'},
                                    description='write longer description here',
                                    tags=['list-of', 'tags', 'goes-here', 'as-list-of-strings'],
                                    upload_source_files=['training_with_pytorch.py', 'net.py'])


   .. method:: _get_experiment_link(self, experiment)


   .. method:: create_notebook(self)

      Create a new notebook object and return corresponding :class:`~neptune.notebook.Notebook` instance.

      :returns: :class:`~neptune.notebook.Notebook` object.

      .. rubric:: Examples

      .. code:: python3

          # Instantiate a session and fetch a project
          project = neptune.init()

          # Create a notebook in Neptune
          notebook = project.create_notebook()


   .. method:: get_notebook(self, notebook_id)

      Get a :class:`~neptune.notebook.Notebook` object with given ``notebook_id``.

      :returns: :class:`~neptune.notebook.Notebook` object.

      .. rubric:: Examples

      .. code:: python3

          # Instantiate a session and fetch a project
          project = neptune.init()

          # Get a notebook object
          notebook = project.get_notebook('d1c1b494-0620-4e54-93d5-29f4e848a51a')


   .. method:: __str__(self)

      Return str(self).


   .. method:: __repr__(self)

      Return repr(self).


   .. method:: __eq__(self, o)

      Return self==value.


   .. method:: __ne__(self, o)

      Return self!=value.


   .. method:: _fetch_leaderboard(self, id, state, owner, tag, min_running_time)


   .. staticmethod:: _sort_leaderboard_columns(column_names)


   .. method:: _get_current_experiment(self)


   .. method:: _push_new_experiment(self, new_experiment)


   .. method:: _remove_stopped_experiment(self, experiment)


   .. method:: _shutdown_hook(self)




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