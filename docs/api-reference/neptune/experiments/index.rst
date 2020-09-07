

:mod:`neptune.experiments`
==========================

.. py:module:: neptune.experiments


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   neptune.experiments.Experiment



.. data:: _logger
   

   

.. py:class:: Experiment(backend, project, _id, internal_id)

   Bases: :class:`object`

   A class for managing Neptune experiment.

   Each time User creates new experiment instance of this class is created.
   It lets you manage experiment, :meth:`~neptune.experiments.Experiment.log_metric`,
   :meth:`~neptune.experiments.Experiment.log_text`,
   :meth:`~neptune.experiments.Experiment.log_image`,
   :meth:`~neptune.experiments.Experiment.set_property`,
   and much more.


   :param backend: A Backend object
   :type backend: :obj:`neptune.Backend`
   :param project: The project this experiment belongs to
   :type project: :obj:`neptune.Project`
   :param _id: Experiment id
   :type _id: :obj:`str`
   :param internal_id: internal UUID
   :type internal_id: :obj:`str`

   .. rubric:: Example

   Assuming that `project` is an instance of :class:`~neptune.projects.Project`.

   .. code:: python3

       experiment = project.create_experiment()

   .. warning::

      User should never create instances of this class manually.
      Always use: :meth:`~neptune.projects.Project.create_experiment`.

   .. attribute:: IMAGE_SIZE_LIMIT_MB
      :annotation: = 15

      

   .. attribute:: id
      

      Experiment short id

      | Combination of project key and unique experiment number.
      | Format is ``<project_key>-<experiment_number>``, for example: ``MPI-142``.

      :returns: :obj:`str` - experiment short id

      .. rubric:: Examples

      Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

      .. code:: python3

          exp_id = experiment.id


   .. attribute:: name
      

      Experiment name

      :returns: :obj:`str` experiment name

      .. rubric:: Examples

      Assuming that `project` is an instance of :class:`~neptune.projects.Project`.

      .. code:: python3

          experiment = project.create_experiment('exp_name')
          exp_name = experiment.name


   .. attribute:: state
      

      Current experiment state

      Possible values: `'running'`, `'succeeded'`, `'failed'`, `'aborted'`.

      :returns: :obj:`str` - current experiment state

      .. rubric:: Examples

      Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

      .. code:: python3

          state_str = experiment.state


   .. attribute:: internal_id
      

      

   .. attribute:: limits
      

      

   .. method:: get_system_properties(self)

      Retrieve experiment properties.

      | Experiment properties are for example: `owner`, `created`, `name`, `hostname`.
      | List of experiment properties may change over time.

      :returns: :obj:`dict` - dictionary mapping a property name to value.

      .. rubric:: Examples

      Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

      .. code:: python3

          sys_properties = experiment.get_system_properties


   .. method:: get_tags(self)

      Get tags associated with experiment.

      :returns: :obj:`list` of :obj:`str` with all tags for this experiment.

      .. rubric:: Example

      Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

      .. code:: python3

          experiment.get_tags()


   .. method:: append_tag(self, tag, *tags)

      Append tag(s) to the current experiment.

      Alias: :meth:`~neptune.experiments.Experiment.append_tags`.
      Only ``[a-zA-Z0-9]`` and ``-`` (dash) characters are allowed in tags.

      :param tag:
                  Tag(s) to add to the current experiment.

                      * If :obj:`str` is passed, singe tag is added.
                      * If multiple - comma separated - :obj:`str` are passed, all of them are added as tags.
                      * If :obj:`list` of :obj:`str` is passed, all elements of the :obj:`list` are added as tags.
      :type tag: single :obj:`str` or multiple :obj:`str` or :obj:`list` of :obj:`str`

      .. rubric:: Examples

      .. code:: python3

          neptune.append_tag('new-tag')  # single tag
          neptune.append_tag('first-tag', 'second-tag', 'third-tag')  # few str
          neptune.append_tag(['first-tag', 'second-tag', 'third-tag'])  # list of str


   .. method:: append_tags(self, tag, *tags)

      Append tag(s) to the current experiment.

      Alias for: :meth:`~neptune.experiments.Experiment.append_tag`


   .. method:: remove_tag(self, tag)

      Removes single tag from the experiment.

      :param tag: Tag to be removed
      :type tag: :obj:`str`

      .. rubric:: Example

      Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

      .. code:: python3

          # assuming experiment has tags: `['tag-1', 'tag-2']`.
          experiment.remove_tag('tag-1')

      .. note:: Removing a tag that is not assigned to this experiment is silently ignored.


   .. method:: get_channels(self)

      Alias for :meth:`~neptune.experiments.Experiment.get_logs`


   .. method:: get_logs(self)

      Retrieve all log names along with their last values for this experiment.

      :returns: :obj:`dict` - A dictionary mapping a log names to the log's last value.

      .. rubric:: Example

      Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

      .. code:: python3

          exp_logs = experiment.get_logs()


   .. method:: _get_system_channels(self)


   .. method:: send_metric(self, channel_name, x, y=None, timestamp=None)

      Log metrics (numeric values) in Neptune.

      Alias for :meth:`~neptune.experiments.Experiment.log_metric`


   .. method:: log_metric(self, log_name, x, y=None, timestamp=None)

      Log metrics (numeric values) in Neptune

      | If a log with provided ``log_name`` does not exist, it is created automatically.
      | If log exists (determined by ``log_name``), then new value is appended to it.

      :param log_name: The name of log, i.e. `mse`, `loss`, `accuracy`.
      :type log_name: :obj:`str`
      :param x: Depending, whether ``y`` parameter is passed:

                * ``y`` not passed: The value of the log (data-point).
                * ``y`` passed: Index of log entry being appended. Must be strictly increasing.
      :type x: :obj:`double`
      :param y: The value of the log (data-point).
      :type y: :obj:`double`, optional, default is ``None``
      :param timestamp: Timestamp to be associated with log entry. Must be Unix time.
                        If ``None`` is passed, `time.time() <https://docs.python.org/3.6/library/time.html#time.time>`_
                        (Python 3.6 example) is invoked to obtain timestamp.
      :type timestamp: :obj:`time`, optional, default is ``None``

      .. rubric:: Example

      Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment` and
      'accuracy' log does not exists:

      .. code:: python3

          # Both calls below have the same effect

          # Common invocation, providing log name and value
          experiment.log_metric('accuracy', 0.5)
          experiment.log_metric('accuracy', 0.65)
          experiment.log_metric('accuracy', 0.8)

          # Providing both x and y params
          experiment.log_metric('accuracy', 0, 0.5)
          experiment.log_metric('accuracy', 1, 0.65)
          experiment.log_metric('accuracy', 2, 0.8)

      .. note::

         For efficiency, logs are uploaded in batches via a queue.
         Hence, if you log a lot of data, you may experience slight delays in Neptune web application.

      .. note::

         Passing either ``x`` or ``y`` coordinate as NaN or +/-inf causes this log entry to be ignored.
         Warning is printed to ``stdout``.


   .. method:: send_text(self, channel_name, x, y=None, timestamp=None)

      Log text data in Neptune.

      Alias for :meth:`~neptune.experiments.Experiment.log_text`


   .. method:: log_text(self, log_name, x, y=None, timestamp=None)

      Log text data in Neptune

      | If a log with provided ``log_name`` does not exist, it is created automatically.
      | If log exists (determined by ``log_name``), then new value is appended to it.

      :param log_name: The name of log, i.e. `mse`, `my_text_data`, `timing_info`.
      :type log_name: :obj:`str`
      :param x: Depending, whether ``y`` parameter is passed:

                * ``y`` not passed: The value of the log (data-point). Must be ``str``.
                * ``y`` passed: Index of log entry being appended. Must be strictly increasing.
      :type x: :obj:`double` or :obj:`str`
      :param y: The value of the log (data-point).
      :type y: :obj:`str`, optional, default is ``None``
      :param timestamp: Timestamp to be associated with log entry. Must be Unix time.
                        If ``None`` is passed, `time.time() <https://docs.python.org/3.6/library/time.html#time.time>`_
                        (Python 3.6 example) is invoked to obtain timestamp.
      :type timestamp: :obj:`time`, optional, default is ``None``

      .. rubric:: Example

      Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`:

      .. code:: python3

          # common case, where log name and data are passed
          neptune.log_text('my_text_data', str(data_item))

          # log_name, x and timestamp are passed
          neptune.log_text(log_name='logging_losses_as_text',
                           x=str(val_loss),
                           timestamp=1560430912)

      .. note::

         For efficiency, logs are uploaded in batches via a queue.
         Hence, if you log a lot of data, you may experience slight delays in Neptune web application.

      .. note::

         Passing ``x`` coordinate as NaN or +/-inf causes this log entry to be ignored.
         Warning is printed to ``stdout``.


   .. method:: send_image(self, channel_name, x, y=None, name=None, description=None, timestamp=None)

      Log image data in Neptune.

      Alias for :meth:`~neptune.experiments.Experiment.log_image`


   .. method:: log_image(self, log_name, x, y=None, image_name=None, description=None, timestamp=None)

      Log image data in Neptune

      | If a log with provided ``log_name`` does not exist, it is created automatically.
      | If log exists (determined by ``log_name``), then new value is appended to it.

      :param log_name: The name of log, i.e. `bboxes`, `visualisations`, `sample_images`.
      :type log_name: :obj:`str`
      :param x: Depending, whether ``y`` parameter is passed:

                * ``y`` not passed: The value of the log (data-point). See ``y`` parameter.
                * ``y`` passed: Index of log entry being appended. Must be strictly increasing.
      :type x: :obj:`double`
      :param y: The value of the log (data-point). Can be one of the following types:

                * :obj:`PIL image`
                  `Pillow docs <https://pillow.readthedocs.io/en/latest/reference/Image.html#image-module>`_
                * :obj:`matplotlib.figure.Figure`
                  `Matplotlib 3.1.1 docs <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.figure.Figure.html>`_
                * :obj:`str` - path to image file
                * 2-dimensional :obj:`numpy.array` - interpreted as grayscale image
                * 3-dimensional :obj:`numpy.array` - behavior depends on last dimension

                    * if last dimension is 1 - interpreted as grayscale image
                    * if last dimension is 3 - interpreted as RGB image
                    * if last dimension is 4 - interpreted as RGBA image
      :type y: multiple types supported, optional, default is ``None``
      :param image_name: Image name
      :type image_name: :obj:`str`, optional, default is ``None``
      :param description: Image description
      :type description: :obj:`str`, optional, default is ``None``
      :param timestamp: Timestamp to be associated with log entry. Must be Unix time.
                        If ``None`` is passed, `time.time() <https://docs.python.org/3.6/library/time.html#time.time>`_
                        (Python 3.6 example) is invoked to obtain timestamp.
      :type timestamp: :obj:`time`, optional, default is ``None``

      .. rubric:: Example

      Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`:

      .. code:: python3

          # path to image file
          experiment.log_image('bbox_images', 'pictures/image.png')
          experiment.log_image('bbox_images', x=5, 'pictures/image.png')
          experiment.log_image('bbox_images', 'pictures/image.png', image_name='difficult_case')

          # PIL image
          img = PIL.Image.new('RGB', (60, 30), color = 'red')
          experiment.log_image('fig', img)

          # 2d numpy array
          array = numpy.random.rand(300, 200)*255
          experiment.log_image('fig', array)

          # 3d grayscale array
          array = numpy.random.rand(300, 200, 1)*255
          experiment.log_image('fig', array)

          # 3d RGB array
          array = numpy.random.rand(300, 200, 3)*255
          experiment.log_image('fig', array)

          # 3d RGBA array
          array = numpy.random.rand(300, 200, 4)*255
          experiment.log_image('fig', array)

          # matplotlib figure example 1
          from matplotlib import pyplot
          pyplot.plot([1, 2, 3, 4])
          pyplot.ylabel('some numbers')
          experiment.log_image('plots', plt.gcf())

          # matplotlib figure example 2
          from matplotlib import pyplot
          import numpy

          numpy.random.seed(19680801)
          data = numpy.random.randn(2, 100)

          figure, axs = pyplot.subplots(2, 2, figsize=(5, 5))
          axs[0, 0].hist(data[0])
          axs[1, 0].scatter(data[0], data[1])
          axs[0, 1].plot(data[0], data[1])
          axs[1, 1].hist2d(data[0], data[1])

          experiment.log_image('diagrams', figure)

      .. note::

         For efficiency, logs are uploaded in batches via a queue.
         Hence, if you log a lot of data, you may experience slight delays in Neptune web application.

      .. note::

         Passing ``x`` coordinate as NaN or +/-inf causes this log entry to be ignored.
         Warning is printed to ``stdout``.

      .. warning:: Only images up to 15MB are supported. Larger files will not be logged to Neptune.


   .. method:: send_artifact(self, artifact, destination=None)

      Save an artifact (file) in experiment storage.

      Alias for :meth:`~neptune.experiments.Experiment.log_artifact`


   .. method:: log_artifact(self, artifact, destination=None)

      Save an artifact (file) in experiment storage.

      :param artifact: A path to the file in local filesystem or IO object. It can be open
                       file descriptor or in-memory buffer like `io.StringIO` or `io.BytesIO`.
      :type artifact: :obj:`str` or :obj:`IO object`
      :param destination: A destination path.
                          If ``None`` is passed, an artifact file name will be used.
      :type destination: :obj:`str`, optional, default is ``None``

      :raises FileNotFound: When ``artifact`` file was not found.
      :raises StorageLimitReached: When storage limit in the project has been reached.

      .. rubric:: Example

      Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`:

      .. code:: python3

          # simple use
          experiment.log_artifact('images/wrong_prediction_1.png')

          # save file in other directory
          experiment.log_artifact('images/wrong_prediction_1.png', 'validation/images/wrong_prediction_1.png')

          # save file under different name
          experiment.log_artifact('images/wrong_prediction_1.png', 'images/my_image_1.png')


   .. method:: delete_artifacts(self, path)

      Removes an artifact(s) (file/directory) from the experiment storage.

      :param path: Path or list of paths to remove from the experiment's output
      :type path: :obj:`list` or :obj:`str`

      :raises FileNotFound: If a path in experiment artifacts does not exist.

      .. rubric:: Examples

      Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

      .. code:: python3

          experiment.delete_artifacts('forest_results.pkl')
          experiment.delete_artifacts(['forest_results.pkl', 'directory'])
          experiment.delete_artifacts('')


   .. method:: download_artifact(self, path, destination_dir=None)

      Download an artifact (file) from the experiment storage.

      Download a file indicated by ``path`` from the experiment artifacts and save it in ``destination_dir``.

      :param path: Path to the file to be downloaded.
      :type path: :obj:`str`
      :param destination_dir: The directory where the file will be downloaded.
                              If ``None`` is passed, the file will be downloaded to the current working directory.
      :type destination_dir: :obj:`str`

      :raises NotADirectory: When ``destination_dir`` is not a directory.
      :raises FileNotFound: If a path in experiment artifacts does not exist.

      .. rubric:: Examples

      Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

      .. code:: python3

          experiment.download_artifact('forest_results.pkl', '/home/user/files/')


   .. method:: download_sources(self, path=None, destination_dir=None)

      Download a directory or a single file from experiment's sources as a ZIP archive.

      Download a subdirectory (or file) ``path`` from the experiment sources and save it in ``destination_dir``
      as a ZIP archive. The name of an archive will be a name of downloaded directory (or file) with '.zip' extension.

      :param path: Path of a directory or file in experiment sources to be downloaded.
                   If ``None`` is passed, all source files will be downloaded.
      :type path: :obj:`str`
      :param destination_dir: The directory where the archive will be downloaded.
                              If ``None`` is passed, the archive will be downloaded to the current working directory.
      :type destination_dir: :obj:`str`

      :raises NotADirectory: When ``destination_dir`` is not a directory.
      :raises FileNotFound: If a path in experiment sources does not exist.

      .. rubric:: Examples

      Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

      .. code:: python3

          # Download all experiment sources to current working directory
          experiment.download_sources()

          # Download a single directory
          experiment.download_sources('src/my-module')

          # Download all experiment sources to user-defined directory
          experiment.download_sources(destination_dir='/tmp/sources/')

          # Download a single directory to user-defined directory
          experiment.download_sources('src/my-module', 'sources/')


   .. method:: download_artifacts(self, path=None, destination_dir=None)

      Download a directory or a single file from experiment's artifacts as a ZIP archive.

      Download a subdirectory (or file) ``path`` from the experiment artifacts and save it in ``destination_dir``
      as a ZIP archive. The name of an archive will be a name of downloaded directory (or file) with '.zip' extension.

      :param path: Path of a directory or file in experiment artifacts to be downloaded.
                   If ``None`` is passed, all artifacts will be downloaded.
      :type path: :obj:`str`
      :param destination_dir: The directory where the archive will be downloaded.
                              If ``None`` is passed, the archive will be downloaded to the current working directory.
      :type destination_dir: :obj:`str`

      :raises NotADirectory: When ``destination_dir`` is not a directory.
      :raises FileNotFound: If a path in experiment artifacts does not exist.

      .. rubric:: Examples

      Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

      .. code:: python3

          # Download all experiment artifacts to current working directory
          experiment.download_artifacts()

          # Download a single directory
          experiment.download_artifacts('data/images')

          # Download all experiment artifacts to user-defined directory
          experiment.download_artifacts(destination_dir='/tmp/artifacts/')

          # Download a single directory to user-defined directory
          experiment.download_artifacts('data/images', 'artifacts/')


   .. method:: _download_from_request(self, download_request, destination_dir, path)


   .. method:: reset_log(self, log_name)

      Resets the log.

      Removes all data from the log and enables it to be reused from scratch.

      :param log_name: The name of log to reset.
      :type log_name: :obj:`str`

      :raises ChannelDoesNotExist: When the log with name ``log_name`` does not exist on the server.

      .. rubric:: Example

      Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

      .. code:: python3

          experiment.reset_log('my_metric')

      .. note:: Check Neptune web application to see that reset charts have no data.


   .. method:: get_parameters(self)

      Retrieve parameters for this experiment.

      :returns: :obj:`dict` - dictionary mapping a parameter name to value.

      .. rubric:: Examples

      Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

      .. code:: python3

          exp_params = experiment.get_parameters()


   .. method:: get_properties(self)

      Retrieve User-defined properties for this experiment.

      :returns: :obj:`dict` - dictionary mapping a property key to value.

      .. rubric:: Examples

      Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`.

      .. code:: python3

          exp_properties = experiment.get_properties()


   .. method:: set_property(self, key, value)

      Set `key-value` pair as an experiment property.

      If property with given ``key`` does not exist, it adds a new one.

      :param key: Property key.
      :type key: :obj:`str`
      :param value: New value of a property.
      :type value: :obj:`obj`

      .. rubric:: Examples

      Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`:

      .. code:: python3

          experiment.set_property('model', 'LightGBM')
          experiment.set_property('magic-number', 7)


   .. method:: remove_property(self, key)

      Removes a property with given key.

      :param key: Key of property to remove.
      :type key: single :obj:`str`

      .. rubric:: Examples

      Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`:

      .. code:: python3

          experiment.remove_property('host')


   .. method:: get_hardware_utilization(self)

      Retrieve GPU, CPU and memory utilization data.

      Get hardware utilization metrics for entire experiment as a single
      `pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
      object. Returned DataFrame has following columns (assuming single GPU with 0 index):

          * `x_ram` - time (in milliseconds) from the experiment start,
          * `y_ram` - memory usage in GB,
          * `x_cpu` - time (in milliseconds) from the experiment start,
          * `y_cpu` - CPU utilization percentage (0-100),
          * `x_gpu_util_0` - time (in milliseconds) from the experiment start,
          * `y_gpu_util_0` - GPU utilization percentage (0-100),
          * `x_gpu_mem_0` - time (in milliseconds) from the experiment start,
          * `y_gpu_mem_0` - GPU memory usage in GB.

      | If more GPUs are available they have their separate columns with appropriate indices (0, 1, 2, ...),
        for example: `x_gpu_util_1`, `y_gpu_util_1`.
      | The returned DataFrame may contain ``NaN`` s if one of the metrics has more values than others.

      :returns: :obj:`pandas.DataFrame` - DataFrame containing the hardware utilization metrics.

      .. rubric:: Examples

      The following values denote that after 3 seconds, the experiment used 16.7 GB of RAM

          * `x_ram` = 3000
          * `y_ram` = 16.7

      Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`:

      .. code:: python3

          hardware_df = experiment.get_hardware_utilization()


   .. method:: get_numeric_channels_values(self, *channel_names)

      Retrieve values of specified metrics (numeric logs).

      The returned
      `pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
      contains 1 additional column `x` along with the requested metrics.

      :param \*channel_names: comma-separated metric names.
      :type \*channel_names: one or more :obj:`str`

      :returns: :obj:`pandas.DataFrame` - DataFrame containing values for the requested metrics.

                | The returned DataFrame may contain ``NaN`` s if one of the metrics has more values than others.

      .. rubric:: Example

      Invoking ``get_numeric_channels_values('loss', 'auc')`` returns DataFrame with columns
      `x`, `loss`, `auc`.

      Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`:

      .. code:: python3

          batch_channels = experiment.get_numeric_channels_values('batch-1-loss', 'batch-2-metric')
          epoch_channels = experiment.get_numeric_channels_values('epoch-1-loss', 'epoch-2-metric')

      .. note::

         It's good idea to get metrics with common temporal pattern (like iteration or batch/epoch number).
         Thanks to this each row of returned DataFrame has metrics from the same moment in experiment.
         For example, combine epoch metrics to one DataFrame and batch metrics to the other.


   .. method:: _start(self, upload_source_entries=None, abort_callback=None, logger=None, upload_stdout=True, upload_stderr=True, send_hardware_metrics=True, run_monitoring_thread=True, handle_uncaught_exceptions=True)


   .. method:: stop(self, exc_tb=None)

      Marks experiment as finished (succeeded or failed).

      :param exc_tb: Additional traceback information
                     to be stored in experiment details in case of failure (stacktrace, etc).
                     If this argument is ``None`` the experiment will be marked as succeeded.
                     Otherwise, experiment will be marked as failed.
      :type exc_tb: :obj:`str`, optional, default is ``None``

      .. rubric:: Examples

      Assuming that `experiment` is an instance of :class:`~neptune.experiments.Experiment`:

      .. code:: python3

          # Marks experiment as succeeded
          experiment.stop()

          # Assuming 'ex' is some exception,
          # it marks experiment as failed with exception info in experiment details.
          experiment.stop(str(ex))


   .. method:: __enter__(self)


   .. method:: __exit__(self, exc_type, exc_val, exc_tb)


   .. method:: __str__(self)

      Return str(self).


   .. method:: __repr__(self)

      Return repr(self).


   .. method:: __eq__(self, o)

      Return self==value.


   .. method:: __ne__(self, o)

      Return self!=value.


   .. staticmethod:: _convert_parameter_value(value, parameter_type)


   .. staticmethod:: _get_valid_x_y(x, y)

      The goal of this function is to allow user to call experiment.log_* with any of:
          - single parameter treated as y value
          - both parameters (named/unnamed)
          - single named y parameter
      If intended X-coordinate is provided, it is validated to be a float value


   .. method:: _send_channels_values(self, channels_with_values)


   .. method:: _get_channels(self, channels_names_with_types)


   .. method:: _get_channel(self, channel_name, channel_type, channel_namespace=ChannelNamespace.USER)


   .. method:: _find_channel(self, channel_name, channel_namespace)


   .. method:: _create_channel(self, channel_name, channel_type, channel_namespace=ChannelNamespace.USER)




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