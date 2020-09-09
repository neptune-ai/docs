

:mod:`neptunecontrib.versioning.data`
=====================================

.. py:module:: neptunecontrib.versioning.data


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   neptunecontrib.versioning.data.log_data_version
   neptunecontrib.versioning.data.log_s3_data_version
   neptunecontrib.versioning.data.log_image_dir_snapshots


.. function:: log_data_version(path, prefix='', experiment=None)

   Logs data version of file or folder to Neptune

   For a path it calculates the hash and logs it along with the path itself as a property to Neptune experiment.
   Path to dataset can be a file or directory.

   :param path: path to the file or directory,
   :type path: str
   :param prefix: Prefix that will be added before 'data_version' and 'data_path'
   :type prefix: str
   :param experiment: if the data should be logged to a particular
                      neptune experiment it can be passed here. By default it is logged to the current experiment.
   :type experiment: neptune.experiemnts.Experiment or None

   .. rubric:: Examples

   Initialize Neptune::

       import neptune
       from neptunecontrib.versioning.data import log_data_version
       neptune.init('USER_NAME/PROJECT_NAME')

   Log data version from filepath::

       FILEPATH = '/path/to/data/my_data.csv'
       with neptune.create_experiment():
           log_data_version(FILEPATH)


.. function:: log_s3_data_version(bucket_name, path, prefix='', experiment=None)

   Logs data version of s3 bucket to Neptune

   For a bucket and path it calculates the hash and logs it along with the path itself as a property to
   Neptune experiment.
   Path is either the s3 bucket key to a file or the begining of a key (in case you use a "folder" structure).

   :param bucket_name: name of the s3 bucket
   :type bucket_name: str
   :param path: path to the file or directory on s3 bucket
   :type path: str
   :param prefix: Prefix that will be added before 'data_version' and 'data_path'
   :type prefix: str
   :param experiment: if the data should be logged to a particular
                      neptune experiment it can be passed here. By default it is logged to the current experiment.
   :type experiment: neptune.experiemnts.Experiment or None

   .. rubric:: Examples

   Initialize Neptune::

       import neptune
       from neptunecontrib.versioning.data import log_s3_data_version
       neptune.init('USER_NAME/PROJECT_NAME')

   Log data version from bucket::

       BUCKET = 'my-bucket'
       PATH = 'train_dir/'
       with neptune.create_experiment():
           log_s3_data_version(BUCKET, PATH)


.. function:: log_image_dir_snapshots(image_dir, channel_name='image_dir_snapshots', experiment=None, sample=16, seed=1234)

   Logs visual snapshot of the directory with image data to Neptune.

   For a given directory with images it logs a sample of images as figure to Neptune.
   If the `image_dir` specified contains multiple folders it will sample per folder and create
   multiple figures naming each figure with the folder name.
   See snapshots per class here https://ui.neptune.ai/jakub-czakon/examples/e/EX-95/channels.

   :param image_dir: path to directory with images.
   :type image_dir: str
   :param sample: number of images that should be sampled for plotting.
   :type sample: int
   :param channel_name: name of the neptune channel. Default is 'image_dir_snapshots'.
   :type channel_name: str
   :param experiment: if the data should be logged to a particular
                      neptune experiment it can be passed here. By default it is logged to the current experiment.
   :type experiment: neptune.experiemnts.Experiment or None
   :param seed: random state for the sampling of images.
   :type seed: int

   .. rubric:: Examples

   Initialize Neptune::

       import neptune
       from neptunecontrib.versioning.data import log_image_dir_snapshots
       neptune.init('USER_NAME/PROJECT_NAME')

   Log visual snapshot of image directory::

       PATH = 'train_dir/'
       with neptune.create_experiment():
           log_image_dir_snapshots(PATH)



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