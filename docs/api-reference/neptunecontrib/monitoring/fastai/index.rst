

:mod:`neptunecontrib.monitoring.fastai`
=======================================

.. py:module:: neptunecontrib.monitoring.fastai


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   neptunecontrib.monitoring.fastai.LearnerCallback
   neptunecontrib.monitoring.fastai.NeptuneMonitor



.. py:class:: LearnerCallback


.. py:class:: NeptuneMonitor(learn=None, experiment=None, prefix='')

   Bases: :class:`fastai.basic_train.LearnerCallback`

   Logs metrics from the fastai learner to Neptune.

   Goes over the `last_metrics` and `smooth_loss` after each batch and epoch
   and logs them to appropriate Neptune channels.

   See the example experiment here
   https://ui.neptune.ai/neptune-ai/neptune-examples/e/NEP-493/charts.


   :param experiment: Neptune experiment.
   :type experiment: `neptune.experiments.Experiment`
   :param prefix: Prefix that should be added before the `metric_name`
                  and `valid_name` before logging to the appropriate channel.
                  Defaul is ''.
   :type prefix: str

   .. rubric:: Examples

   Prepare data::

       from fastai.vision import *
       path = untar_data(URLs.MNIST_TINY)

       data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []), bs=64)
       data.normalize(imagenet_stats)

       learn = cnn_learner(data, models.resnet18, metrics=accuracy)

       learn.lr_find()
       learn.recorder.plot()

   Now, create Neptune experiment, instantiate the monitor and pass
   it to callbacks::

       import neptune
       from neptunecontrib.monitoring.fastai import NeptuneMonitor

       neptune.init(qualified_project_name='USER_NAME/PROJECT_NAME')

       with neptune.create_experiment():
           learn = create_cnn(data, models.resnet18,
                              metrics=accuracy,
                              callbacks_fns=[NeptuneMonitor])
           learn.fit_one_cycle(20, 1e-2)

   .. note:: you need to have the fastai library installed on your computer to use this module.

   .. method:: on_epoch_end(self, **kwargs)


   .. method:: on_batch_end(self, last_loss, iteration, train, **kwargs)




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