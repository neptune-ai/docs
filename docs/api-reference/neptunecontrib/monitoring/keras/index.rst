

:mod:`neptunecontrib.monitoring.keras`
======================================

.. py:module:: neptunecontrib.monitoring.keras


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   neptunecontrib.monitoring.keras.NeptuneMonitor



.. py:class:: NeptuneMonitor(experiment=None, prefix='')

   Bases: :class:`tensorflow.keras.callbacks.Callback`

   Logs Keras metrics to Neptune.

   Goes over the `last_metrics` and `smooth_loss` after each batch and epoch
   and logs them to appropriate Neptune channels.

   See the example experiment here TODO

   :param experiment: `neptune.Experiment`, optional:
                      Neptune experiment. If not provided, falls back on the current
                      experiment.
   :param prefix: str, optional:
                  Prefix that should be added before the `metric_name`
                  and `valid_name` before logging to the appropriate channel.
                  Defaul is empty string ('').

   .. rubric:: Examples

   Now, create Neptune experiment, instantiate the monitor and pass
   it to callbacks::

       TODO update for keras

   .. note:: You need to have Keras or Tensorflow 2 installed on your computer to use this module.

   .. method:: _log_metrics(self, logs, trigger)


   .. method:: on_batch_end(self, batch, logs=None)

      A backwards compatibility alias for `on_train_batch_end`.


   .. method:: on_epoch_end(self, epoch, logs=None)

      Called at the end of an epoch.

      Subclasses should override for any actions to run. This function should only
      be called during TRAIN mode.

      :param epoch: Integer, index of epoch.
      :param logs: Dict, metric results for this training epoch, and for the
                   validation epoch if validation is performed. Validation result keys
                   are prefixed with `val_`.




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