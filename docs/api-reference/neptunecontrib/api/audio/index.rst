

:mod:`neptunecontrib.api.audio`
===============================

.. py:module:: neptunecontrib.api.audio


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   neptunecontrib.api.audio.log_audio


.. function:: log_audio(path_to_file, audio_name=None, experiment=None)

   Logs audio file to 'artifacts/audio' with player.

   Logs audio file to the 'artifacts/audio' in the experiment, where you can play it directly from the browser.
   You can also download raw audio file to the local machine.
   Just use "three vertical dots" located to the right from the player.

   :param path_to_file: Path to audio file.
   :type path_to_file: :obj:`str`
   :param audio_name: Name to be displayed in artifacts/audio.
                      | If `None`, file name is used.
   :type audio_name: :obj:`str`, optional, default is ``None``
   :param experiment:
                      | For advanced users only. Pass Neptune
                        `Experiment <https://docs.neptune.ai/neptune-client/docs/experiment.html#neptune.experiments.Experiment>`_
                        object if you want to control to which experiment data is logged.
                      | If ``None``, log to currently active, and most recent experiment.
   :type experiment: :obj:`neptune.experiments.Experiment`, optional, default is ``None``

   .. rubric:: Example

   .. code:: python3

       log_audio('audio-file.wav')
       log_audio('/full/path/to/some/other/audio/file.mp3')
       log_audio('/full/path/to/some/other/audio/file.mp3', 'my_audio')

   .. note::

      Check out how the logged audio file looks in Neptune:
      `here <https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-1485/artifacts?path=audio%2F>`_.



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