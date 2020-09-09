

:mod:`neptunecontrib.monitoring.sacred`
=======================================

.. py:module:: neptunecontrib.monitoring.sacred


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   neptunecontrib.monitoring.sacred.NeptuneObserver



Functions
~~~~~~~~~

.. autoapisummary::

   neptunecontrib.monitoring.sacred._flatten_dict
   neptunecontrib.monitoring.sacred._str_dict_values


.. py:class:: NeptuneObserver(project_name, api_token=None, source_extensions=None)

   Bases: :class:`sacred.observers.RunObserver`

   Logs sacred experiment data to Neptune.

   Sacred observer that logs experiment metadata to neptune.
   The experiment data can be accessed and shared via web UI or experiment API.
   Check Neptune docs for more information https://docs.neptune.ai.

   :param project_name: project name in Neptune app
   :type project_name: str
   :param api_token: Neptune API token. If it is kept in the NEPTUNE_API_TOKEN environment
                     variable leave None here.
   :type api_token: str
   :param source_extensions: list of extensions that Neptune should treat as source files
                             extensions and send. If None is passed, Python file from which experiment was created will be uploaded.
                             Pass empty list ([]) to upload no files. Unix style pathname pattern expansion is supported.
                             For example, you can pass '*.py' to upload all python source files from the current directory.
                             For recursion lookup use '**/*.py' (for Python 3.5 and later). For more information see glob library.
   :type source_extensions: list(str)

   .. rubric:: Examples

   Create sacred experiment::

       from numpy.random import permutation
       from sklearn import svm, datasets
       from sacred import Experiment

       ex = Experiment('iris_rbf_svm')

   Add Neptune observer::

       from neptunecontrib.monitoring.sacred import NeptuneObserver
       ex.observers.append(NeptuneObserver(api_token='YOUR_LONG_API_TOKEN',
                                           project_name='USER_NAME/PROJECT_NAME'))

   Run experiment::

       @ex.config
       def cfg():
           C = 1.0
           gamma = 0.7

       @ex.automain
       def run(C, gamma, _run):
           iris = datasets.load_iris()
           per = permutation(iris.target.size)
           iris.data = iris.data[per]
           iris.target = iris.target[per]
           clf = svm.SVC(C, 'rbf', gamma=gamma)
           clf.fit(iris.data[:90],
                   iris.target[:90])
           return clf.score(iris.data[90:],
                            iris.target[90:])

   Go to the app and see the experiment. For example, https://ui.neptune.ai/jakub-czakon/examples/e/EX-341

   .. method:: started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id)


   .. method:: completed_event(self, stop_time, result)


   .. method:: interrupted_event(self, interrupt_time, status)


   .. method:: failed_event(self, fail_time, fail_trace)


   .. method:: artifact_event(self, name, filename, metadata=None, content_type=None)


   .. method:: resource_event(self, filename)


   .. method:: log_metrics(self, metrics_by_name, info)



.. function:: _flatten_dict(d, parent_key='', sep=' ')


.. function:: _str_dict_values(d)



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