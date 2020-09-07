

:mod:`neptunecontrib.api.html`
==============================

.. py:module:: neptunecontrib.api.html


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   neptunecontrib.api.html.log_html


.. function:: log_html(name, html, experiment=None)

   Logs html to neptune.

   HTML is logged to Neptune as an artifact with path html/{name}.html

   :param name: | Name of the chart (without extension) that will be used as a part of artifact's destination.
   :type name: :obj:`str`
   :param html_body: | HTML string that is logged and rendered as HTML.
   :type html_body: :obj:`str`
   :param experiment:
                      | For advanced users only. Pass Neptune
                        `Experiment <https://docs.neptune.ai/neptune-client/docs/experiment.html#neptune.experiments.Experiment>`_
                        object if you want to control to which experiment data is logged.
                      | If ``None``, log to currently active, and most recent experiment.
   :type experiment: :obj:`neptune.experiments.Experiment`, optional, default is ``None``

   .. rubric:: Examples

   Start an experiment::

       import neptune

       neptune.init(api_token='ANONYMOUS',
                    project_qualified_name='shared/showroom')
       neptune.create_experiment(name='experiment_with_html')

   Create an HTML string::

       html = "<button type='button',style='background-color:#005879; width:300px; height:200px; font-size:30px'>                  <a style='color: #ccc', href='https://docs.neptune.ai'> Take me back to the docs!!<a> </button>"

   Log it to Neptune::

        from neptunecontrib.api import log_html

        log_html('go_to_docs_button', html)

   Check out how the logged table looks in Neptune:
   https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-988/artifacts?path=html%2F&file=button_example.html



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