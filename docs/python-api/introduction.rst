About neptune-client
--------------------

|neptune-client-github|
is an open source Python library that lets you integrate your Python scripts with Neptune so that you can more easily track and organize your experiments in the rich Neptune `dashboard <https://ui.neptune.ai/shared/onboarding/experiments>`_.

Once you have integrated with Neptune, you can also:

* Create experiments. `Example <https://ui.neptune.ai/USERNAME/example-project/e/HELLO-48/source-code?path=.&file=classification-example.py>`_.
* Manage running experiments. `Example <https://ui.neptune.ai/USERNAME/example-project/e/HELLO-48/source-code?path=.&file=classification-example.py>`_.
* Query experiments and projects. `Example <https://ui.neptune.ai/USERNAME/example-project/n/Experiments-analysis-with-Query-API-and-Seaborn-31510158-04e2-47a5-a823-1cd97a0d8fcd/91350522-2b98-482d-bc14-a6ff5c061b6b>`_.

The `Neptune Python Library reference <api-reference.html>`_ provides a complete description of these capabilities.

.. note:: You must register with |neptune| to use neptune-client.

.. _installation:

Installation
============

.. code:: bash

    pip install neptune-client

Once installed, add ``import neptune`` to your code to use neptune-client capabilities.

Example
=======

The following code creates a Neptune experiment in the project |onboarding| and logs *iteration* and *loss* metrics to Neptune in real time. It also showcases a common use case for Neptune client, that is, tracking progress of machine learning experiments.

.. code-block::

   import neptune

   neptune.init('shared/onboarding',
                api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiJiNzA2YmM4Zi03NmY5LTRjMmUtOTM5ZC00YmEwMzZmOTMyZTQifQ==')
   with neptune.create_experiment(name='hello-neptune'):
       neptune.append_tag('introduction-minimal-example')
       n = 117
       for i in range(1, n):
           neptune.log_metric('iteration', i)
           neptune.log_metric('loss', 1/i**0.5)
           neptune.log_text('magic values', 'magic value {}'.format(0.95*i**2))
       neptune.set_property('n_iterations', n)

.. note:: Save the code as ``main.py`` and run it using the command: ``python main.py``.

.. External links

.. |neptune-client-github| raw:: html

    <a href="https://github.com/neptune-ai/neptune-client" target="_blank">Neptune client</a>

.. |neptune| raw:: html

    <a href="https://neptune.ai/register" target="_blank">Neptune</a>

.. |onboarding| raw:: html

    <a href="https://ui.neptune.ai/shared/onboarding/experiments" target="_blank">shared/onboarding</a>

.. |github-issues| raw:: html

    <a href="https://github.com/neptune-ai/neptune-client/issues" target="_blank">GitHub issues</a>

.. |spectrum| raw:: html

    <a href="https://spectrum.chat/neptune-community" target="_blank">spectrum</a>
