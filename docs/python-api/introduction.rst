Introduction
============
|neptune-client-github| is open source Python library that allows you to integrate your Python scripts with Neptune. Neptune client supports handful of use cases:

* creating and tracking experiments
* managing running experiment
* querying experiments and projects (search/download)

.. note:: Make sure to register to |neptune|, to use it.

Neptune implements client-server architecture. Because of that you can log and access your results from many different devices:

* laptops
* cluster of machines
* cloud services

.. image:: ../_static/images/python_api/server_client_arch.png
   :target: ../_static/images/python_api/server_client_arch.png
   :alt: basic architecture

.. _installation:

Installation
------------
.. code:: bash

    pip install neptune-client

Once installed, ``import neptune`` in your code to use it.

Example
-------
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

.. note:: Install :ref:`neptune-client <installation>`, save the code as ``main.py``, and run like this: ``python main.py``.

Example above creates Neptune experiment in the project: |onboarding| and logs *iteration* and *loss* metrics to Neptune in real time. It also presents common use case for Neptune client, that is tracking progress of machine learning experiments.

Questions and feature requests
------------------------------
If you like to suggest feature or improvement simply drop an issue on |github-issues|, or ask us on the |spectrum| chat.

.. External links

.. |neptune-client-github| raw:: html

    <a href="https://github.com/neptune-ml/neptune-client" target="_blank">Neptune client</a>

.. |neptune| raw:: html

    <a href="https://neptune.ml/register" target="_blank">Neptune</a>

.. |onboarding| raw:: html

    <a href="https://ui.neptune.ml/shared/onboarding/experiments" target="_blank">shared/onboarding</a>

.. |github-issues| raw:: html

    <a href="https://github.com/neptune-ml/neptune-client/issues" target="_blank">GitHub issues</a>

.. |spectrum| raw:: html

    <a href="https://spectrum.chat/neptune-community" target="_blank">spectrum</a>
