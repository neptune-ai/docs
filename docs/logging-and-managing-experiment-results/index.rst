Logging and managing experiment results
=======================================

|Youtube Video|

|neptune-client-github|
is an open source Python library that lets you integrate your Python scripts with Neptune so that you can more easily track and organize your experiments in the rich Neptune |dashboard|.

Once you have integrated with Neptune, you can also:

* Create experiments. |Example1|.
* Manage running experiments. |Example2|.
* Fetch experiment and project data. |Example3|.

**Example**

The following code creates a Neptune experiment in the project |onboarding|. Name (*Python str*) and parameters (*Python dict*) are added to the experiment in the ``create_experiment()`` method. The code logs ``iteration``, ``loss`` and ``text_info`` metrics to Neptune in real time, using three dedicated methods. It also showcases a common use case for Neptune client, that is, tracking progress of machine learning experiments.


.. code-block::

    import neptune
    import numpy as np

    # select project
    neptune.init('shared/onboarding',
                 api_token='ANONYMOUS')

    # define parameters
    PARAMS = {'decay_factor': 0.7,
              'n_iterations': 117}

    # create experiment
    neptune.create_experiment(name='quick_start_example',
                              params=PARAMS)

    # log some metrics
    for i in range(1, PARAMS['n_iterations']):
        neptune.log_metric('iteration', i)
        neptune.log_metric('loss', PARAMS['decay_factor']/i**0.5)
        neptune.log_text('text_info', 'some value {}'.format(0.95*i**2))

    # add tag to the experiment
    neptune.append_tag('quick_start')

    # log some images
    for j in range(5):
        array = np.random.rand(10, 10, 3)*255
        array = np.repeat(array, 30, 0)
        array = np.repeat(array, 30, 1)
        neptune.log_image('mosaics', array)

.. note:: Save the code as ``main.py`` and run it using the command: ``python main.py``.

.. External links

.. |Youtube Video| raw:: html

    <iframe width="720" height="420" src="https://www.youtube.com/embed/w9S5srkfSI4" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

.. |neptune-client-github| raw:: html

    <a href="https://github.com/neptune-ai/neptune-client" target="_blank">Neptune client</a>

.. |Neptune| raw:: html

    <a href="https://neptune.ai/register" target="_blank">Neptune</a>

.. |onboarding| raw:: html

    <a href="https://ui.neptune.ai/shared/onboarding/experiments" target="_blank">shared/onboarding</a>

.. |github-issues| raw:: html

    <a href="https://github.com/neptune-ai/neptune-client/issues" target="_blank">GitHub issues</a>

.. |spectrum| raw:: html

    <a href="https://spectrum.chat/neptune-community" target="_blank">spectrum</a>

.. |Example1| raw:: html

    <a href="https://ui.neptune.ai/USERNAME/example-project/e/HELLO-48/source-code?path=.&file=classification-example.py" target="_blank">Example</a>

.. |Example2| raw:: html

    <a href="https://ui.neptune.ai/USERNAME/example-project/e/HELLO-48/source-code?path=.&file=classification-example.py" target="_blank">Example</a>

.. |Example3| raw:: html

    <a href="https://ui.neptune.ai/USERNAME/example-project/n/Experiments-analysis-with-Query-API-and-Seaborn-31510158-04e2-47a5-a823-1cd97a0d8fcd/91350522-2b98-482d-bc14-a6ff5c061b6b" target="_blank">Example</a>

.. |dashboard| raw:: html

    <a href="https://ui.neptune.ai/shared/onboarding/experiments" target="_blank">dashboard</a>


.. toctree::
   :maxdepth: 1

   Logging experiment data <logging-experiment-data/index.rst>
   Updating existing experiments <updating-existing-experiment/index.rst>
   Downloading experiment data programmatically <downloading-experiment-data-programmatically/index.rst>
