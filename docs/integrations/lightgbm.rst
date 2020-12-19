.. _integrations-lightgbm:

Neptune-LightGBM Integration
============================

|Run on Colab|

What will you get with this integration?
----------------------------------------

|lightgbm-tour|

|lightGBM| is a popular gradient boosting library. The integration with Neptune lets you log training and evaluation metrics and have them visualized in Neptune.
   
.. note::

    This integration is tested with ``neptune-client==0.4.129``, ``neptune-contrib==0.25.0``, and ``lightgbm==2.2.3``, but should also work with later versions of ``neptune-client`` and ``neptune-contrib``
	
.. _lightgbm-quickstart:

Quickstart
----------

This quickstart will show you how to:

* Install the necessary Neptune packages
* Log lightGBM metrics to and visualize them with Neptune

|Run on Colab|

.. _lightgbm-before-you-start-basic:

Before you start
^^^^^^^^^^^^^^^^

#. Ensure that you have ``Python 3.x`` and following libraries installed:

   * ``neptune-client>=0.4.129``. See |neptune-client|
   * ``neptune-contrib>=0.25.0``. See |neptune-contrib|
   * ``lightgbm==2.2.3``. See the |lightgbm-install|
   
   .. code-block:: bash
   	
      pip install --quiet lightgbm==2.2.3 neptune-client neptune-contrib[monitoring]

#. You also need minimal familiarity with lightGBM. Have a look at the |lightgbm-guide| guide to get started.

Step 1: Initialize Neptune
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python3

    import neptune

    neptune.init(api_token='ANONYMOUS', project_qualified_name='shared/showroom')

.. tip::

    You can also use your personal API token. Read more about how to :ref:`securely set the Neptune API token <how-to-setup-api-token>`.

Step 2: Create an Experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python3

	neptune.create_experiment(name='lightGBM-training')

This also creates a link to the experiment. Open the link in a new tab. 
The charts will currently be empty, but keep the window open. You will be able to see live metrics once logging starts.

Step 3: Pass ``neptune_monitor`` to ``lgb.train``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simply pass ``neptune_monitor`` to the callbacks argument of ``lgb.train``

.. code-block:: python3

    from neptunecontrib.monitoring.lightgbm import neptune_monitor

    gbm = lgb.train(params,
            lgb_train,
            num_boost_round = 500,
            valid_sets = [lgb_train, lgb_eval],
            valid_names = ['train','valid'],
            callbacks = [neptune_monitor()], # Just add this callback
           )

Step 4: Monitor your lightGBM training in Neptune
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Now you can switch to the Neptune tab which you had opened previously to watch the training live!

.. image:: ../_static/images/integrations/lightgbm_neptuneml.png
   :target: ../_static/images/integrations/lightgbm_neptuneml.png
   :alt: lightGBM neptune.ai integration
	   
|Run on Colab|

.. External Links

.. |Run on Colab| raw:: html

    <div class="run-on-colab">

        <a target="_blank" href="https://colab.research.google.com//github/neptune-ai/neptune-examples/blob/master/integrations/lightgbm/docs/Neptune-lightGBM.ipynb">
            <img width="50" height="50" src="https://neptune.ai/wp-content/uploads/colab_logo_120.png">
            <span>Run in Google Colab</span>
        </a>

        <a target="_blank" href="https://github.com/neptune-ai/neptune-examples/blob/master/integrations/lightgbm/docs/Neptune-lightGBM.py">
            <img width="50" height="50" src="https://neptune.ai/wp-content/uploads/GitHub-Mark-120px-plus.png">
            <span>View source on GitHub</span>
        </a>
        <a target="_blank" href="https://ui.neptune.ai/o/shared/org/showroom/e/SHOW-2228">
            <img width="50" height="50" src="https://gist.githubusercontent.com/kamil-kaczmarek/7ac1e54c3b28a38346c4217dd08a7850/raw/8880e99a434cd91613aefb315ff5904ec0516a20/neptune-ai-blue-vertical.png">
            <span>See example in Neptune</span>
        </a>
    </div>

.. |lightgbm-tour| raw:: html

	<div style="position: relative; padding-bottom: 53.65126676602087%; height: 0;">
		<iframe src="https://www.loom.com/embed/0ab27ecd5c584cdf9802c820c965358b" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;">
		</iframe>
	</div>
	
.. |lightGBM| raw:: html

    <a href="https://lightgbm.readthedocs.io/en/latest/" target="_blank">lightGBM</a>

.. |neptune-client| raw:: html

    <a href="https://github.com/neptune-ai/neptune-client" target="_blank">neptune-client</a>

.. |neptune-contrib| raw:: html

    <a href="https://github.com/neptune-ai/neptune-contrib" target="_blank">neptune-contrib</a>
	
.. |lightgbm-install| raw:: html

	<a href="https://github.com/microsoft/LightGBM/tree/master/python-package" target="_blank">lightGBM Installation Guide</a>

.. |lightgbm-guide| raw:: html

	<a href="https://lightgbm.readthedocs.io/en/latest/Python-Intro.html" target="_blank">lightGBM Quickstart</a>
