.. _execution-environments-index:

Run Neptune anywhere
====================

You can run your experiments in any environment and log them to Neptune.
There are just two things you need to do:

- install Neptune client

    .. code:: bash

        pip install neptune-client

- add logging snippet to your scripts

    .. code:: python

        import neptune

        neptune.init(api_token='', # use your api token
                     project_qualified_name='') # use your project name
        neptune.create_experiment('my-experiment')
        # training logic
        neptune.log_metric('accuracy', 0.92)

.. note::

    if you are not using Python don't worry.
    Read how to :ref:`use Neptune with R <integrations-r>` or :ref:`with any other language <integrations-any-language>`


.. _execution-use-environment-variables:

Environment variables
---------------------

Instead of passing `api_token` and `project_qualified_name` directly to `neptune.init()` you can use environment variables:

- `NEPTUNE_API_TOKEN`: where you put your Neptune API token

    .. code:: bash

        NEPTUNE_API_TOKEN='YOUR_API_TOKEN'

    .. note::

        It is recommended to keep your API token in the environment variable for security reasons.
        To see how to get api token read :ref:`how to get Neptune API token <how-to-setup-api-token>`.

- `NEPTUNE_PROJECT`: where you put your project qualified name

    .. code:: bash

        NEPTUNE_PROJECT='YOUR_QUALIFIED_PROJECT_NAME'

    .. note::

        Remember that project qualified name has two parts 'WORKSPACE/PROJECT_NAME' for example 'neptune-ai/credit-default-prediction'.
        Read :ref:`how to create a project here <create-project>`.

See how you can run Neptune with:

Different languages:

- :ref:`Python <execution-python>`
- :ref:`R <integrations-r>`
- :ref:`Any other language <integrations-any-language>`

Various Notebook flavours:

- :ref:`Jupyter Lab and Juypter Notebooks <execution-jupyter-notebooks>`
- :ref:`Google Colab <integrations-google-colab>`
- :ref:`Deepnote <execution-deepnote>`
- :ref:`Amazon Sagemaker <integrations-amazon-sagemaker>`

.. toctree::
   :hidden:
   :maxdepth: 1

   Python <python-support.rst>
   R <r-support.rst>
   Any other language <any-language-support.rst>
   Jupyter Lab and Jupyter Notebook <jupyter-notebooks.rst>
   Google Colab <google-colab.rst>
   Deepnote <deepnote.rst>
   Amazon SageMaker <amazon_sagemaker.rst>

.. External links

.. |forum| raw:: html

    <a href="https://community.neptune.ai/c/feature-requests" target="_blank">feature request</a>

.. |neptune-client| raw:: html

    <a href="/api-reference/neptune/index.html" target="_blank">neptune-client</a>