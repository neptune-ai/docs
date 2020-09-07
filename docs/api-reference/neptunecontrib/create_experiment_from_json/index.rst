

:mod:`neptunecontrib.create_experiment_from_json`
=================================================

.. py:module:: neptunecontrib.create_experiment_from_json

.. autoapi-nested-parse::

   Syncs json file containg experiment data with Neptune project.

   You can run your experiment in any language, create a `.json` file
   that contains your hyper parameters, metrics, tags or properties and log that to Neptune.

   .. attribute:: filepath

      filepath to the `.json` file that contains experiment data. It can have
      ['tags', 'channels', 'properties', 'parameters', 'name', 'log_metric', 'log_image', 'log_artifact'] sections.
      You can pass it either as --filepath or -f.

      :type: str

   .. attribute:: project_name

      Full name of the project. E.g. "neptune-ai/neptune-examples",
      If you have PROJECT_NAME environment variable set to your Neptune project you can skip this parameter.
      You can pass it either as --project_name or -p.

      :type: str

   .. attribute:: neptune_api_token

      Neptune api token. If you have NEPTUNE_API_TOKEN environment
      variable set to your API token you can skip this parameter.
      You can pass it either as --neptune_api_token or -t.

      :type: str

   .. rubric:: Example

   Run the experiment and create experiment json in any language.
   For example, lets say your `experiment_data.json` is::

       {
         "name": "example",
         "description": "json tracking experiment",
         "params": {
           "lr": 0.1,
           "batch_size": 128,
           "dropount": 0.5
         },
         "properties": {
           "data_version": "1231ffwefef9",
           "data_path": "data/train.csv"
         },
         "tags": [
           "resnet",
           "no_preprocessing"
         ],
         "upload_source_files": [
           "run.sh"
         ],
         "log_metric": {
           "log_loss": {
             "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             "y": [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
           },
           "accuracy": {
             "x": [0, 4, 5, 8, 9],
             "y": [0.23, 0.47, 0.62, 0.89, 0.92]
           }
         },
         "log_text": {
           "hash": {
             "x": [0, 4, 5, 8, 9],
             "y": ["123123", "as32e132", "123sdads", "123asdasd", " asd324132a"]
           }
         },
         "log_image": {
           "diagnostic_charts": {
             "x": [0, 1, 2],
             "y": ["data/roc_auc_curve.png", "data/confusion_matrix.png"
             ]
           }
         },
         "log_artifact": ["data/model.pkl", "data/results.csv"]
       }


   Now you can sync your file with neptune::

       $ python -m neptunecontrib.create_experiment_from_json.py
           --api_token 'ey7123qwwskdnaqsojnd1ru0129e12e=='
           --project_name neptune-ai/neptune-examples
           --filepath experiment_data.json

   Checkout an example experiment here:
   https://ui.neptune.ai/o/shared/org/any-language-integration/e/AN-2/logs

   .. note::

      If you keep your neptune api token in the NEPTUNE_API_TOKEN environment variable
      you can skip the --api_token.
      If you keep your full neptune project name in the PROJECT_NAME environment variable
      you can skip the --project_name.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   neptunecontrib.create_experiment_from_json.main
   neptunecontrib.create_experiment_from_json.parse_args


.. function:: main(arguments)


.. function:: parse_args()


.. data:: args
   

   


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