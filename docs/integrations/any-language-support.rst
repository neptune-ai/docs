Any language support
====================

You can track experiments that you run in any language in Neptune by logging data to a ``.json`` files of a certain format and syncing that with Neptune.

Install packages
----------------
Install Python and pip package manager on your system.

.. note:: you need Python running on your system.

After that simply install |neptune-client| and |neptune-contrib|.

.. code:: 

    pip install neptune-client neptune-contrib --user


Log your experiment data to a ``.json`` file
--------------------------------------------

You can log anything you want as long as you can save it in a following format:

.. code:: json

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


Sync your ``.json`` file with Neptune
-------------------------------------

Now you can convert your ``.json`` file into a Neptune experiment by running:

.. code:: bash

    python -m neptunecontrib.sync.with_json \
       --api_token "ANONYMOUS" \
       --project_name shared/any-language-integration \
       --filepath experiment_data.json


Explore your experiment in Neptune
----------------------------------
Now you can watch your metrics, parameters and logs in Neptune!

Check out this |example experiment|.

.. image:: ../_static/images/any_language/any_language_monitoring.gif
   :target: ../_static/images/any_language/any_language_monitoring.gif
   :alt: Any language integration with Neptune


.. External links

.. |example experiment| raw:: html

    <a href="https://ui.neptune.ai/o/shared/org/any-language-integration/e/AN-2/charts" target="_blank">example experiment</a>


.. |neptune-client| raw:: html

    <a href="https://github.com/neptune-ai/neptune-client" target="_blank">neptune-client</a>

.. |neptune-contrib| raw:: html

    <a href="s://github.com/neptune-ai/neptune-contrib" target="_blank">neptune-contrib</a>
