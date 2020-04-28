Experiment Tracking
===================

Neptune was designed to let you track whatever you need, wherever you run it.

.. contents::
    :local:
    :depth: 1
    :backlinks: top

Supported Programming Languages and Platforms
---------------------------------------------

There are various ways of connecting Neptune to your experiments.


- **Python neptune-client**: Neptune provides an open-source library that lets you connect your experiments directly from your scripts.

    For more information, see `Python Library <../python-api/introduction.html>`_ and `Get Started <../python-api/tutorials/get-started.html>`_.

- **Jupyter Notebook extension**: (package: neptune-notebooks): With this extension, you can version all the exploratory work or experimentation done in Jupyter Notebooks. ``.ipynb`` checkpoints are sent to the Neptune server and can be commented, shared, compared and downloaded by anyone on your project.

    For more information, see `Using Jupyter Notebooks in Neptune <../notebooks/introduction.html>`_.

- **R via reticulate**: You can track experiments run in R, as well. An interface to Neptune can be installed via the `CRAN package manager <https://cran.r-project.org/web/packages/neptune/index.html>`_ and used just as you would any R package.

    For more information, see `R support <../integrations/r-support.html>`_.

- **Any other programming language**: For any other language, you can save your experiment data to a JSON file, which can be synced with the Neptune server. In this way, you can log any experiment, no matter how complicated, multi-staged or multi-lingual. Use Neptune API methods to parse the JSON file and upload it as an experiment.

    For more information, see `Sync experiments with Neptune via JSON file <https://neptune-contrib.readthedocs.io/user_guide/sync/with_json.html>`_.


Logging to Neptune
------------------

There are various object types that you can log to Neptune. Some of them are logged automatically; some you need to specify explicitly. There is a place in the UI associated with every logging object type defined below.

- **Parameters**: You can log your experiment hyperparameters, such as learning rate, tree depth or regularization by passing them in as a dictionary during experiment creation. Note that Neptune parameters are immutable - you cannot change them later.

    For more information, see the `create_experiment() <../neptune-client/docs/project.html#neptune.projects.Project.create_experiment>`_ method.

- **Metrics**: You can log numerical values to Neptune. These could be machine learning metrics like accuracy or MSE, timing metrics like time of the forward pass or any other numerical value. You can log one value (like final validation loss) or multiple values (accuracy after every training epoch). If more than one value is sent to the same log, then charts are automatically created. Simply tell Neptune the name of the log and what value you want to send. There are specific methods for `logging metrics <../neptune-client/docs/experiment.html#neptune.experiments.Experiment.log_metric>`_, `logging text <../neptune-client/docs/experiment.html#neptune.experiments.Experiment.log_text>`_, `logging images <../neptune-client/docs/experiment.html#neptune.experiments.Experiment.log_image>`_, and `setting properties <../neptune-client/docs/experiment.html#neptune.experiments.Experiment.set_property>`_.

    For more information, see the `neptune.experiments.Experiment <../neptune-client/docs/experiment.html#neptune.experiments.Experiment>`_ class methods.

- **Text**: You can send text values like warning messages, current parameter values or anything else. It can be one value (like “idea worked”) or multiple values (like parameters after each hyperparameter sweep iteration). Simply tell Neptune the name of the log and what value you want to send.

    For more information, see the `log_text() <../neptune-client/docs/experiment.html#neptune.experiments.Experiment.log_text>`_ method.

- **Images**: You can send image data like ROC AUC charts, object detection predictions after every epoch, or anything else.  It can be one image (like a test confusion matrix) or multiple images (like validation predictions after every epoch). Simply tell Neptune the name of the log and what images you want to send.  You must first save the image file on disk and then send the file to Neptune. The following image types are supported:

        - PIL
        - Matplotlib
        - Numpy
        - Image files (PNG, JPG, and so on) on disk

    For more information, see the `log_image() <../neptune-client/docs/experiment.html#neptune.experiments.Experiment.log_image>`_ method.



- **Artifacts**: You can send any data type as a file artifact in Neptune. It can be a model binary, validation prediction, model checkpoint or anything else. Simply tell Neptune which file you want to log. `Example <https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-48/artifacts>`_.
- **Hardware consumption**: Neptune automatically saves your hardware consumption data if the psutil library has been installed. Hardware types are:

    - GPU utilization, information from the `nvidia-smi` command - works both for single and multi-GPU setups.
    - GPU memory, information from the `nvidia-smi` command - works both for single and multi-GPU setups.
    - CPU utilization
    - Memory

    `Here is an example <https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-48/monitoring>`_.

- **Terminal outputs**: Neptune automatically saves everything that is printed to your terminal and groups it into stdout (output) and stderr (error messages). `Example <https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-48/monitoring>`_.

- **Properties**: You can log your experiment information like status, data version, or anything else as a name:value (text) pair. Neptune properties are mutable - you can change them later. See the lower section in `this example <https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-48/details>`_.

- **Tags**: You can attach tags (text) to every experiment to make the experiment organization easier. For more information, see `Organize experiments <../learn-about-neptune/ui.html#organize-experiments>`_.

- **Code**: The following methods are available for versioning your code in Neptune. Whatever method you use, whenever you create an experiment, the code is versioned.

    - **Git**: Neptune automatically fetches your Git information, like ``commit id`` or ``commit message``. If you have a Git repo (meaning, a `.git` in the directory from which you are starting an experiment), then Neptune automatically shows a Git reference in the experiment details. The same is true if the `.git` repo is above, in the directory tree from which you start an experiment. `Example <https://ui.neptune.ai/o/neptune-ai/org/fastai2-integration/e/FAI-3/details>`_.

    - **Code snapshots**: You can specify files, directories or use `regexp` to choose files you want to snapshot and log directly to Neptune. `Example <https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-48/source-code?path=.&file=classification-example.py>`_.

    - **Notebook snapshots**: If you are running your experiments from Jupyter Notebooks and are using the Neptune extension, your ``.ipynb`` code is automatically snapshot whenever you create an experiment. For more information, see `Using Jupyter Notebooks in Neptune <../notebooks/introduction.html>`_.

- **Jupyter Notebook checkpoints**: You can version any analysis you do in Jupyter Notebooks with the neptune-notebooks extension. The extension also lets you keep track of all your exploratory work by uploading Notebook checkpoints, naming them and adding descriptions for every piece of work you find important.

    For more information, see `Uploading and Downloading Notebook Checkpoints <../notebooks/introduction.html#uploading-and-downloading-notebook-checkpoints>`_.

- **Integrations**: We have created loggers for many machine learning frameworks so that you don’t have to implement them from the atomic logging functions mentioned above. Learn more about the `MLflow <https://docs.neptune.ai/integrations/mlflow.html#>`_ and `TensorBoard <https://docs.neptune.ai/integrations/tensorboard.html#>`_ or `Sacred <https://neptune-contrib.readthedocs.io/examples/observer_sacred.html>`_ integrations, for example.

Fetching Experiments from Neptune
---------------------------------

Every piece of information that is logged to Neptune can be easily retrieved programmatically using the `Query API <../python-api/query-api.html>`_.

Experiment dashboard
""""""""""""""""""""
You can fetch data on the Project level. One example is the `get_leaderboard() method <../neptune-client/docs/project.html#neptune.projects.Project.get_leaderboard>`_, which lets you fetch the entire experiment dashboard or use filters to query only parts of it that you care about.

Single experiment
"""""""""""""""""
Alternatively, you can fetch data relating to a specific experiment. One example is the `get_experiment() method <../neptune-client/docs/project.html#neptune.projects.Project.get_experiments>`_, which lets you fetch existing experiments and then access information like the parameters, metrics, properties or artifacts of that experiment.


If you are running experiments in Jupyter Notebooks, you can download all checkpoints from Notebooks that were previously logged to Neptune. These could be yours or one of your teammate's. See `Uploading and Downloading Notebook Checkpoints <../notebooks/introduction.html#uploading-and-downloading-notebook-checkpoints>`_.

Main Python Objects
-------------------
.. I want to use the name of the API - not "Python". What is best to call it?

There are several key objects in the client library that let you interact with your Neptune projects or experiment data:

- `Neptune <../neptune-client/docs/neptune.html>`_: A global object that provides the convenience of doing most of the logging using a single `neptune` global variable, similar to Numpy's ``import numpy as np`` statement - in Neptune, write ``import neptune``.

- `Project <../neptune-client/docs/project.html>`_: This is the Neptune project to which you want to log things. You need to create it in the application. This is a place where you can create experiments. You can create new ones and update or download information from the existing one.

- `Experiment <../neptune-client/docs/experiment.html>`_:  This is an object to which you log any piece of information you consider to be important during your run. Interaction with the experiment feels similar to interacting with a Singleton dictionary object. Neptune gives you all the freedom: You simply log metrics, images, text and everything else to particular names and those objects are sent to the application. You can have one or multiple experiments in one script. You can reinstantiate the experiments you have created in the past and update them.

- `Session <../neptune-client/docs/session.html>`_: When you are creating a Neptune session you identify yourself (with an API token) so that the client knows which projects you have access to.
