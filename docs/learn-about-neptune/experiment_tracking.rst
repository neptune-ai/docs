Experiment Tracking
===================

Neptune was designed to let you track whatever you need, wherever you run it.

Supported programming languages and platforms
---------------------------------------------

There are various ways of connecting Neptune to your experiments.

.. note::
    You connect Neptune directly from your scripts - No CLI, or third-party platform is needed.

- **Python neptune-client**: Python is the primary language of Neptune. After package installation, you simply paste a snippet on top of your scripts and start logging.

    For more information, see:


- **Jupyter notebook extension**: (package: neptune-notebooks): With this extension, you can version all the exploratory work or experimentation done in Jupyter notebooks. .ipynb checkpoints are sent to the Neptune server and can be commented, shared, compared and downloaded by anyone on your project.

    For more information, see `Using Jupyter Notebooks in Neptune <../notebooks/introduction.html>`_.

- **R via reticulate**: You can track experiments run in R as well. It can be installed via CRAN package manager and used as any R package.

    For more information, see `R support <../integrations/r-support.html>`_.

- **Any other programming language**: For any other language, you can use a certain “protocol” and save your experiment data to a .json file which can be synced with the Neptune server. It lets you log any experiment, no matter how complicated, multi-staged or multi-lingual.

    You write a JSON file and save it to disk. Then, you can use the method that we provide to parse this JSON file and upload as an experiment.

    For more information, see:


Main Python Objects
-------------------
.. I want to use the name of the API - not "Python". What is best to call it?

There are several key objects in the client library that let you interact with your Neptune projects or experiment data:

- `Project <../neptune-client/docs/project.html>`_: This is the Neptune project to which you want to log things. You need to create it in the application. This is a place where you can create experiments. You can create new ones and update or download information from the existing one.

- `Experiment <../neptune-client/docs/experiment.html>`_:  This is an object to which you log any piece of information you consider to be important during your run. Interaction with the experiment feels similar to interacting with a Singleton dictionary object. Neptune gives you all the freedom: You simply “attach” metrics, images, text and everything else to particular names and those objects are sent to the application. You can have one or multiple experiments in one script. You can reinstantiate the experiments you have created in the past and update them.

- `Session <../neptune-client/docs/session.html>`_: When you are creating a Neptune session you identify yourself (with an API token) so that the client knows which projects you have access to.

    - ``backend`` argument in `neptune.init() <../neptune-client/docs/neptune.html#neptune.init>`_: For the `backend` argument of the `neptune.init()` function, you specify where your experiment data should go.

    - ``backend`` argument in `neptune.sessions.Session() <../neptune-client/docs/session.html#neptune.sessions.Session>`_: This mode lets you run experiments while disconnected from the Neptune backend. It's useful, for example, when you want to quickly check something and not send data to Neptune. For the `backend` argument of the `neptune.sessions.Session()` function, you specify where your experiment data should go.

    - `Hosted <../neptune-client/docs/hosted.html>`_: In this mode, the Neptune backend is fully connected to your experiment execution and you send information to the application. You need to have an account created to do that.

- `neptune.init() <../neptune-client/docs/neptune.html#neptune.init>`_: A utility that creates a Session and fetches Project information from Neptune.


Logging to Neptune
------------------

There are various object types that you can log to Neptune. Some of them are logged automatically, some you need to specify explicitly. There is a place in the UI associated with every logging object type defined below.

- **Parameters**: You can log your experiment hyperparameters like learning rate, tree depth or regularization by passing a dictionary. Regardless of how you like to pass your hyperparameters: .yaml config, CLI + argparse or global variables Neptune lets you track them. Neptune parameters are immutable: you cannot change them later.

    For more information, see:

- **Metrics**: You can log numerical values to Neptune. Those could be machine learning metrics like Accuracy or MSE, timing metrics like time of the forward pass or any other numerical value. You can log one value (i.e. final validation loss) or multiple values (accuracy after every training epoch). If more than one value is sent to the same channel then charts are automatically created. Simply tell Neptune what is the name of the channel and what value you want to send.

    For more information, see:

- **Text**: You can send text values like warning messages, current parameter values or anything else. It can be one value (i.e. “idea worked”) or multiple values (i.e. parameters after each hyperparameter sweep iteration). Simply tell Neptune what is the name of the channel and what value you want to send.

    For more information, see:


- **Images**: You can send image data like ROC AUC charts, object detection predictions after every epoch, or anything else.  It can be one image (i.e. test confusion matrix) or multiple images (i.e. validation predictions after every epoch). Simply tell Neptune the name of the channel and what images you want to send.  You must first save the image file on disk and then send the file to Neptune. The following image types are supported:

        - PIL
        - Matplotlib
        - Numpy
        - Saved image files (.png, .jpg, etc)

- **Artifacts**: You can send any data type as file artifact in Neptune. Those could be model binaries, validation predictions, model checkpoints or anything else. Simply tell Neptune which file you want to log.

- **Hardware consumption**: Neptune automatically saves your hardware consumption data if the psutil library has been installed. Hardware types are:

    - GPU utilization, info from nvidia-smi command, works both for single any multi GPU setups
    - GPU memory, info from nvidia-smi command, works both for single any multi GPU setups
    - CPU utilization
    - Memory

- **Terminal outputs**: Neptune automatically saves everything that is printed to your terminal and groups it into stdout (output) and stderr (error messages)

- **Properties**: You can log your experiment information like status, data version, or anything else as a name: value(text)  pair. Neptune properties are mutable: you can change them later.

- **Tags**: You can attach tags (text) to every experiment to make the experiment organization easier.

- **Code**: There are various ways to version your code in Neptune but in either way whenever you create your experiment the code will be versioned.

    - **Git**: Neptune automatically fetches your .git information like commit id or commit message.

    - **Code snapshots**: You can specify files, directories or use regexp to choose files you want to snapshot and log directly to Neptune.

    - **Notebook snapshots**: If you are running your experiments from jupyter notebooks and using the Neptune extension your .ipynb code will be automatically snapshots whenever you create an experiment.

- **Jupyter Notebook checkpoints**: You can version any analysis you do in Jupyter notebooks with neptune-notebooks extension.  With that, you can keep track of all your exploratory work by uploading notebook checkpoints, naming them and adding descriptions for every piece of work you find important.

- **Integrations**: We have created loggers for many machine learning frameworks so that you don’t have to implement them from the atomic logging functions mentioned before. See the list of integrations here.

Fetching experiments from Neptune
---------------------------------

Every piece of information that is logged to Neptune can be easily retrieved programmatically using the Query API. Additionally, all Notebook checkpoints that were logged can be downloaded directly into your Jupyter Notebook or Jupyter Lab using neptune-notebooks extension.
Query API
It lets you access the information that you logged to Neptune. It is useful when you want to explore experiment results in Jupyter notebooks, fetch information for CI/CD pipelines, or integrate Neptune with your internal dashboards.
Fetching Functionalities
Experiment dashboard
You can fetch the entire experiment dashboard or use filters to query only parts of it that you care about.
Single experiment
You can fetch existing experiments and then access information like parameters, metrics, properties or artifacts from that experiment.
You can update existing experiment information like metrics, properties or artifacts after they have finished.
Notebooks extension
Lets you download notebook checkpoints from notebooks previously logged to Neptune. Those could be yours or one of your teammates.