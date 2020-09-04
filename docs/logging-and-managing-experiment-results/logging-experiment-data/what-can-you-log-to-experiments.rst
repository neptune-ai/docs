What you can log to experiments
===============================

There are various object types that you can log to Neptune. Some of them are logged automatically; some you need to specify explicitly. There is a place in the UI associated with every logging object type defined below.

- **Parameters**: You can log your experiment hyper-parameters, such as learning rate, tree depth or regularization by passing them in as a dictionary during experiment creation. Note that Neptune parameters are immutable - you cannot change them later.

    For more information, see the :meth:`~neptune.projects.Project.create_experiment` method.


- **Metrics**: You can log numerical values to Neptune. These could be machine learning metrics like accuracy or MSE, timing metrics like time of the forward pass or any other numerical value. You can log one value (like final validation loss) or multiple values (accuracy after every training epoch). If more than one value is sent to the same log, then charts are automatically created. Simply tell Neptune the name of the log and what value you want to send. There are specific methods for |logging metrics|,  |logging text|, |logging images|, |logging audio|, |logging interactive charts|, |logging custom html|, and |setting properties|.

    For more information, see the :class:`~neptune.experiments.Experiment` class methods.


- **Text**: You can send text values like warning messages, current parameter values or anything else. It can be one value (like “idea worked”) or multiple values (like parameters after each hyperparameter sweep iteration). Simply tell Neptune the name of the log and what value you want to send.

    For more information, see the :meth:`~neptune.experiments.Experiment.log_text` method.


- **Images**: You can send image data like ROC AUC charts, object detection predictions after every epoch, or anything else.  It can be one image (like a test confusion matrix) or multiple images (like validation predictions after every epoch). Simply tell Neptune the name of the log and what images you want to send.  You must first save the image file on disk and then send the file to Neptune. The following data types are supported:

        - PIL
        - Matplotlib
        - Numpy
        - Image files (PNG, JPG, and so on) on disk

    For more information, see the :meth:`~neptune.experiments.Experiment.log_image` method.

- **Artifacts**: You can send any data type as a file artifact in Neptune. It can be a model binary, validation prediction, model checkpoint or anything else. Simply tell Neptune which file you want to log. |artifacts|.

- **Hardware consumption**: Neptune automatically saves your hardware consumption data if the psutil library has been installed. Hardware types are:

    - GPU utilization, information from the `nvidia-smi` command - works both for single and multi-GPU setups.
    - GPU memory, information from the `nvidia-smi` command - works both for single and multi-GPU setups.
    - CPU utilization
    - Memory

    |Here is an example|.

- **Terminal outputs**: Neptune automatically saves everything that is printed to your terminal and groups it into stdout (output) and stderr (error messages). |monitoring|.

- **Properties**: You can log your experiment information like status, data version, or anything else as a name:value (text) pair. Neptune properties are mutable - you can change them later. See the lower section in |properties|.

- **Tags**: You can attach tags (text) to every experiment to make the experiment organization easier. For more information, see |Organize experiments|.

- **Code**: The following methods are available for versioning your code in Neptune. Whatever method you use, whenever you create an experiment, the code is versioned.

    - **Git**: Neptune automatically fetches your Git information, like ``commit id`` or ``commit message``. If you have a Git repo (meaning, a `.git` in the directory from which you are starting an experiment), then Neptune automatically shows a Git reference in the experiment details. The same is true if the `.git` repo is above, in the directory tree from which you start an experiment. |Example Git reference|.

    - **Code snapshots**: You can specify files, directories or use `regexp` to choose files you want to snapshot and log directly to Neptune. |Example|.

    - **Notebook snapshots**: If you are running your experiments from Jupyter Notebooks and are using the Neptune extension, your ``.ipynb`` code is automatically snapshot whenever you create an experiment. For more information, see |Using Jupyter Notebooks in Neptune|.

- **Jupyter Notebook checkpoints**: You can version any analysis you do in Jupyter Notebooks with the neptune-notebooks extension. The extension also lets you keep track of all your exploratory work by uploading Notebook checkpoints, naming them and adding descriptions for every piece of work you find important.

    For more information, see |Uploading and Downloading Notebook Checkpoints|.

- **Integrations**: We have created convenient integrations with many machine learning frameworks so that you don’t have to implement them from the atomic logging functions mentioned above.

    Learn |more about integrations here|, or study some examples: `Keras <../integrations/keras.html>`_, `PyTorch Lightning <../integrations/pytorch_lightning.html>`_, `XGBoost <../integrations/xgboost.html>`_, `Matplotlib <../integrations/matplotlib.html>`_.

.. External Links

.. |Here is an example| raw:: html

    <a href="https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-48/monitoring" target="_blank">Here is an example</a>

.. |CRAN package manager| raw:: html

    <a href="https://cran.r-project.org/web/packages/neptune/index.html" target="_blank">CRAN package manager</a>

.. |Sync experiments with Neptune using a JSON file| raw:: html

    <a href="/api-reference/neptunecontrib/create_experiment_from_json/index.html?highlight=json#module-neptunecontrib.create_experiment_from_json" target="_blank">Sync experiments with Neptune using a JSON file</a>

.. |artifacts| raw:: html

    <a href="https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-48/artifacts" target="_blank">See this example</a>

.. |monitoring| raw:: html

     <a href="https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-48/monitoring" target="_blank">See this example</a>

.. |properties| raw:: html

     <a href="https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-48/details" target="_blank">in this experiment</a>

.. |in this example| raw:: html

     <a href="https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-48/details" target="_blank">in this example</a>


.. |Example Git reference| raw:: html

     <a href="https://ui.neptune.ai/o/neptune-ai/org/fastai2-integration/e/FAI-3/details" target="_blank">Example Git reference</a>

.. |Example| raw:: html

    <a href="https://ui.neptune.ai/o/USERNAME/org/example-project/e/HELLO-48/source-code?path=.&file=classification-example.py" target="_blank">Example</a>

.. |logging metrics| raw:: html

     <a href="/api-reference/neptune/experiments/index.html?highlight=log_metric#neptune.experiments.Experiment.log_metric" target="_blank">logging metrics</a>

.. |logging text| raw:: html

     <a href="/api-reference/neptune/experiments/index.html?highlight=log_text#neptune.experiments.Experiment.log_text" target="_blank">logging text</a>

.. |logging images| raw:: html

     <a href="/api-reference/neptune/experiments/index.html?highlight=log%20image#neptune.experiments.Experiment.log_image" target="_blank">logging images</a>

.. |logging audio| raw:: html

     <a href="/api-reference/neptunecontrib/api/index.html?highlight=log%20audio#neptunecontrib.api.log_audio" target="_blank">logging audio</a>

.. |logging interactive charts| raw:: html

     <a href="/api-reference/neptunecontrib/api/index.html?highlight=log%20chart#neptunecontrib.api.log_chart" target="_blank">logging interactive charts</a>

.. |logging custom html| raw:: html

     <a href="/api-reference/neptunecontrib/api/index.html?highlight=log_html#neptunecontrib.api.log_html" target="_blank">logging custom html</a>

.. |setting properties| raw:: html

     <a href="/api-reference/neptune/index.html?highlight=set_property#neptune.set_property" target="_blank">setting properties</a>

.. |Organize experiments| raw:: html

     <a href="/use-cases/organize-experiments/index.html" target="_blank">Organize experiments</a>

.. |Using Jupyter Notebooks in Neptune| raw:: html

     <a href="/keep-track-of-jupyter-notebooks/index.html" target="_blank">Using Jupyter Notebooks in Neptune</a>

.. |Uploading and Downloading Notebook Checkpoints| raw:: html

     <a href="/keep-track-of-jupyter-notebooks/index.html" target="_blank">Uploading and Downloading Notebook Checkpoints</a>

.. |more about integrations here| raw:: html

     <a href="/integrations/index.html" target="_blank">more about integrations here</a>
