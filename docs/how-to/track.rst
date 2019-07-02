Track
=====

How to log keras metrics?
-------------------------
I have a training script written in `keras <https://keras.io>`_. How do I adjust it to log metrics to Neptune?

Solution
^^^^^^^^
**Step 1**

Say your training script looks like this:

.. code-block::

   import keras
   from keras import backend as K

   mnist = keras.datasets.mnist
   (x_train, y_train),(x_test, y_test) = mnist.load_data()
   x_train, x_test = x_train / 255.0, x_test / 255.0

   model = keras.models.Sequential([
     keras.layers.Flatten(),
     keras.layers.Dense(512, activation=K.relu),
     keras.layers.Dropout(0.2),
     keras.layers.Dense(10, activation=K.softmax)
   ])
   model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

   model.fit(x_train, y_train, epochs=5)

**Step 2**

Now let's use Keras Callback

.. code-block::

   from keras.callbacks import Callback

   class NeptuneMonitor(Callback):
       def on_epoch_end(self, epoch, logs={}):
           innovative_metric = logs['acc'] - 2 * logs['loss']
           neptune.send_metric('innovative_metric', epoch, innovative_metric)

**Step 3**

Instantiate it and add it to your callbacks list:

.. code-block::

   with neptune.create_experiment():
       neptune_monitor = NeptuneMonitor()
       model.fit(x_train, y_train, epochs=5, callbacks=[neptune_monitor])

All your metrics are now logged to Neptune:

.. image:: ../_static/images/how-to/ht-log-keras-1.png
   :target: ../_static/images/how-to/ht-log-keras-1.png
   :alt: image

How to log PyTorch metrics?
---------------------------
I have a training script written in `PyTorch <https://pytorch.org>`_. How do I adjust it to log metrics to Neptune?

Solution
^^^^^^^^
Say your training script looks like this:

.. code-block::

   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   import torch.optim as optim
   from torchvision import datasets, transforms

   DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ITERATIONS = 10000

   class Net(nn.Module):
       def __init__(self):
           super(Net, self).__init__()
           self.conv1 = nn.Conv2d(1, 20, 5, 1)
           self.conv2 = nn.Conv2d(20, 50, 5, 1)
           self.fc1 = nn.Linear(4*4*50, 500)
           self.fc2 = nn.Linear(500, 10)

       def forward(self, x):
           x = F.relu(self.conv1(x))
           x = F.max_pool2d(x, 2, 2)
           x = F.relu(self.conv2(x))
           x = F.max_pool2d(x, 2, 2)
           x = x.view(-1, 4*4*50)
           x = F.relu(self.fc1(x))
           x = self.fc2(x)
           return F.log_softmax(x, dim=1)

   train_loader = torch.utils.data.DataLoader(
       datasets.MNIST('../data',
                      train=True,
                      download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))])
                      ),
       batch_size=64,
       shuffle=True)

   model = Net().to(DEVICE)

   optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

   for batch_idx, (data, target) in enumerate(train_loader):
       data, target = data.to(DEVICE), target.to(DEVICE)
       optimizer.zero_grad()
       output = model(data)
       loss = F.nll_loss(output, target)
       loss.backward()
       optimizer.step()

       if batch_idx == ITERATIONS:
           break

Add a snippet to the training loop, that sends your loss or metric to Neptune:

.. code-block::

   import neptune

   neptune.init('shared/onboarding')
   neptune.create_experiment()
   ...
   for batch_idx, (data, target) in enumerate(train_loader):
       ...
       neptune.send_metric('batch_loss', batch_idx, loss.data.cpu().numpy())

Your loss is now logged to Neptune:

.. image:: ../_static/images/how-to/ht-log-pytorch-1.png
   :target: ../_static/images/how-to/ht-log-pytorch-1.png
   :alt: image

How to log LightGBM metrics?
----------------------------
I have a training script written in `LightGBM <https://lightgbm.readthedocs.io>`_. How do I adjust it to log metrics to Neptune?

Solution
^^^^^^^^
Say your training script looks like this:

.. code-block::

   import lightgbm as lgb
   from sklearn.model_selection import train_test_split
   from sklearn.datasets import load_wine

   data = load_wine()

   X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.1)
   lgb_train = lgb.Dataset(X_train, y_train)
   lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

   params = {
       'boosting_type': 'gbdt',
       'objective': 'multiclass',
       'num_class': 3,
       'num_leaves': 31,
       'learning_rate': 0.05,
       'feature_fraction': 0.9,
   }

   gbm = lgb.train(params,
                   lgb_train,
                   num_boost_round=500,
                   valid_sets=[lgb_train, lgb_eval],
                   valid_names=['train','valid'],
                   )

Now, you need to use lightGBM callbacks to pass log metrics to Neptune:

**Step 1**

Take this callback:

.. code-block::

   import neptune

   neptune.init('shared/onboarding')
   neptune.create_experiment()

   def neptune_monitor():
       def callback(env):
           for name, loss_name, loss_value, _ in env.evaluation_result_list:
               neptune.send_metric('{}_{}'.format(name, loss_name), x=env.iteration, y=loss_value)
       return callback

**Step 2**

Pass it to ``lgb.train`` object via ``callbacks`` parameter:

.. code-block::

   gbm = lgb.train(params,
                   lgb_train,
                   num_boost_round=500,
                   valid_sets=[lgb_train, lgb_eval],
                   valid_names=['train','valid'],
                   callbacks=[neptune_monitor()],
                   )

All your metrics are now logged to Neptune

.. image:: ../_static/images/how-to/ht-log-lightgbm-1.png
   :target: ../_static/images/how-to/ht-log-lightgbm-1.png
   :alt: image

How to log matplotlib figure to Neptune?
----------------------------------------
How to log charts generated in `matplotlib <https://matplotlib.org/>`_, like confusion matrix or distribution in Neptune?

Solution
^^^^^^^^
**Step 1**

Create matplotlib figure

.. code-block::

   import matplotlib.pyplot as plt
   import seaborn as sns

   fig = plt.figure()
   sns.distplot(np.random.random(100))

**Step 2**

Convert your matplotlib figure object into PIL image.

For example you could use the following function, taken from `here <http://www.icare.univ-lille1.fr/wiki/index.php/How_to_convert_a_matplotlib_figure_to_a_numpy_array_or_a_PIL_image>`_, and adjusted slightly:

.. code-block::

   import numpy as np
   from PIL import Image

   def fig2pil(fig):
       fig.canvas.draw()

       w, h = fig.canvas.get_width_height()
       buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
       buf.shape = (w, h, 4)
       buf = np.roll(buf, 3, axis=2)

       w, h, d = buf.shape
       return Image.frombytes("RGBA", (w, h), buf.tostring())

   pil_image = fig2pil(fig)

**Step 3**

Send it to Neptune!

.. code-block::

   neptune.create_experiment()
   neptune.send_image('distplot', pil_image)

**Step 5**

Explore it in the browser:

.. image:: ../_static/images/how-to/ht-matplotlib-1.png
   :target: ../_static/images/how-to/ht-matplotlib-1.png
   :alt: image

.. image:: ../_static/images/how-to/ht-matplotlib-2.png
   :target: ../_static/images/how-to/ht-matplotlib-2.png
   :alt: image

How to save experiment output?
------------------------------
I can run my experiment but I am struggling to save the model weights and the ``csv`` file with the results when it completes. How can I do that in Neptune?

Solution
^^^^^^^^
Save everything as you go! For example:

.. code-block::

   with neptune.create_experiment() as exp:
       exp.send_artifact('/path/to/model_weights.h5')
       ...
       exp.send_artifact('/path/to/results.csv')

Your results will be available for you to download in the ``Output`` section of your experiment.

.. image:: ../_static/images/how-to/ht-output-download-1.png
   :target: ../_static/images/how-to/ht-output-download-1.png
   :alt: image

How specify experiment parameters?
----------------------------------
I saw that Neptune logs experiment parameters.

.. image:: ../_static/images/how-to/ht-specify-params-1.png
   :target: ../_static/images/how-to/ht-specify-params-1.png
   :alt: image

But I don't know how to specify parameters for my experiments.

Solution
^^^^^^^^
You define your parameters at experiment creation, like this:

.. code-block::

   import neptune

   # This function assumes that NEPTUNE_API_TOKEN environment variable is defined.
   neptune.init('username/my_project')

   # check params argument
   with neptune.create_experiment(name='first-pytorch-ever',
                                  params={'dropout': 0.3,
                                          'lr': 0.01,
                                          'nr_epochs': 10}):
   # your training script

Where ``params`` is standard Python dict.

How to log images to Neptune?
-----------------------------
I generate model predictions after every epoch. How can I log them as images to Neptune?

Solution
^^^^^^^^
**Log single image to Neptune**

Create PIL image that you want to log. For example:

.. code-block::

   import imgaug as ia
   from PIL import Image

   img = ia.quokka()
   img_pil = Image.fromarray(img)

Log it to Neptune:

.. code-block::

   import neptune

   # This function assumes that NEPTUNE_API_TOKEN environment variable is defined.
   neptune.init(project_qualified_name='shared/onboarding')

   with neptune.create_experiment() as exp:
       exp.send_image('quokka', img_pil)

As a result, quokka image is associated with the experiment

.. image:: ../_static/images/how-to/ht-img-channel-1.png
   :target: ../_static/images/how-to/ht-img-channel-1.png
   :alt: image

**Log multiple images to neptune**

You can log images in a loop. For example, you can augment your image and log it to Neptune:

.. code-block::

   from imgaug import augmenters as iaa

   aug_seq = iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-25, 25),
                        )

   exp2 = neptune.create_experiment()
   for run in range(20):
       img_aug= aug_seq.augment_image(img)
       img_pil_aug = Image.fromarray(img_aug)
       exp2.send_image('quokka_version_{}'.format(run), img_pil_aug)

   exp2.close()

.. image:: ../_static/images/how-to/ht-img-channel-2.png
   :target: ../_static/images/how-to/ht-img-channel-2.png
   :alt: image

How to log metrics to Neptune?
-------------------------------
How to track multiple metrics (loss, scores) in the experiment?

Solution
^^^^^^^^
**Step 1: Log**

In order to log metrics to Neptune, you simply need to:

.. code-block::

   import neptune

   neptune.init('shared/onboarding')
   with neptune.create_experiment():
       # 'log_loss' is User defined metric name
       neptune.send_metric('log_loss', 0.753)
       neptune.send_metric('AUC', 0.95)

Another option is to log `key: value` pair like this:

.. code-block::

   neptune.set_property('model_score', '0.871')

.. note:: You can create as many metrics as you wish.

**Step 2: Analyze**

Browse and analyse your metrics on the dashboard (`example <https://app.neptune.ml/neptune-ml/Home-Credit-Default-Risk/experiments>`_) or in the particular experiment (`example experiment <https://app.neptune.ml/neptune-ml/Home-Credit-Default-Risk/e/HC-11860/channels>`_).

How to version datasets?
------------------------
When working on a project, it is not unusual that I change the datasets on which I train my models. How can I keep track of that in Neptune?

Solution
^^^^^^^^
Under many circumstances it is possible to calculate a hash of your dataset. Even if you are working with large image datasets, you have some sort of a smaller metadata file, that points to image paths. If this is the case you should:

**Step 1**

Create hashing function. For example:

.. code-block::

   import hashlib

   def md5(fname):
       hash_md5 = hashlib.md5()
       with open(fname, "rb") as f:
           for chunk in iter(lambda: f.read(4096), b""):
               hash_md5.update(chunk)
       return hash_md5.hexdigest()

**Step 2**

Calculate the hash of your training data and send it to Neptune as text:

.. code-block::

   TRAIN_FILEPATH = 'PATH/TO/TRAIN/DATA'
   train_hash = md5(TRAIN_FILEPATH)

   neptune.send_text('train_data_version', train_hash)
   ...

**Step 3**

Add data version column to your project dashboard:

.. image:: ../_static/images/how-to/ht-data-version-1.png
   :target: ../_static/images/how-to/ht-data-version-1.png
   :alt: image

.. note:: If your dataset is too large for fast hashing you could think about rearranging your data to have a light-weight metadata file.

How to keep my code private?
----------------------------
My code is proprietary, so I do not want to send any sources to Neptune, while training locally. How to do it?

Solution
^^^^^^^^
All you need to do it to pass empty list ``[]`` to the ``upload_source_files`` parameter, like this:

.. code-block::

   import neptune

   # This function assumes that NEPTUNE_API_TOKEN environment variable is defined.
   neptune.init(project_qualified_name='shared/onboarding')

   with neptune.create_experiment(upload_source_files=[]) as exp:
       ...

As a result you will not send sources to Neptune, so they will not be available in the Source Code tab in the Web app.

How to upload notebook checkpoint?
----------------------------------
I want to add Notebook checkpoint to my project. How to do it?

Solution
^^^^^^^^
Go to your Jupyter, where you can see two Neptune buttons:

* **n** button is for configuration changes
* **Upload** button is for making checkpoint in Neptune

.. image:: ../_static/images/notebooks/buttons_02_1.png
   :target: ../_static/images/notebooks/buttons_02_1.png
   :alt: image

Click **Upload**, whenever you want to create new checkpoint in Neptune. You will see tooltip with link as a confirmation.

.. image:: ../_static/images/notebooks/buttons_03_1.png
   :target: ../_static/images/notebooks/buttons_03_1.png
   :alt: image

.. note:: You can use **Upload** as many times as you want.

How to setup Neptune-enabled JupyterLab on AWS?
-----------------------------------------------
I would like to run Neptune and track experiments that I run on AWS cloud.
How do I do that?

Solution
^^^^^^^^
**Register to AWS**

Follow the `registration instructions <https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/>`_ from official webpage to create your AWS account.

**Start EC2 instance**

Start a new EC2 instance. Select `ubuntu` as your instance type and choose a worker type you need.
You can go with `t2.micro` just to test it out.

**ssh to your instance**

Connect to your instance by going to the terminal and running:

.. code-block:: Bash

    ssh -i /path_to_key/my_key.pem ubuntu@public_dns_name

*(make sure that you put correct key and public_dns_name)*

**Install docker**

Create a new file `install_docker.sh`:

.. code-block:: Bash

    nano install_docker.sh

Copy the following commands to it:

.. code-block:: Bash

    sudo apt-get update
    sudo apt-get install \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg-agent \
        software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo apt-key fingerprint 0EBFCD88
    sudo add-apt-repository \
       "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
       $(lsb_release -cs) \
       stable"
    sudo apt-get update
    sudo apt-get install docker-ce docker-ce-cli containerd.io

Run the installation script:

.. code-block:: Bash

    source install_docker.sh

**Define your secrets**

| Go to Neptune web app, get your ``NEPTUNE_API_TOKEN`` and copy it. Then, create a password for your JupyterLab server.
| Set those two secrets to your environment variables ``NEPTUNE_API_TOKEN`` and ``JUPYTERLAB_PASSWORD``, like below:

.. code-block:: Bash

    export NEPTUNE_API_TOKEN='your_api_token=='
    export JUPYTERLAB_PASSWORD='difficult_password'

**Build docker image**

Create a new file `Dockerfile`:

.. code-block:: Bash

    nano Dockerfile

Copy insights of the following `Dockerfile` to your newly created file:

.. code-block:: Docker

    # Use a miniconda3 as base image
    FROM continuumio/miniconda3

    # Installation of jupyterlab and extensions
    RUN pip install jupyterlab==0.35.6  && \
        pip install jupyterlab-server==0.2.0  && \
        conda install -c conda-forge nodejs

    # Installation of Neptune and enabling neptune extension
    RUN pip install neptune-client  && \
        pip install neptune-notebooks  && \
        jupyter labextension install neptune-notebooks

    # Setting up Neptune API token as env variable
    ARG NEPTUNE_API_TOKEN
    ENV NEPTUNE_API_TOKEN=$NEPTUNE_API_TOKEN

    # Adding current directory to container
    ADD . /mnt/workdir
    WORKDIR /mnt/workdir

| *(If you want to run on GPU make sure to change your `Dockerfile` to start from nvidia docker images)*.

Run following command to build your docker image:

.. code-block:: Bash

    sudo docker build -t jupyterlab --build-arg NEPTUNE_API_TOKEN=$NEPTUNE_API_TOKEN .

**Start JupyterLab server**

Spin up JupyterLab server with docker:

.. code-block:: Bash

    sudo docker run --rm -v `pwd`:/work/output -p 8888:8888 jupyterlab:latest \
    /opt/conda/bin/jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token=$JUPYTERLAB_PASSWORD

**Forward ports via ssh tunnel**

Open a new terminal on your local machine and run:

.. code-block:: Bash

    ssh -L 8888:localhost:8888 ubuntu@public_dns_name &

*(make sure that you put correct public_dns_name)*

**Open JupyterLab server in your browser**

Go to `localhost:8888` and enjoy your JupyterLab server with Neptune!

**Final result**

Neptune extensions are enabled and ``NEPTUNE_API_TOKEN`` is already in the environment variable so you can work with Notebooks and run experiments with no problems.

How to track Google Colab experiments with Neptune?
-----------------------------------------------
I would like to run my experiments on google colab and track them with Neptune.
How do I do that?

Solution
^^^^^^^^
**Install Neptune client**

Go to your first cell and install `neptune-client`:

.. code-block:: Bash

    ! pip install neptune-client

**Set Neptune API token**

Go to Neptune app and get your API token.
Set it to the environment variable `NEPTUNE_API_TOKEN`:

.. code-block:: Bash

    ! export NEPTUNE_API_TOKEN='your_private_neptune_api_token=='

Delete this cell.

.. warning::

    It is very important that you delete this cell not to share your private token with anyone.

**Run your training script with Neptune**

.. code-block:: Bash

    import neptune
    neptune.init('USER_NAME/'PROJECT_NAME')

    with neptune.create_experiment():
        neptune.send_metric('auc', 0.92)
