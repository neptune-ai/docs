Run on cloud
================

How to setup Neptune-enabled jupyterlab on AWS?
--------------------------------------------------
I would like to run Neptune and track experiments that I run on AWS cloud.
How do I do that?

Solution
^^^^^^^^
**Register to AWS**

Follow the instructions from `here <https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/>`_ to create your AWS account.

**Start EC2 instance**

Start a new EC2 instance.
Select `ubuntu` as your instance type and choose a worker type you need.
You can go with `t2.micro` just to test it out.

**ssh to your instance**

Connect to your instance by going to the terminal and running:

.. code-block::

    ssh -i /path_to_key/my_key.pem ubuntu@public_dns_name

**Install docker**

Create a new file `install_docker.sh`

.. code-block::

    nano install_docker.sh

Copy the following commands to it:

.. code-block::

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

.. code-block::

    source install_docker.sh

**Define your secrets**

Go to Neptune app, get you Neptune API token and copy it.
Create a password for your jupyterlab server.
Set those two secrets to your environment variables `NEPTUNE_API_TOKEN`  and `JUPYTERLAB_PASSWORD`

.. code-block::

    export NEPTUNE_API_TOKEN='your_api_token=='
    export JUPYTERLAB_PASSWORD='difficult_password'

**Build docker image**

Create a new file `Dockerfile`.

.. code-block::

    nano Dockerfile

Copy insights of the following `Dockerfile` to your newly created file:

.. code-block::

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

If you want to run on GPU make sure to change your `Dockerfile` to start from nvidia docker images.
Build your docker image:

.. code-block::

    sudo docker build -t jupyterlab --build-arg NEPTUNE_API_TOKEN=$NEPTUNE_API_TOKEN .

**Start jupyterlab server**

Spin up jupyterlab server with docker:

.. code-block::

    sudo docker run --rm -v `pwd`:/work/output -p 8888:8888 jupyterlab:latest \
    /opt/conda/bin/jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token=$JUPYTERLAB_PASSWORD

**Forward ports via ssh tunnel**

Open a new terminal on your machine and run:

.. code-block::

    ssh -L 8888:localhost:8888 ubuntu@public_dns_name &

.. image:: ../_static/images/how-to/ht-output-download-1.png
   :target: ../_static/images/how-to/ht-output-download-1.png
   :alt: image

**Open jupyterlab server in your browser**

Go to `localhost:8888` and enjoy your jupyterlab server with Neptune!
Neptune extensions are enabled and NEPTUNE_API_TOKEN is already in the environment variable so you
can run you experiments with no problems.
