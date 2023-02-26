.. _installation_setting_up_on_your_personal_machine:

Setting up on your personal machine
===================================

Below are the full dependency installation steps so as to ensure
reproducibility across platforms, utilizing container technologies
Docker & Apptainer (a.k.a. Singularity).

While setting up a Python library through a container might seem a
little bit too involved, especially when compared to standard practices, we
believe it to be best in terms of simplicity & reproducibility.

Do note that we also provide pre-built Docker & Apptainer images, meaning
that you do not need to go through the following setting up steps if you do
not intend to run Gran on your personal machine.

.. note::

    Docker/Apptainer free installation is of course possible but is not
    considered in this documentation. If you still wish to proceed without
    container technologies, please refer to the contents of our
    `Dockerfile
    <https://github.com/MaximilienLC/gran/blob/main/docker/Dockerfile>`_
    for general directives.

.. _installation_setting_up_on_your_personal_machine_setting_up_the_github_repository:

1. Setting up the GitHub repository
-----------------------------------

Set the ``GRAN_PATH`` variable.

.. code-block:: console

   # Examples
   $ GRAN_PATH=/home/stav/gran
   $ GRAN_PATH=/home/max/Dropbox/gran

Clone the repository.

.. code-block:: console

   $ git clone git@github.com:MaximilienLC/gran.git ${GRAN_PATH}

.. _installation_setting_up_docker_apptainer_on_your_personal_machine:

2. Setting up Docker / Apptainer on your personal machine
---------------------------------------------------------

**a) Install and setup Docker**

.. note::

    If you are not running Ubuntu, please refer to
    `the official installation guide
    <https://docs.docker.com/engine/install/>`_.

Install the deb package.

.. code-block:: console

    $ sudo apt-get install -y docker.io

Give yourself permissions to not require sudo for later docker commands.

.. code-block:: console

    $ sudo groupadd docker
    $ sudo usermod -aG docker ${USER}

Finally, log out and log back to apply changes.

**b) Install Apptainer**

.. note::

    You can skip this section if you have Docker privileges across all your
    machines.
    
    If you do not have Docker privileges and are not running Ubuntu, please
    refer to `the official installation guide 
    <https://apptainer.org/docs/admin/main/installation.html>`_.

Add the repository and install the deb package.

.. code-block:: console

    $ sudo add-apt-repository -y ppa:apptainer/ppa
    $ sudo apt-get update
    $ sudo apt-get install -y apptainer

**c) Install the NVIDIA Driver**

.. note::

    If you are not running Ubuntu Desktop, please refer to
    `the official installation guide
    <https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html>`_.

GUI Instructions:

1. Press the Super key.
2. Type "Software & Updates".
3. Select the "Additional Drivers" tab.
4. Under the NVIDIA Corporation section, select: "Using NVIDIA driver
metapackage from nvidia-driver-XXX (proprietery, tested)".

**d) Install the NVIDIA Container Toolkit**

.. note::

    If you are not running Ubuntu, please refer to
    `the official installation guide
    <https://docs.docker.com/engine/install/>`_.

Setup the package repository and the GPG key.

.. code-block:: console

    $ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
        && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

Install the deb package and restart the docker engine.

.. code-block:: console

    $ sudo apt-get update
    $ sudo apt-get install -y nvidia-docker2
    $ sudo systemctl restart docker

**e) Build the Docker image**

.. code-block:: console

    $ cd ${GRAN_PATH}/
    $ docker build -f docker/Dockerfile -t gran:latest .

**f) Build the Apptainer image**

.. note::

    Skip if you did not install Apptainer.

.. code-block:: console

    $ cd ${LOCAL_GRAN_PATH}/docker/
    $ docker save gran:latest -o image.tar
    $ apptainer build image.sif docker-archive://image.tar
