.. _installation_on_your_own_machine:

On your own machine
===================

.. note::

    Docker/Apptainer free installation is possible but not considered in this
    documentation. If you still wish to proceed without container technologies,
    please refer to the contents of the
    `Dockerfile
    <https://github.com/MaximilienLC/gran/blob/main/docker/Dockerfile>`_
    for approximate directives.

1. Install the repository
-------------------------

Set the ``GRAN_PATH`` variable.

.. code-block:: console

   $ # Examples
   $ GRAN_PATH=/home/stav/gran
   $ GRAN_PATH=/home/max/Dropbox/gran

Clone the repository.

.. code-block:: console

   $ git clone git@github.com:MaximilienLC/gran.git ${GRAN_PATH}

2. Install Docker
-----------------

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

3. Install the NVIDIA Driver
----------------------------

.. note::

    If you are not running Ubuntu Desktop, please refer to
    `the official installation guide
    <https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html>`_.

GUI Instructions:

    1. Press the Super key.
    2. Type "Software & Updates".
    3. Select the "Additional Drivers" tab.
    4. Select: "Using NVIDIA driver metapackage from nvidia-driver-XXX (proprietery, tested)".

4. Install the NVIDIA Container Toolkit
---------------------------------------

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

5. Get the Docker image
-----------------------

Option A: Build it.

.. code-block:: console

    $ cd ${GRAN_PATH}/
    $ docker build -f docker/Dockerfile -t gran:latest .

Option B : Download it.

.. code-block:: console

    $ cd ${GRAN_PATH}/docker/
    $ wget https://nextcloud.computecanada.ca/index.php/s/2ZJHsXjoNr7QatG/download \
          -O image.tar
    $ docker load -i ${GRAN_PATH}/docker/image.tar

6. Install Apptainer
--------------------

.. note::

    You can skip this section if you only want to use pre-built Apptainer
    images.
    
.. note::

    If you are not running Ubuntu, please refer to
    `the official installation guide 
    <https://apptainer.org/docs/admin/main/installation.html>`_.

Add the repository and install the deb package.

.. code-block:: console

    $ sudo add-apt-repository -y ppa:apptainer/ppa
    $ sudo apt-get update
    $ sudo apt-get install -y apptainer

7. Get the Apptainer image
--------------------------

.. note::

    Skip if you did not install Apptainer.

Option A: Build it.

.. code-block:: console

    $ cd ${LOCAL_GRAN_PATH}/docker/
    $ docker save gran:latest -o image.tar
    $ apptainer build image.sif docker-archive://image.tar

Option B : Download it.

.. code-block:: console

    $ cd ${GRAN_PATH}/docker/
    $ wget https://nextcloud.computecanada.ca/index.php/s/DCx46ZYsc22xYd2/download -O image.sif
    $ docker load -i ${GRAN_PATH}/docker/image.tar
