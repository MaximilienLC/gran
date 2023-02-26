.. _installation_on_a_machine_without_sudo_privileges:

On a machine without sudo privileges
====================================

Installing on a machine without sudo privileges requires that you either have
Docker privileges **or** Apptainer installed on that machine. 

As mentioned in the beginning of the :ref:`installation_on_your_own_machine`
installation section, Docker/Apptainer-free installation is possible by
refering to the contents of the `Dockerfile
<https://github.com/MaximilienLC/gran/blob/main/docker/Dockerfile>`_
for general directives. 

1. Install the repository
-------------------------

Set the ``GRAN_PATH`` variable.

.. code-block:: console

   # Examples
   $ GRAN_PATH=/home/stav/gran
   $ GRAN_PATH=/home/mleclei/Dropbox/gran

Clone the repository.

.. code-block:: console

   $ git clone git@github.com:MaximilienLC/gran.git ${GRAN_PATH}

2. Option A : Download the Docker image
---------------------------------------

.. code-block:: console

    $ cd ${GRAN_PATH}/docker/
    $ wget https://nextcloud.computecanada.ca/index.php/s/2ZJHsXjoNr7QatG/download -O image.tar
    $ docker load -i ${GRAN_PATH}/docker/image.tar

2. Option B : Download the Apptainer image
------------------------------------------

.. code-block:: console

    $ cd ${GRAN_PATH}/docker/
    $ wget https://nextcloud.computecanada.ca/index.php/s/DCx46ZYsc22xYd2/download -O image.sif
    $ docker load -i ${GRAN_PATH}/docker/image.tar
