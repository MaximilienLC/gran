.. _installation_setting_up:

Setting up
==========

.. _installation_setting_up_github_repository_on_your_local_machine:

1. Setting up the GitHub repository on your local machine
---------------------------------------------------------

Set the ``LOCAL_GRAN_PATH`` variable.
.. code-block:: console
   # Examples
   $ LOCAL_GRAN_PATH=/home/stav/gran
   $ LOCAL_GRAN_PATH=/home/max/Dropbox/gran

Clone the repository
.. code-block:: console
   $ git clone git@github.com:MaximilienLC/gran.git ${LOCAL_GRAN_PATH}

2. Setting up Docker / Apptainer on your local machine
------------------------------------------------------

**a) Install and setup Docker**
Install the deb package
test1  
test2    
test3

.. note::
    Any existing yet recent version of Docker already installed on your machine should be fine.
    e.g. installing[[link](https://docs.docker.com/engine/install/)]).

.. code-block:: console
    $ sudo apt-get install -y docker.io

#### ii. 
   ```
   sudo apt-get install -y docker.io
   ```
   Give yourself docker permissions to not require sudo privileges for docker later on.
   ```
   sudo groupadd docker
   sudo usermod -aG docker ${USER}
   ```
   Finally, log out and log back to apply changes.

