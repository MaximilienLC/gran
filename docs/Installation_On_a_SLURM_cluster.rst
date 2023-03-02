.. _installation_on_a_slurm_cluster:

On a SLURM cluster
==================

.. note::

    Installing on a SLURM cluster machine requires Apptainer and Python to
    be loadable as modules.

1. Install the repository
-------------------------

Set the ``GRAN_PATH`` variable.

.. code-block:: console

   $ # Examples
   $ GRAN_PATH=/scratch/stavb/gran
   $ GRAN_PATH=/scratch/mleclei/Dropbox/gran

Clone the repository.

.. code-block:: console

   $ git clone git@github.com:MaximilienLC/gran.git ${GRAN_PATH}

2. Install the Submitit related packages
----------------------------------------

.. code-block:: console

    $ module load python/3.10
    $ python3 -m venv ${GRAN_PATH}/venv
    $ . ${GRAN_PATH}/venv/bin/activate
    $ pip install -r ${GRAN_PATH}/reqs/sweep_reqs.txt

3. Hard-code the Submitit + Apptainer integration
-------------------------------------------------

.. note::

    This command is of course less than ideal but the easiest and quickest fix.

.. code-block:: console

    $ sed -i "s|shlex.quote(sys.executable)|\"apptainer exec --nv --bind ${GRAN_PATH}:${GRAN_PATH} --pwd ${GRAN_PATH} \
          --env PYTHONPATH=\${PYTHONPATH}:${GRAN_PATH} ${GRAN_PATH}\/docker\/image.sif python3\"|" ${GRAN_PATH}/venv/lib/python3.10/site-packages/submitit/slurm/slurm.py

4. Download the Apptainer image
-------------------------------

.. code-block:: console

    $ cd ${GRAN_PATH}/docker/
    $ wget https://nextcloud.computecanada.ca/index.php/s/DCx46ZYsc22xYd2/download \
          -O image.sif
