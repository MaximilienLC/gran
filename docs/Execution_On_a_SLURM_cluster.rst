.. _execution_on_a_slurm_cluster:

On a SLURM cluster
==================

1. Verify the path
------------------

Make sure the ``GRAN_PATH`` variable is still set.

.. code-block:: console

   $ # Examples
   $ GRAN_PATH=/scratch/stavb/gran
   $ GRAN_PATH=/scratch/mleclei/Dropbox/gran

2. Activate the Submitit venv in tmux
-------------------------------------

.. code-block:: console

    $ tmux
    $ cd ${GRAN_PATH}
    $ . ${GRAN_PATH}/venv/bin/activate

3. a. Execute the sample code
-----------------------------

.. code-block:: console

    $ python3 -m gran -m hydra/launcher=submitit_slurm +launcher=slurm

3. b. Run jupyter-lab
---------------------

From your own machine create a SSH tunnel.

.. code-block:: console

    $ # Example
    $ sshuttle --dns -Nr mleclei@beluga.computecanada.ca

Run the lab.

.. code-block:: console

    $ salloc --account=rrg-pbellec --gres=gpu:1 apptainer exec --nv \
          --bind ${GRAN_PATH}:${GRAN_PATH} --pwd ${GRAN_PATH} \
          --env PYTHONPATH=${PYTHONPATH}:${GRAN_PATH} \
          ${GRAN_PATH}/docker/image.sif jupyter-lab --allow-root \
          --ip $(hostname -f) --no-browser