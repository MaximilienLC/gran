.. _execution_on_a_machine_without_sudo_privileges:

On a machine without sudo privileges
====================================

1. Verify the path
------------------

Make sure the ``GRAN_PATH`` variable is still set.

.. code-block:: console

   # Examples
   $ GRAN_PATH=/home/stav/gran
   $ GRAN_PATH=/home/mleclei/Dropbox/gran

2. Option A. Execute the sample code
------------------------------------

.. code-block:: console

   $ apptainer exec --nv --bind ${GRAN_PATH}:${GRAN_PATH} \
         --pwd ${GRAN_PATH} --env PYTHONPATH=${PYTHONPATH}:${GRAN_PATH} \
         ${GRAN_PATH}/docker/image.sif python3 gran

2. Option B. Run Jupyter-lab
----------------------------

From your own machine create a SSH tunnel.

.. code-block:: console

   # Example
   $ ssh ginkgo -NL 1234:localhost:1234

Start the notebook on the running machine.

.. code-block:: console

   $ apptainer exec --nv --bind ${GRAN_PATH}:${GRAN_PATH} \
         --pwd ${GRAN_PATH} --env PYTHONPATH=${GRAN_PATH}:${GRAN_PATH} \
         ${GRAN_PATH}/docker/image.sif jupyter-lab \
         --allow-root --no-browser --port 1234