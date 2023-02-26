.. _execution_on_your_own_machine:

On your own machine
===================

1. Verify the path
------------------

Make sure the ``GRAN_PATH`` variable is still set.

.. code-block:: console

   $ # Examples
   $ GRAN_PATH=/home/stav/gran
   $ GRAN_PATH=/home/max/Dropbox/gran

2. a. Execute the sample code
-----------------------------

.. code-block:: console

    $ docker run --rm --privileged --gpus=all \
           -e PYTHONPATH=${PYTHONPATH}:${GRAN_PATH} \
           -v ${GRAN_PATH}:${GRAN_PATH} -w ${GRAN_PATH} \
           -v /dev/shm/:/dev/shm/ gran:latest python3 gran/rand/main.py

2. b. Run jupyter-lab
---------------------

.. code-block:: console

    $ docker run --rm --privileged --gpus=all -p 8888:8888 \
          -e PYTHONPATH=${PYTHONPATH}:${GRAN_PATH} \
          -v ${GRAN_PATH}:${GRAN_PATH} -w ${GRAN_PATH} \
          -v /dev/shm/:/dev/shm/ gran:latest jupyter-lab \
          --allow-root --ip 0.0.0.0
