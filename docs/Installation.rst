.. _installation:

Installation
************

Below are the full dependency installation steps so as to ensure
reproducibility across platforms, utilizing container technologies Docker
& Apptainer (a.k.a. Singularity).

While setting up a Python library through Docker might seem a bit involved,
especially in comparison to standard practices, we believe it to be best for
simplicity & reproducibility purposes.

Do note that we also provide pre-built Docker & Apptainer images, meaning
that you do not need to go through the upcoming setup steps presented in the next
section if you do not intend to run Gran on your own machine.

.. include:: Installation_On_your_own_machine.rst
.. include:: Installation_On_a_SLURM_cluster.rst
.. include:: Installation_On_a_machine_without_sudo_privileges.rst
