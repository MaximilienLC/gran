python3 -m gran.rands -m hydra/launcher=submitit_slurm hydra/launcher=submitit_slurm task=acrobot pop_size=40 num_gens

python3 -m gran.rand.main task=acrobot pop_size=40 num_gens