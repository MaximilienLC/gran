"""
Training agents in environments.
"""

import os
import sys
import subprocess

import hydra
from hydra import instantiate


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main():
    for iter in range(cfg.num_iters):
        # Train the encoder if decided
        if cfg.encoder.in_use:
            instantiate(cfg.encoder)()

        # Train the predictor if decided
        if cfg.predictor.in_use:
            instantiate(cfg.predictor)()

        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", INSIDE_MPI_FORK="1"
        )

        subprocess.check_call(
            [
                "mpiexec",
                "--allow-run-as-root",  # Docker
                "--oversubscribe",  # Submitit
                "-n",
                str(cfg.num_mpi_procs),
                sys.executable,
                "gran/main.py",
                str(HydraConfig.get().runtime.output_dir),
            ],
            env=env,
        )


if __name__ == "__main__":
    if os.getenv("INSIDE_MPI_FORK"):
        cfg = OmegaConf.create(sys.argv[1])

        assert cfg.stage in ["train", "test"]

        if cfg.stage == "train":
            from gran.nevo.evolve import evolve

            evolve()

        else:  # cfg.stage == "test":
            from gran.nevo.evaluate import evaluate

            evaluate()

    else:
        main()
