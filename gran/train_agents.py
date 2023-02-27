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
    for iter in range(config.num_iters):
        # Train the encoder if decided
        if config.encoder.in_use:
            instantiate(config.encoder)()

        # Train the predictor if decided
        if config.predictor.in_use:
            instantiate(config.predictor)()

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
                str(config.num_cpus),
                sys.executable,
                "gran/main.py",
                str(HydraConfig.get().runtime.output_dir),
            ],
            env=env,
        )


if __name__ == "__main__":
    if os.getenv("INSIDE_MPI_FORK"):
        config = OmegaConf.create(sys.argv[1])

        assert config.stage in ["train", "test"]

        if config.stage == "train":
            from gran.ne.evolve import evolve

            evolve()

        else:  # config.stage == "test":
            from gran.ne.evaluate import evaluate

            evaluate()

    else:
        main()
