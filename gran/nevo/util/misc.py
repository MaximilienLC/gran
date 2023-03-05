import numpy as np

from gran.util.misc import config


def generate_new_seeds(curr_gen: int, num_pops: int) -> np.ndarray:
    """
    Method that produces new seeds meant to mutate the agents.

    Args:
        curr_gen - Current generation.
        num_pops - Number of populations.
    Returns:
        np.ndarray - Array of seeds.
    """
    new_seeds = np.random.randint(
        0,
        2**32,
        (
            config.agent.pop_size,
            1 if config.agent.pop_merge else num_pops,
        ),
        dtype=np.uint32,
    )

    if config.agent.pop_merge:

        new_seeds = np.repeat(new_seeds, 2, axis=1)

        if curr_gen == 0:
            new_seeds[:, 1] = new_seeds[:, 1][::-1]

    return new_seeds


def verify_assertions():

    assert config.agent.pop_size % config.num_cpus == 0

    if config.agent.algorithm == "es":
        assert config.agent.gen_transfer == "none"
        assert config.agent.pop_merge == False

    assert config.agent.run_num_steps >= 0

    assert not (
        config.agent.run_num_steps == 0 and "env" in config.agent.gen_transfer
    )

    if config.agent.algorithm == "ga":

        assert config.agent.gen_transfer in [
            "none",
            "env",
            "fit",
            "mem",
            "env+fit",
            "env+mem",
            "fit+mem",
            "env+fit+mem",
        ]

    assert config.agent.save_interval in range(
        1, config.agent.num_gens_per_iter
    )

    assert config.agent.save_interval % config.agent.num_gens_per_iter
