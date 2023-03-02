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
            config.ecosystem.pop_size,
            1 if config.ecosystem.pop_merge else num_pops,
        ),
        dtype=np.uint32,
    )

    if config.ecosystem.pop_merge:

        new_seeds = np.repeat(new_seeds, 2, axis=1)

        if curr_gen == 0:
            new_seeds[:, 1] = new_seeds[:, 1][::-1]

    return new_seeds
