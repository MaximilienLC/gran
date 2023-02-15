import numpy as np

from gran.util.misc import standardize


class BaseCollector:
    """
    Collector base class.
    Collector objects collect observations and rewards from an emulator and
    save it to disk.
    """

    def __init__(self, rank: int = -1) -> None:
        self.iter = 0
        self.rank = rank

    def __call__(self) -> None:
        self.collect()
        self.dump()

    def collect(self) -> None:
        """
        Collect data from emulator.
        """
        assert hasattr(self, "agent"), "Collector objects require the "
        "attribute `agent`: the agent to run in the emulator."

        assert hasattr(self, "emulator"), "Collector objects require the "
        "attribute `emulator`: the emulator to collect data from."

        assert hasattr(self, "num_steps"), "Collector objects require the "
        "attribute `num_steps`: the number of steps to run in the emulator."

        assert hasattr(self, "reset_emulator_state"), "Collector objects "
        "require the attribute `reset_emulator_state`: a function that "
        "resets the emulator."

        assert hasattr(self, "run_emulator_step"), "Collector objects require "
        "the attribute `run_emulator_step`: a function that runs a single "
        "step in the emulator."

        obs_list, rew_list = [], []

        for i in range(self.num_steps):
            if i == 0 or done:
                obs, rew, done = (
                    self.reset_emulator_state(self.emulator, seed),
                    np.nan,
                    False,
                )
            else:
                obs, rew, done = self.run_emulator_step(
                    self.emulator, self.agent(obs)
                )

            obs_list.append(obs)
            rew_list.append(rew)

    def dump(self):
        """
        Dump collected data to disk.
        """
        np.save(f"obs_{self.rank}_{self.iter}.npy", np.array(obs_list))
        np.save(f"rew_{self.rank}_{self.iter}.npy", np.array(rew_list))
