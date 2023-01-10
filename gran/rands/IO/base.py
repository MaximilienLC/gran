# Copyright 2022 The Gran Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle

from omegaconf import DictConfig


class IOBase:
    """
    IO Base class. IO objects are used to handle input/output files. Concrete
    subclasses need to be named *IO* (a default subclass is defined below).
    """

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg - Experiment specific configuration.
        """
        self.cfg = cfg

        self.setup_save_points()

    def setup_save_points(self) -> None:
        """
        Method called upon object initialization. Sets up points at which to
        save the experiment's current state.
        """
        assert self.cfg.rands.save_frequency in range(
            0, self.cfg.rands.nb_generations
        ), "'cfg.rands.save_frequency' needs to be in range [0, nb_gen]."

        self.save_points = [
            self.cfg.rands.nb_elapsed_generations
            + self.cfg.rands.nb_generations
        ]

        if self.cfg.rands.save_frequency == 0:
            return

        for i in range(
            self.cfg.rands.nb_generations // self.cfg.rands.save_frequency
        ):
            self.save_points.append(
                self.cfg.rands.nb_elapsed_generations
                + self.cfg.rands.save_frequency * (i + 1)
            )

    def load_state(self) -> list:
        """
        Load a previous experiment's state.

        Returns:
            list - state (fitnesses, bots, ...) of experiment.
        """
        state_path = (
            self.cfg.path
            + str(self.args.nb_elapsed_generations)
            + "/state.pkl"
        )

        if not os.path.isfile(state_path):
            raise Exception("No saved state found at " + state_path + ".")

        with open(state_path, "rb") as f:
            state = pickle.load(f)

        return state

    def save_state(self, state: list, gen_nb: int) -> None:
        """
        Save the current experiment's state.

        Args:
            state - state (fitnesses, bots, ...) of experiment.
            gen_nb - Generation number.
        """
        state_dir_path = os.getcwd() + "/" + str(gen_nb) + "/"

        if not os.path.exists(state_dir_path):
            os.makedirs(state_dir_path, exist_ok=True)

        with open(state_dir_path + "state.pkl", "wb") as f:
            pickle.dump(state, f)


class IO(IOBase):
    pass
