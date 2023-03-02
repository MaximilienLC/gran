# Copyright 2023 The Gran Authors.
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

import glob
import os
import pickle

import numpy as np

from gran.util.misc import config


class BaseIO:
    """
    Base IO class.
    Concrete subclasses need to be named *IO*.
    A default subclass is defined below.
    """

    def __init__(self):

        assert config.ecosystem.save_interval in range(
            1, config.ecosystem.num_gens_per_iter
        )

        assert (
            config.ecosystem.save_interval % config.ecosystem.num_gens_per_iter
        )

        self.compute_prev_new_num_gens()
        self.compute_save_timesteps()

    def compute_prev_new_num_gens(self):

        completed_generations = []

        for folder in glob.glob("*"):
            if folder.isdigit() and os.path.isdir(folder):
                if os.path.isfile(folder + "state.pkl"):
                    completed_generations += int(folder)

        self.prev_num_gens = (
            0
            if completed_generations == []
            else np.amax(completed_generations)
        )

        curr_iter_elapsed_num_gens = (
            self.prev_num_gens - config.ecosystem.prev_num_gens
        )

        self.new_num_gens = (
            config.ecosystem.num_gens_per_iter - curr_iter_elapsed_num_gens
        )

    def compute_save_timesteps(self):

        self.save_points = []

        if config.ecosystem.save_interval != 1:
            self.save_points += [config.ecosystem.prev_num_gens + 1]

        for i in range(
            config.ecosystem.num_gens_per_iter
            // config.ecosystem.save_interval
        ):
            self.save_points.append(
                config.ecosystem.prev_num_gens
                + config.ecosystem.save_interval * (i + 1)
            )

    def load_state(self) -> list:
        """
        Load a previous experiment's state.

        Returns:
            list - state (fitnesses, agents, ...) of experiment.
        """
        state_path = str(self.prev_num_gens) + "/state.pkl"

        if not os.path.isfile(state_path):
            raise Exception("No saved state found at " + state_path + ".")

        with open(state_path, "rb") as f:
            state = pickle.load(f)

        return state

    def save_state(self, state: list, curr_gen: int) -> None:
        """
        Save the current experiment's state.

        Args:
            state - state (fitnesses, agents, ...) of experiment.
            curr_gen - Current generation.
        """
        state_dir_path = os.getcwd() + "/" + str(curr_gen) + "/"

        if not os.path.exists(state_dir_path):
            os.makedirs(state_dir_path, exist_ok=True)

        with open(state_dir_path + "state.pkl", "wb") as f:
            pickle.dump(state, f)


class IO(BaseIO):
    pass
