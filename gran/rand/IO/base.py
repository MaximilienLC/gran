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

import os
import pickle

from gran.util.misc import cfg


class BaseIO:
    """
    Base IO class.
    Concrete subclasses need to be named *IO*.
    A default subclass is defined below.
    """

    def __init__(self):
        self.setup_save_points()

    def setup_save_points(self) -> None:
        """
        Method called upon object initialization. Sets up points at which to
        save the experiment's current state.
        """
        assert cfg.save_frequency in range(0, cfg.num_gens)

        self.save_points = [cfg.prev_num_gens + cfg.num_gens]

        if cfg.save_frequency == 0:
            return

        for i in range(cfg.num_gens // cfg.save_frequency):
            self.save_points.append(
                cfg.prev_num_gens + cfg.save_frequency * (i + 1)
            )

    def load_state(self) -> list:
        """
        Load a previous experiment's state.

        Returns:
            list - state (fitnesses, bots, ...) of experiment.
        """
        state_path = cfg.path + str(self.args.prev_num_gens) + "/state.pkl"

        if not os.path.isfile(state_path):
            raise Exception("No saved state found at " + state_path + ".")

        with open(state_path, "rb") as f:
            state = pickle.load(f)

        return state

    def save_state(self, state: list, curr_gen: int) -> None:
        """
        Save the current experiment's state.

        Args:
            state - state (fitnesses, bots, ...) of experiment.
            curr_gen - Current generation.
        """
        state_dir_path = os.getcwd() + "/" + str(curr_gen) + "/"

        if not os.path.exists(state_dir_path):
            os.makedirs(state_dir_path, exist_ok=True)

        with open(state_dir_path + "state.pkl", "wb") as f:
            pickle.dump(state, f)


class IO(BaseIO):
    pass
