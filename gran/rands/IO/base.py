# Copyright 2022 Maximilien Le Clei.
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

import argparse
import os
import pickle


class IOBase:
    """
    IO Base class. IO objects are used to handle input/output files. Concrete
    subclasses need to be named *IO* (a default subclass is defined below).
    """

    def __init__(self, args: argparse.Namespace):
        """
        Constructor.

        Args:
            args - Experiment specific arguments (obtained through argparse).
        """
        self.args = args

        self.setup_save_points()
        self.setup_state_path()

    def setup_save_points(self) -> None:
        """
        Method called upon object initialization. Sets up points at which to
        save the experiment's current state.
        """
        if (
            self.args.save_frequency > self.args.nb_generations
            or self.args.save_frequency < 0
        ):
            raise Exception(
                "'save_frequency' needs to be in the range [0, nb_generations]."
            )

        self.save_points = [
            self.args.nb_elapsed_generations + self.args.nb_generations
        ]

        if self.args.save_frequency == 0:
            return

        for i in range(self.args.nb_generations // self.args.save_frequency):
            self.save_points.append(
                self.args.nb_elapsed_generations
                + self.args.save_frequency * (i + 1)
            )

    def setup_state_path(self) -> None:
        """
        Method called upon object initialization. Sets up the path which
        states will be loaded from and saved into.
        """
        self.path = (
            self.args.data_path
            + "states/"
            + self.args.env_path.replace("/", ".").replace(".py", "")
            + "/"
        )

        if self.args.extra_arguments == {}:

            self.path += "~"

        else:

            for key in sorted(self.args.extra_arguments):
                self.path += (
                    str(key) + "." + str(self.args.extra_arguments[key]) + "~"
                )

            self.path = self.path[:-1] + "/"

        self.path += (
            self.args.bots_path.replace("/", ".").replace(".py", "") + "/"
        )

    def load_state(self) -> list:
        """
        Load a previous experiment's state.

        Returns:
            list - state (fitnesses, bots, ...) of experiment.
        """
        state_path = (
            self.path
            + str(self.args.population_size)
            + "/"
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
        state_dir_path = (
            self.path
            + str(self.args.population_size)
            + "/"
            + str(gen_nb)
            + "/"
        )

        if not os.path.exists(state_dir_path):
            os.makedirs(state_dir_path, exist_ok=True)

        with open(state_dir_path + "state.pkl", "wb") as f:
            pickle.dump(state, f)


class IO(IOBase):
    pass
