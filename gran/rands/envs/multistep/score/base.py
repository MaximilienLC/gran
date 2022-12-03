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

from typing import Any, Tuple

import numpy as np

from gran.rands.envs.multistep.base import MultistepEnvBase


class ScoreMultistepEnvBase(MultistepEnvBase):

    """
    Score Multistep Env Base class.
    Concrete subclasses need to be named *Env*.
    """

    def __init__(self, args):

        if not hasattr(self, "get_emulator_state"):
            raise NotImplementedError(
                "Score Multistep Environments require the attribute "
                "'get_emulator_state': a function that returns the "
                "emulator's state."
            )
        if not hasattr(self, "set_emulator_state"):
            raise NotImplementedError(
                "Score Multistep Environments require the attribute "
                "'set_emulator_state': a function that sets the emulator's "
                "state."
            )

        if "trials" not in args.extra_arguments:
            args.extra_arguments["trials"] = 1
        elif not isinstance(args.extra_arguments["trials"], int):
            raise Exception(
                "Extra argument 'trials' is of the wrong type. "
                "It needs to be an integer >= 1."
            )
        elif args.extra_arguments["trials"] < 1:
            raise Exception("Extra argument 'trials' needs to be >= 1.")

        super().__init__(args, io_path="IO.base", nb_pops=1)

        if (
            args.extra_arguments["trials"] > 1
            and "fit" in args.extra_arguments["transfer"]
        ):
            raise Exception(
                "Extra argument 'trials' > 1 requires "
                "extra argument 'transfer' = 'no'."
            )

    def reset(self, gen_nb: int, trial_nb: int) -> np.ndarray:
        """
        First reset function called during the run.
        Used to either set the emulator seed or resume from a previous state.

        Args:
            gen_nb - Generation number.
            trial_nb - Trial number.
        Returns:
            np.ndarray - The initial environment observation.
        """
        if "env" in self.args.extra_arguments["transfer"]:

            # Unstable seeding behaviour for Atari
            if gen_nb == 0 and "atari" not in self.args.env_path:
                self.bot.seed = 0

            obs, _ = self.emulator.reset(seed=self.bot.seed)

            if gen_nb > 0:
                self.set_emulator_state(self.emulator, self.bot.emulator_state)
                obs = self.bot.obs.copy()

        else:

            if self.args.extra_arguments["seeding"] != "reg":
                seed = self.args.extra_arguments["seeding"]
            else:
                seed = gen_nb * self.args.extra_arguments["trials"] + trial_nb

            obs, _ = self.emulator.reset(seed=seed)

        return obs

    def done_reset(self, gen_nb: int) -> Tuple[Any, bool]:
        """
        Reset function called whenever the emulator returns done.

        Args:
            gen_nb - Generation number.
        Returns:
            Any - A new environment observation (np.ndarray) or nothing.
            bool - Whether the episode should terminate.
        """
        self.bot.reset()

        if "env" in self.args.extra_arguments["transfer"]:

            if self.log:
                print(self.bot.episode_score)
                self.bot.episode_score = 0

            if "atari" not in self.args.env_path:
                if "seeding" in self.args.extra_arguments:
                    self.bot.seed = self.args.extra_arguments["seeding"]
                else:
                    self.bot.seed = gen_nb

            obs, _ = self.emulator.reset(seed=self.bot.seed)

            return obs, False

        else:

            return None, True

    def final_reset(self, obs: np.ndarray) -> None:
        """
        Reset function called at the end of every run.

        Args:
            obs - The final environment observation.
        """
        if "mem" not in self.args.extra_arguments["transfer"]:

            self.bot.reset()

        if "env" in self.args.extra_arguments["transfer"]:

            self.bot.emulator_state = self.get_emulator_state(self.emulator)
            self.bot.obs = obs.copy()

        else:

            if self.log:
                if "fit" in self.args.extra_arguments["transfer"]:
                    print(self.bot.run_score)

    def run(self, gen_nb: int) -> float:

        self.log = False

        if not hasattr(self, "emulator"):
            raise NotImplementedError(
                "Score Multistep Environments require the attribute "
                "'emulator': the emulator to run the agents on."
            )

        [self.bot] = self.bots
        self.bot.run_score = 0
        self.bot.run_nb_steps = 0

        for trial in range(self.args.extra_arguments["trials"]):

            obs, done, nb_obs = self.reset(gen_nb, trial), False, 0

            while not done:

                obs, rew, terminated, truncated, _ = self.emulator.step(
                    self.bot(obs)
                )

                self.bot.run_score += rew
                self.bot.run_nb_steps += 1
                nb_obs += 1

                if self.log == True:
                    if "env" in self.args.extra_arguments["transfer"]:
                        self.bot.episode_score += rew

                if terminated or truncated:
                    obs, done = self.done_reset(gen_nb)

                if nb_obs == self.args.extra_arguments["steps"]:
                    done = True

            self.final_reset(obs)

        if "fit" in self.args.extra_arguments["transfer"]:

            self.bot.continual_fitness += self.bot.run_score

            return self.bot.continual_fitness, self.bot.run_nb_steps

        else:
            return self.bot.run_score, self.bot.run_nb_steps
