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

from typing import Tuple

import numpy as np
import wandb
from omegaconf import DictConfig

from gran.rands.envs.base import EnvBase


class CPUScoreEnvBase(EnvBase):
    """
    CPU Score Env Base class.
    Concrete subclasses need to be named *Env*.
    """

    def __init__(self, cfg: DictConfig) -> None:

        assert hasattr(self, "get_emulator_state"), "Attribute "
        "'get_emulator_state' required: a function that returns the "
        "emulator's state."

        assert hasattr(self, "set_emulator_state"), "Attribute "
        "'set_emulator_state' required: a function that sets the"
        "emulator's state."

        assert hasattr(self, "reset_emulator_state"), "Attribute "
        "'reset_emulator_state' required: a function that resets the "
        "emulator with or without the help of a random seed."

        assert hasattr(self, "run_emulator_step"), "Attribute "
        "'run_emulator_step' required: a function that runs an "
        "emulator step given an action."

        assert not (
            cfg.rands.env.steps == 0 and "env" in cfg.rands.env.transfer
        ), "'env' in 'cfg.rands.env.transfer' requires "
        "'cfg.rands.env.steps' > 0"

        ge_1_int = lambda x: isinstance(x, int) and x >= 1

        assert ge_1_int(cfg.rands.env.trials), "'cfg.rands.trials' needs to "
        "be an integer >= 1."

        assert not (
            cfg.rands.env.trials > 1 and "fit" in cfg.rands.env.transfer
        ), "'cfg.rands.trials' > 1 requires 'cfg.rands.transfer' = 'no'."

        super().__init__(cfg, io_path="IO.base", nb_pops=1)

    def reset(self, gen_nb: int, trial_nb: int) -> np.ndarray:
        """
        First reset function called during the run.
        Used to reset the emulator & potentially resume from a previous state.

        Args:
            gen_nb - Generation number.
            trial_nb - Trial number.
        Returns:
            np.ndarray - The initial environment observation.
        """
        if isinstance(self.cfg.rands.env.seeding, int):
            new_seed = self.cfg.rands.env.seeding
        else:
            new_seed = gen_nb * self.cfg.rands.env.trials + trial_nb

        if "env" in self.cfg.rands.env.transfer:

            if gen_nb == 0:
                self.bot.saved_emulator_seed = new_seed

            obs = self.reset_emulator_state(
                self.emulator, self.bot.saved_emulator_seed
            )

            if gen_nb > 0:

                self.set_emulator_state(
                    self.emulator, self.bot.saved_emulator_state
                )

                obs = self.bot.saved_emulator_obs.copy()

        else:  # self.cfg.rands.env.transfer in ["no", "fit"]:

            obs = self.reset_emulator_state(self.emulator, new_seed)

        return obs

    def done_reset(self, gen_nb: int) -> Tuple[np.ndarray, bool]:
        """
        Reset function called whenever the emulator returns done.

        Args:
            gen_nb - Generation number.
        Returns:
            np.ndarray - A new environment observation (np.ndarray).
            bool - Whether the episode should terminate.
        """
        self.bot.reset()

        if "env" in self.cfg.rands.env.transfer:

            if self.cfg.rands.wandb_logging:

                wandb.log(
                    {"score": self.bot.current_episode_score, "gen": gen_nb}
                )

                self.bot.current_episode_score = 0

            if self.cfg.rands.env.seeding == "reg":
                self.bot.saved_emulator_seed = gen_nb

            obs = self.reset_emulator_state(
                self.emulator, self.bot.saved_emulator_seed
            )

            return obs, False

        else:  # self.cfg.rands.env.transfer in ["no", "fit"]:

            return np.empty(0), True

    def final_reset(self, obs: np.ndarray) -> None:
        """
        Reset function called at the end of every run.

        Args:
            obs - The final environment observation.
        """
        if "mem" not in self.cfg.rands.env.transfer:

            self.bot.reset()

        if "env" in self.cfg.rands.env.transfer:

            self.bot.saved_emulator_state = self.get_emulator_state(
                self.emulator
            )

            self.bot.saved_emulator_obs = obs.copy()

        else:  # self.cfg.rands.env.transfer in ["no", "fit"]:

            if self.cfg.rands.wandb_logging:

                wandb.log(
                    {
                        "score": self.bot.current_run_score,
                        "gen": gen_nb,
                    }
                )

    def run_bots(self, gen_nb: int) -> float:

        assert hasattr(self, "emulator"), "Attribute "
        "'emulator' required: the emulator to run the agents on."

        [self.bot] = self.bots
        self.bot.current_run_score = 0
        self.bot.current_run_nb_steps = 0

        for trial in range(self.cfg.rands.env.trials):

            obs, done, nb_obs = self.reset(gen_nb, trial), False, 0

            while not done:

                obs, rew, done = self.run_emulator_step(
                    self.emulator, self.bot(obs)
                )

                nb_obs += 1

                self.bot.current_run_score += rew
                self.bot.current_run_nb_steps += 1

                if "env" in self.cfg.rands.env.transfer:
                    self.bot.current_episode_score += rew
                    self.bot.current_episode_nb_steps += 1

                if done:
                    obs, done = self.done_reset(gen_nb)
                    nb_obs = 0

                if nb_obs == self.cfg.rands.env.steps:
                    done = True

            self.final_reset(obs)

        if "fit" in self.cfg.rands.env.transfer:

            self.bot.continual_fitness += self.bot.current_run_score

            return np.array(
                (self.bot.continual_fitness, self.bot.current_run_nb_steps)
            )

        else:
            return np.array(
                (self.bot.current_run_score, self.bot.current_run_nb_steps)
            )
