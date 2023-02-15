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

from typing import Tuple

import numpy as np
import wandb

from gran.rand.env.base import BaseEnv
from gran.util.misc import cfg


class BaseScoreEnv(BaseEnv):
    """
    Base Score Env class.
    Concrete subclasses need to be named *Env*.
    """

    def __init__(self) -> None:
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

        assert not (cfg.num_steps == 0 and "env" in cfg.transfer)

        super().__init__(io_path="IO.base", num_pops=1)

    def reset(self, curr_gen: int) -> np.ndarray:
        """
        First reset function called during the run.
        Used to reset the emulator & potentially resume from a previous state.

        Args:
            curr_gen - Current generation.
        Returns:
            np.ndarray - The initial environment observation.
        """
        if cfg.seed >= 0:
            new_seed = cfg.seed
        else:
            new_seed = curr_gen

        if "env" in cfg.transfer:
            if curr_gen == 0:
                self.bot.saved_emulator_seed = new_seed

            obs = self.reset_emulator_state(
                self.emulator, self.bot.saved_emulator_seed
            )

            if curr_gen > 0:
                self.set_emulator_state(
                    self.emulator, self.bot.saved_emulator_state
                )

                obs = self.bot.saved_emulator_obs.copy()

        else:  # cfg.transfer in ["no", "fit"]:
            obs = self.reset_emulator_state(self.emulator, new_seed)

        return obs

    def done_reset(self, curr_gen: int) -> Tuple[np.ndarray, bool]:
        """
        Reset function called whenever the emulator returns done.

        Args:
            curr_gen - Current generation.
        Returns:
            np.ndarray - A new environment observation (np.ndarray).
            bool - Whether the episode should terminate.
        """
        self.bot.reset()

        if "env" in cfg.transfer:
            if cfg.wandb_logging:
                wandb.log(
                    {"score": self.bot.curr_episode_score, "gen": curr_gen}
                )

                self.bot.curr_episode_score = 0

            obs = self.reset_emulator_state(
                self.emulator, self.bot.saved_emulator_seed
            )

            return obs, False

        else:  # cfg.transfer in ["no", "fit"]:
            return np.empty(0), True

    def final_reset(self, obs: np.ndarray) -> None:
        """
        Reset function called at the end of every run.

        Args:
            obs - The final environment observation.
        """
        if "mem" not in cfg.transfer:
            self.bot.reset()

        if "env" in cfg.transfer:
            self.bot.saved_emulator_state = self.get_emulator_state(
                self.emulator
            )

            self.bot.saved_emulator_obs = obs.copy()

        else:  # cfg.transfer in ["no", "fit"]:
            if cfg.wandb_logging:
                wandb.log(
                    {
                        "score": self.bot.curr_run_score,
                        "gen": curr_gen,
                    }
                )

    def run_bots(self, curr_gen: int) -> float:
        assert hasattr(self, "emulator"), "Attribute "
        "'emulator' required: the emulator to run the agents on."

        [self.bot] = self.bots
        self.bot.curr_run_score = 0
        self.bot.curr_run_num_steps = 0

        obs, done, num_obs = self.reset(curr_gen), False, 0

        while not done:
            obs, rew, done = self.run_emulator_step(
                self.emulator, self.bot(obs)
            )

            num_obs += 1

            self.bot.curr_run_score += rew
            self.bot.curr_run_num_steps += 1

            if "env" in cfg.transfer:
                self.bot.curr_episode_score += rew
                self.bot.curr_episode_num_steps += 1

            if done:
                obs, done = self.done_reset(curr_gen)
                num_obs = 0

            if num_obs == cfg.num_steps:
                done = True

        self.final_reset(obs)

        if "fit" in cfg.transfer:
            self.bot.continual_fitness += self.bot.curr_run_score

            return np.array(
                (self.bot.continual_fitness, self.bot.curr_run_num_steps)
            )

        else:
            return np.array(
                (self.bot.curr_run_score, self.bot.curr_run_num_steps)
            )
