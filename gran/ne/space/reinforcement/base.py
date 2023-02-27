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

from gran.ne.space.base import BaseSpace


class BaseReinforcementSpace(BaseSpace):
    """
    Base Reinforcement Space class. Inside Reinforcement Spaces, agents evolve
    to maximize a reward function.
    Concrete subclasses need to be named *Space*.
    """

    def __init__(self, config) -> None:

        assert not (
            config.ecosystem.run_num_steps == "infinite"
            and "env" in config.ecosystem.transfer
        )

        super().__init__(config, "IO.base", num_pops=1)

    def init_reset(self, curr_gen: int) -> np.ndarray:
        """
        First reset function called during the run. Used to reset the
        environment & potentially resume from a previous state.

        Args:
            curr_gen - Current generation.
        Returns:
            np.ndarray - The initial environment observation.
        """
        if "env" in config.ecosystem.transfer:

            if curr_gen == 0:
                self.agent.saved_env_seed = curr_gen

            obs = self.env.reset(self.agent.saved_env_seed)

            if curr_gen > 0:

                self.env.set_state(self.agent.saved_env_state)

                obs = self.agent.saved_env_obs.copy()

        else:  # config.ecosystem.transfer in ["no", "fit"]:

            obs = self.env.reset(curr_gen)

        return obs

    def done_reset(self, curr_gen: int) -> Tuple[np.ndarray, bool]:
        """
        Reset function called whenever the environment returns done.

        Args:
            curr_gen - Current generation.
        Returns:
            np.ndarray - A new environment observation (np.ndarray).
            bool - Whether the episode should terminate.
        """
        self.agent.reset()

        if "env" in config.ecosystem.transfer:

            if config.wandb.mode != "disabled":

                wandb.log(
                    {"score": self.agent.curr_episode_score, "gen": curr_gen}
                )

            self.agent.curr_episode_score = 0
            self.agent.curr_episode_num_steps = 0

            self.agent.saved_env_seed = curr_gen

            obs = self.env.reset(self.agent.saved_env_seed)

            return obs, False

        else:  # config.ecosystem.transfer in ["no", "fit"]:

            return np.empty(0), True

    def final_reset(self, obs: np.ndarray) -> None:
        """
        Reset function called at the end of every run.

        Args:
            obs - The final environment observation.
        """
        if "mem" not in config.ecosystem.transfer:
            self.agent.reset()

        if "env" in config.ecosystem.transfer:
            self.agent.saved_env_state = self.env.get_state()
            self.agent.saved_env_obs = obs.copy()

        else:  # config.ecosystem.transfer in ["no", "fit"]:

            if config.wandb.mode != "disabled":

                wandb.log(
                    {
                        "score": self.agent.curr_run_score,
                        "gen": curr_gen,
                    }
                )

    def run_agents(self, curr_gen: int) -> float:

        [self.agent] = self.agents
        self.agent.curr_run_score = 0
        self.agent.curr_run_num_steps = 0

        obs, done = self.init_reset(curr_gen), False

        while not done:

            obs, rew, done = self.env.step(self.agent(obs))

            self.agent.curr_run_score += rew
            self.agent.curr_run_num_steps += 1

            if "env" in config.ecosystem.transfer:
                self.agent.curr_episode_score += rew
                self.agent.curr_episode_num_steps += 1

            if "fit" in config.ecosystem.transfer:
                self.agent.continual_fitness += rew

            if done:
                obs, done = self.done_reset(curr_gen)

            if self.agent.curr_run_num_steps == config.ecosystem.run_num_steps:
                done = True

        self.final_reset(obs)

        if "fit" in config.ecosystem.transfer:

            return np.array(
                (self.agent.continual_fitness, self.agent.curr_run_num_steps)
            )

        else:

            return np.array(
                (self.agent.curr_run_score, self.agent.curr_run_num_steps)
            )
