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

from gran.nevo.space.base import BaseSpace
from gran.util.misc import cfg


class BaseReinforcementSpace(BaseSpace):
    """
    Base Reinforcement Space class. Inside Reinforcement Spaces, agents evolve
    to maximize a reward function.
    Concrete subclasses need to be named *Space*.
    """

    def __init__(self, io_path="IO.base") -> None:
        """
        .
        """
        self.verify_assertions()

        super().__init__(io_path, num_pops=1)

    def verify_assertions(self) -> None:
        """
        Verify Assertions.
        """
        assert hasattr(self, "get_env_state"), "Attribute 'get_env_state' "
        "required: a function that returns the environment's state."

        assert hasattr(self, "set_env_state"), "Attribute 'set_env_state' "
        "required: a function that sets the environment's state."

        assert hasattr(self, "reset_env_state"), "Attribute 'reset_env_state' "
        "required: a function that resets the environment with or without the "
        "help of a random seed."

        assert hasattr(self, "run_env_step"), "Attribute 'run_env_step' "
        "required: a function that runs an environment step given an action."

        assert not (
            cfg.ecosystem.run_num_steps == "infinite"
            and "env" in cfg.ecosystem.gen_transfer
        )

    def reset(self, curr_gen: int) -> np.ndarray:
        """
        First reset function called during the run. Used to reset the
        environment & potentially resume from a previous state.

        Args:
            curr_gen - Current generation.
        Returns:
            np.ndarray - The initial environment observation.
        """
        if cfg.seed != "scheduled":
            new_seed = cfg.seed
        else:
            new_seed = curr_gen

        if "env" in cfg.ecosystem.gen_transfer:
            if curr_gen == 0:
                self.agent.saved_env_seed = new_seed

            obs = self.reset_env_state(self.env, self.agent.saved_env_seed)

            if curr_gen > 0:
                self.set_env_state(self.env, self.agent.saved_env_state)

                obs = self.agent.saved_env_obs.copy()

        else:  # cfg.ecosystem.gen_transfer in ["no", "fit"]:
            obs = self.reset_env_state(self.env, new_seed)

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

        if "env" in cfg.ecosystem.gen_transfer:
            if cfg.wandb_logging:
                wandb.log(
                    {"score": self.agent.curr_episode_score, "gen": curr_gen}
                )

                self.agent.curr_episode_score = 0

            obs = self.reset_env_state(self.env, self.agent.saved_env_seed)

            return obs, False

        else:  # cfg.ecosystem.gen_transfer in ["no", "fit"]:
            return np.empty(0), True

    def final_reset(self, obs: np.ndarray) -> None:
        """
        Reset function called at the end of every run.

        Args:
            obs - The final environment observation.
        """
        if "mem" not in cfg.ecosystem.gen_transfer:
            self.agent.reset()

        if "env" in cfg.ecosystem.gen_transfer:
            self.agent.saved_env_state = self.get_env_state(self.env)

            self.agent.saved_env_obs = obs.copy()

        else:  # cfg.ecosystem.gen_transfer in ["no", "fit"]:
            if cfg.wandb_logging:
                wandb.log(
                    {
                        "score": self.agent.curr_run_score,
                        "gen": curr_gen,
                    }
                )

    def run_agents(self, curr_gen: int) -> float:
        """
        Run agents.
        """
        if curr_gen == 0:
            assert hasattr(self, "env"), "Attribute 'env' required: the "
            "environment to run the agents on."

        [self.agent] = self.agents
        self.agent.curr_run_score = 0
        self.agent.curr_run_num_steps = 0

        obs, done, num_obs = self.reset(curr_gen), False, 0

        while not done:
            obs, rew, done = self.run_env_step(self.env, self.agent(obs))

            num_obs += 1

            self.agent.curr_run_score += rew
            self.agent.curr_run_num_steps += 1

            if "env" in cfg.ecosystem.gen_transfer:
                self.agent.curr_episode_score += rew
                self.agent.curr_episode_num_steps += 1

            if done:
                obs, done = self.done_reset(curr_gen)
                num_obs = 0

            if num_obs == cfg.ecosystem.run_num_steps:
                done = True

        self.final_reset(obs)

        if "fit" in cfg.ecosystem.gen_transfer:
            self.agent.continual_fitness += self.agent.curr_run_score

            return np.array(
                (self.agent.continual_fitness, self.agent.curr_run_num_steps)
            )

        else:
            return np.array(
                (self.agent.curr_run_score, self.agent.curr_run_num_steps)
            )
