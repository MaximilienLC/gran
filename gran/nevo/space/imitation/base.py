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

from abc import ABCMeta, abstractmethod
from typing import final, Tuple

import numpy as np
import wandb

from gran.nevo.space.base import BaseSpace
from gran.util.misc import config


class BaseImitationSpace(BaseSpace, metaclass=ABCMeta):
    """
    Base Imitation Space class. Inside Imitation Spaces, agents evolve
    to imitate a target.
    Concrete subclasses need to be named *Env*.
    """

    def __init__(self) -> None:

        assert hasattr(self, "envs"), "Attribute 'envs' required: a list of "
        "one of two environments to run the generator and target."

        assert hasattr(self, "hide_score"), "Attribute 'hide_score' "
        "required: a function that hides the environment's score portion of "
        "the screen to prevent the discriminator agent from utilizing it."

        # config.ecosystem.gen_transfer can be either bool or string
        if config.ecosystem.gen_transfer:
            assert not (
                config.ecosystem.run_num_steps == "infinite"
                and "env" in config.ecosystem.gen_transfer
            )

        assert isinstance(config.ecosystem.pop_merge, bool)

        super().__init__(io_path="gran.nevo.IO.imitation.sb3", num_pops=2)

    @final
    def init_reset(self, curr_gen: int) -> np.ndarray:
        """
        First reset function called during the match.
        Used to either set the env seed or resume from a previous state.

        Args:
            curr_gen - Current generation.
        Returns:
            np.ndarray - The initial environment observation.
        """
        if (
            config.ecosystem.gen_transfer
            and "env" in config.ecosystem.gen_transfer
        ):

            if curr_gen == 0:
                self.curr_actor_data_holder.saved_env_seed = curr_gen

            obs = self.curr_env.reset(
                self.curr_actor_data_holder.saved_env_seed
            )

            if curr_gen > 0:

                self.curr_env.set_state(
                    self.curr_actor_data_holder.saved_env_state
                )

                obs = self.curr_actor_data_holder.saved_env_obs.copy()

            if self.imitation_target == self.curr_actor:

                self.imitation_target.reset(
                    self.curr_actor_data_holder.saved_env_seed,
                    self.curr_actor_data_holder.curr_episode_num_steps,
                )

        else:  # config.ecosystem.gen_transfer in [False, "fit"]:

            obs = self.curr_env.reset(curr_gen)

            if self.imitation_target == self.curr_actor:
                self.imitation_target.reset(curr_gen, 0)

        return obs

    @final
    def done_reset(self, curr_gen: int) -> Tuple[np.ndarray, bool]:
        """
        Reset function called whenever the env returns done.

        Args:
            curr_gen - Current generation.
        Returns:
            np.ndarray - A new environment observation (np.ndarray).
            bool - Whether the episode should terminate.
        """
        if self.generator == self.curr_actor:
            self.generator.reset()

        self.discriminator.reset()

        if (
            config.ecosystem.gen_transfer
            and "env" in config.ecosystem.gen_transfer
        ):

            if self.generator == self.curr_actor:  # check target

                if config.wandb.mode != "disabled":

                    wandb.log(
                        {
                            "score": self.generator.curr_episode_score,
                            "gen": curr_gen,
                        }
                    )

                self.curr_actor_data_holder.curr_episode_score = 0

            self.curr_actor_data_holder.curr_episode_num_steps = 0

            self.curr_actor_data_holder.saved_env_seed = curr_gen

            obs = self.curr_env.reset(
                self.curr_actor_data_holder.saved_env_seed
            )

            return obs, False

        else:  # config.ecosystem.gen_transfer in [False, "fit"]:

            return np.empty(0), True

    @final
    def final_reset(self, obs: np.ndarray) -> None:
        """
        Reset function called at the end of every match.

        Args:
            obs - The final environment observation.
        """
        if (
            config.ecosystem.gen_transfer
            and "mem" not in config.ecosystem.gen_transfer
        ):

            if self.generator == self.curr_actor:
                self.generator.reset()

            self.discriminator.reset()

        if (
            config.ecosystem.gen_transfer
            and "env" in config.ecosystem.gen_transfer
        ):

            self.curr_actor_data_holder.saved_env_state = self.get_env_state(
                self.curr_env
            )

            self.curr_actor_data_holder.saved_env_obs = obs.copy()

        if config.ecosystem.gen_transfer in [False, "fit"]:

            if self.generator == self.curr_actor:  # check target

                if config.wandb.mode != "disabled":

                    wandb.log(
                        {
                            "score": self.generator.curr_run_score,
                            "gen": curr_gen,
                        }
                    )

    @final
    def run_agents(self, curr_gen: int) -> float:

        [self.generator, self.discriminator] = self.agents
        generator_fitness, discriminator_fitness = 0, 0

        # 0) Generator & Discriminator 1) Imitation Target & Discriminator
        for match in [0, 1]:

            self.curr_env = self.envs[-match]  # 1 or 2 envs

            if match == 0:
                self.curr_actor = self.generator
                self.curr_actor_data_holder = self.generator
            else:  # match == 1:
                self.curr_actor = self.imitation_target
                self.curr_actor_data_holder = self.discriminator

            self.curr_actor_data_holder.curr_run_score = 0
            self.curr_actor_data_holder.curr_run_num_steps = 0

            obs, done, p_imitation_target = self.init_reset(curr_gen), False, 0
            hidden_score_obs = self.hide_score(obs, config.task)

            while not done:

                if self.generator == self.curr_actor:
                    output = self.generator(hidden_score_obs)
                else:  # self.imitation_target == self.curr_actor:
                    output = self.imitation_target(obs)

                obs, rew, done = self.curr_env.step(output)
                hidden_score_obs = self.hide_score(obs, config.task)

                self.curr_actor_data_holder.curr_run_score += rew
                self.curr_actor_data_holder.curr_run_num_steps += 1

                if (
                    config.ecosystem.gen_transfer
                    and "env" in config.ecosystem.gen_transfer
                ):
                    self.curr_actor_data_holder.curr_episode_score += rew
                    self.curr_actor_data_holder.curr_episode_num_steps += 1

                p_imitation_target += self.discriminator(hidden_score_obs)

                if self.imitation_target == self.curr_actor:
                    if self.imitation_target.is_done:
                        done = True

                if done:
                    obs, done = self.done_reset(curr_gen)

                if (
                    self.curr_actor_data_holder.curr_run_num_steps
                    == config.ecosystem.run_num_steps
                ):
                    done = True

            p_imitation_target /= (
                self.curr_actor_data_holder.curr_run_num_steps
            )

            if self.generator == self.curr_actor:
                generator_fitness += p_imitation_target
                discriminator_fitness -= p_imitation_target
            else:  # self.imitation_target == self.curr_actor:
                discriminator_fitness += p_imitation_target

            self.final_reset(obs)

        if config.ecosystem.pop_merge:

            # Scale generator & discriminator fitnesses to [0, .5]
            generator_fitness = generator_fitness / 2
            discriminator_fitness = (discriminator_fitness + 1) / 4

        else:

            # Scale discriminator fitnesses to [0, 1]
            discriminator_fitness = (discriminator_fitness + 1) / 2

        if (
            config.ecosystem.gen_transfer
            and "fit" in config.ecosystem.gen_transfer
        ):

            self.generator.continual_fitness += generator_fitness
            self.discriminator.continual_fitness += discriminator_fitness

            return np.array(
                (
                    self.generator.continual_fitness,
                    self.discriminator.continual_fitness,
                ),
                (
                    self.generator.curr_run_num_steps,
                    self.discriminator.curr_run_num_steps,
                ),
            )

        else:
            return np.array(
                (
                    generator_fitness,
                    discriminator_fitness,
                ),
                (
                    self.generator.curr_run_num_steps,
                    self.discriminator.curr_run_num_steps,
                ),
            )
