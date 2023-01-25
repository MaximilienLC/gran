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

from gran.rand.env.base import BaseEnv
from gran.util.misc import cfg


class BaseImitateEnv(BaseEnv):
    """
    Base Imitate Env class.
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

        assert hasattr(self, "hide_score"), "Attribute "
        "'hide_score' required: a function that hides the "
        "emulator's score."

        assert not (cfg.num_steps == 0 and "env" in cfg.transfer)

        assert cfg.merge in ["yes", "no"]

        super().__init__(cfg, io_path="IO.imitate.sb3", num_pops=2)

    def reset(self, curr_gen: int) -> np.ndarray:
        """
        First reset function called during the match.
        Used to either set the emulator seed or resume from a previous state.

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
                self.curr_actor_data_holder.saved_emulator_seed = new_seed

            obs = self.reset_emulator_state(
                self.curr_emulator,
                self.curr_actor_data_holder.saved_emulator_seed,
            )

            if curr_gen > 0:

                self.set_emulator_state(
                    self.curr_emulator,
                    self.curr_actor_data_holder.saved_emulator_state,
                )

                obs = self.curr_actor_data_holder.saved_emulator_obs.copy()

            if self.curr_actor == self.imitation_target:
                self.imitation_target.reset(
                    self.curr_actor_data_holder.saved_emulator_seed,
                    self.curr_actor_data_holder.curr_episode_num_steps,
                )

        else:  # cfg.transfer in ["no", "fit"]:

            obs = self.reset_emulator_state(self.curr_emulator, new_seed)

            if self.curr_actor == self.imitation_target:
                self.imitation_target.reset(new_seed, 0)

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
        if self.curr_actor == self.generator:
            self.generator.reset()

        self.discriminator.reset()

        if "env" in cfg.transfer:

            if cfg.wandb_logging:
                if self.curr_actor == self.generator:  # sanity check target
                    wandb.log(
                        {
                            "score": self.generator.curr_episode_score,
                            "gen": curr_gen,
                        }
                    )

                self.generator.curr_episode_score = 0

            self.curr_actor_data_holder.curr_episode_num_steps = 0

            obs = self.reset_emulator_state(
                self.curr_emulator,
                self.curr_actor_data_holder.saved_emulator_seed,
            )

            return obs, False

        else:  # cfg.transfer in ["no", "fit"]:

            return np.empty(0), True

    def final_reset(self, obs: np.ndarray) -> None:
        """
        Reset function called at the end of every match.

        Args:
            obs - The final environment observation.
        """
        if "mem" not in cfg.transfer:

            if self.curr_actor == self.generator:
                self.generator.reset()

            self.discriminator.reset()

        if "env" in cfg.transfer:

            self.curr_actor_data_holder.saved_emulator_state = (
                self.get_emulator_state(self.curr_emulator)
            )

            self.curr_actor_data_holder.saved_emulator_obs = obs.copy()

        else:  # cfg.transfer in ["no", "fit"]:

            if cfg.wandb_logging:
                if self.curr_actor == self.generator:  # sanity check target
                    wandb.log(
                        {
                            "score": self.generator.curr_run_score,
                            "gen": curr_gen,
                        }
                    )

    def run_bots(self, curr_gen: int) -> float:

        assert hasattr(self, "emulator"), "Attribute "
        "'emulator' required: the emulator to run the agents on."

        assert hasattr(self, "imitation_target"), "Attribute "
        "'emulator' required: the target behaviour/agent to imitate."

        [self.generator, self.discriminator] = self.bots
        generator_fitness, discriminator_fitness = 0, 0

        # 0) Generator & Discriminator 1) Imitation Target & Discriminator
        for match in [0, 1]:

            self.curr_emulator = self.emulators[-match]  # 1 or 2 emulators

            if match == 0:
                self.curr_actor = self.generator
                self.curr_actor_data_holder = self.generator
            else:  # match == 1:
                self.curr_actor = self.imitation_target
                self.curr_actor_data_holder = self.discriminator

            self.curr_actor_data_holder.curr_run_score = 0
            self.curr_actor_data_holder.curr_run_num_steps = 0

            obs, done, p_imitation_target = self.reset(curr_gen), False, 0
            hidden_score_obs = self.hide_score(obs, cfg.task)
            num_obs = 0

            while not done:

                if self.generator == self.curr_actor:
                    output = self.generator(hidden_score_obs)
                else:  # self.imitation_target == self.curr_actor:
                    output = self.imitation_target(obs)

                obs, rew, done = self.run_emulator_step(
                    self.curr_emulator, output
                )
                hidden_score_obs = self.hide_score(obs, cfg.task)

                num_obs += 1

                self.curr_actor_data_holder.curr_run_score += rew
                self.curr_actor_data_holder.curr_run_num_steps += 1

                if "env" in cfg.transfer:
                    self.curr_actor_data_holder.curr_episode_score += rew
                    self.curr_actor_data_holder.curr_episode_num_steps += 1

                p_imitation_target += self.discriminator(hidden_score_obs)

                if self.curr_actor == self.imitation_target:
                    done = self.imitation_target.is_done

                if done:

                    obs, done = self.done_reset(curr_gen)
                    num_obs = 0

                    if not done:
                        p_imitation_target = 0

                if num_obs == cfg.num_steps:
                    done = True

            p_imitation_target /= cfg.num_steps

            if self.curr_actor == self.generator:
                generator_fitness += p_imitation_target
                discriminator_fitness -= p_imitation_target
            else:  # self.curr_actor == self.imitation_target:
                discriminator_fitness += p_imitation_target

            self.final_reset(obs)

        if cfg.merge == "no":
            # Scale discriminator fitnesses to [0, 1]
            discriminator_fitness = (discriminator_fitness + 1) / 2

        else:  # cfg.merge == "yes":
            # Scale generator & discriminator fitnesses to [0, .5]
            generator_fitness = generator_fitness / 2
            discriminator_fitness = (discriminator_fitness + 1) / 4

        if "fit" in cfg.transfer:

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
