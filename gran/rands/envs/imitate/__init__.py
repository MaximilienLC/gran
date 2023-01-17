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

from gran.rands.envs import BaseEnv


class BaseImitateEnv(BaseEnv):
    """
    Base Imitate Env class.
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

        assert hasattr(self, "hide_score"), "Attribute "
        "'hide_score' required: a function that hides the "
        "emulator's score."

        assert not (
            cfg.env.steps == 0 and "env" in cfg.env.transfer
        ), "'env' in 'cfg.env.transfer' requires "
        "'cfg.env.steps' > 0"

        assert cfg.merge in ["yes", "no"], "'cfg.merge' "
        "needs be chosen from one of " + str(["yes", "no"])

        super().__init__(cfg, io_path="IO.imitate.sb3", nb_pops=2)

    def reset(self, gen_nb: int) -> np.ndarray:
        """
        First reset function called during the match.
        Used to either set the emulator seed or resume from a previous state.

        Args:
            gen_nb - Generation number.
        Returns:
            np.ndarray - The initial environment observation.
        """
        if isinstance(self.cfg.env.seeding, int):
            new_seed = self.cfg.env.seeding
        else:
            new_seed = gen_nb

        if "env" in self.cfg.env.transfer:

            if gen_nb == 0:
                self.current_actor_data_holder.saved_emulator_seed = new_seed

            obs = self.reset_emulator_state(
                self.current_emulator,
                self.current_actor_data_holder.saved_emulator_seed,
            )

            if gen_nb > 0:

                self.set_emulator_state(
                    self.current_emulator,
                    self.current_actor_data_holder.saved_emulator_state,
                )

                obs = self.current_actor_data_holder.saved_emulator_obs.copy()

            if self.current_actor == self.imitation_target:
                self.imitation_target.reset(
                    self.current_actor_data_holder.saved_emulator_seed,
                    self.current_actor_data_holder.current_episode_nb_steps,
                )

        else:  # self.cfg.env.transfer in ["no", "fit"]:

            obs = self.reset_emulator_state(self.current_emulator, new_seed)

            if self.current_actor == self.imitation_target:
                self.imitation_target.reset(new_seed, 0)

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
        if self.current_actor == self.generator:
            self.generator.reset()

        self.discriminator.reset()

        if "env" in self.cfg.env.transfer:

            if self.cfg.wandb_logging:
                if self.current_actor == self.generator:  # sanity check target
                    wandb.log(
                        {
                            "score": self.generator.current_episode_score,
                            "gen": gen_nb,
                        }
                    )

                self.generator.current_episode_score = 0

            if self.cfg.env.seeding == "reg":
                self.current_actor_data_holder.saved_emulator_seed = gen_nb
            self.current_actor_data_holder.current_episode_nb_steps = 0

            obs = self.reset_emulator_state(
                self.current_emulator,
                self.current_actor_data_holder.saved_emulator_seed,
            )

            return obs, False

        else:  # self.cfg.env.transfer in ["no", "fit"]:

            return np.empty(0), True

    def final_reset(self, obs: np.ndarray) -> None:
        """
        Reset function called at the end of every match.

        Args:
            obs - The final environment observation.
        """
        if "mem" not in self.cfg.env.transfer:

            if self.current_actor == self.generator:
                self.generator.reset()

            self.discriminator.reset()

        if "env" in self.cfg.env.transfer:

            self.current_actor_data_holder.saved_emulator_state = (
                self.get_emulator_state(self.current_emulator)
            )

            self.current_actor_data_holder.saved_emulator_obs = obs.copy()

        else:  # self.cfg.env.transfer in ["no", "fit"]:

            if self.cfg.wandb_logging:
                if self.current_actor == self.generator:  # sanity check target
                    wandb.log(
                        {
                            "score": self.generator.current_run_score,
                            "gen": gen_nb,
                        }
                    )

    def run_bots(self, gen_nb: int) -> float:

        assert hasattr(self, "emulator"), "Attribute "
        "'emulator' required: the emulator to run the agents on."

        assert hasattr(self, "imitation_target"), "Attribute "
        "'emulator' required: the target behaviour/agent to imitate."

        [self.generator, self.discriminator] = self.bots
        generator_fitness, discriminator_fitness = 0, 0

        # 0) Generator & Discriminator 1) Imitation Target & Discriminator
        for match in [0, 1]:

            self.current_emulator = self.emulators[-match]  # 1 or 2 emulators

            if match == 0:
                self.current_actor = self.generator
                self.current_actor_data_holder = self.generator
            else:  # match == 1:
                self.current_actor = self.imitation_target
                self.current_actor_data_holder = self.discriminator

            self.current_actor_data_holder.current_run_score = 0
            self.current_actor_data_holder.current_run_nb_steps = 0

            obs, done, p_imitation_target = self.reset(gen_nb), False, 0
            hidden_score_obs = self.hide_score(obs, self.cfg.task)
            nb_obs = 0

            while not done:

                if self.generator == self.current_actor:
                    output = self.generator(hidden_score_obs)
                else:  # self.imitation_target == self.current_actor:
                    output = self.imitation_target(obs)

                obs, rew, done = self.run_emulator_step(
                    self.current_emulator, output
                )
                hidden_score_obs = self.hide_score(obs, self.cfg.task)

                nb_obs += 1

                self.current_actor_data_holder.current_run_score += rew
                self.current_actor_data_holder.current_run_nb_steps += 1

                if "env" in self.cfg.transfer:
                    self.current_actor_data_holder.current_episode_score += rew
                    self.current_actor_data_holder.current_episode_nb_steps += (
                        1
                    )

                p_imitation_target += self.discriminator(hidden_score_obs)

                if self.current_actor == self.imitation_target:
                    done = self.imitation_target.is_done

                if done:

                    obs, done = self.done_reset(gen_nb)
                    nb_obs = 0

                    if not done:
                        p_imitation_target = 0

                if nb_obs == self.cfg.env.steps:
                    done = True

            p_imitation_target /= self.cfg.steps

            if self.current_actor == self.generator:
                generator_fitness += p_imitation_target
                discriminator_fitness -= p_imitation_target
            else:  # self.current_actor == self.imitation_target:
                discriminator_fitness += p_imitation_target

            self.final_reset(obs)

        if self.cfg.merge == "no":
            # Scale discriminator fitnesses to [0, 1]
            discriminator_fitness = (discriminator_fitness + 1) / 2

        else:  # self.cfg.merge == "yes":
            # Scale generator & discriminator fitnesses to [0, .5]
            generator_fitness = generator_fitness / 2
            discriminator_fitness = (discriminator_fitness + 1) / 4

        if "fit" in self.cfg.transfer:

            self.generator.continual_fitness += generator_fitness
            self.discriminator.continual_fitness += discriminator_fitness

            return np.array(
                (
                    self.generator.continual_fitness,
                    self.discriminator.continual_fitness,
                ),
                (
                    self.generator.current_run_nb_steps,
                    self.discriminator.current_run_nb_steps,
                ),
            )

        else:

            return np.array(
                (
                    generator_fitness,
                    discriminator_fitness,
                ),
                (
                    self.generator.current_run_nb_steps,
                    self.discriminator.current_run_nb_steps,
                ),
            )
