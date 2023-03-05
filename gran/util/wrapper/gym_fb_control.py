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

import copy

import gymnasium
import numpy as np

from gran.util.misc import config
from gran.util.wrapper.base import BaseWrapper


class GymFeatureBasedControlWrapper(BaseWrapper):
    """
    .
    """

    def __init__(self):
        """
        .
        """
        self.imitation_envs = [
            "acrobot",
            "cart_pole",
            "mountain_car",
            "mountain_car_continuous",
            "pendulum",
            "bipedal_walker",
            "bipedal_walker_hardcore",
            "lunar_lander",
            "lunar_lander_continuous",
            "ant",
            "half_cheetah",
            "hopper",
            "humanoid",
            "swimmer",
            "walker_2d",
        ]

        self.reinforcement_envs = self.imitation_envs + [
            "humanoid_standup",
            "inverted_double_pendulum",
            "inverted_pendulum",
            "reacher",
        ]

        if config.paradigm == "reinforcement":
            assert config.env in self.reinforcement_envs, f"config.env "
            "needs to be one of {self.reinforcement_envs}."
        elif config.paradigm == "imitation":
            assert config.env in self.imitation_envs, f"config.env "
            "needs to be one of {self.imitation_envs}."

        self.env_name = get_env_name(config.env)

        env = gymnasium.make(self.env_name)

        super().__init__(env)

    def set_state(self, state):
        """
        .
        """
        self.env = state

    def get_state(self):
        """
        .
        """
        return copy.deepcopy(self.env)


def get_env_name(env):
    """
    .
    """
    if env == "acrobot":
        return "Acrobot-v1"
    elif env == "cart_pole":
        return "CartPole-v1"
    elif env == "mountain_car":
        return "MountainCar-v0"
    elif env == "mountain_car_continuous":
        return "MountainCarContinuous-v0"
    elif env == "pendulum":
        return "Pendulum-v1"
    elif env == "bipedal_walker":
        return "BipedalWalker-v3"
    elif env == "bipedal_walker_hardcore":
        return "BipedalWalkerHardcore-v3"
    elif env == "lunar_lander":
        return "LunarLander-v2"
    elif env == "lunar_lander_continuous":
        return "LunarLanderContinuous-v2"
    elif env == "ant":
        return "Ant-v4"
    elif env == "half_cheetah":
        return "HalfCheetah-v4"
    elif env == "hopper":
        return "Hopper-v4"
    elif env == "humanoid":
        return "Humanoid-v4"
    elif env == "humanoid_standup":
        return "HumanoidStandup-v2"
    elif env == "inverted_double_pendulum":
        return "InvertedDoublePendulum-v2"
    elif env == "inverted_pendulum":
        return "InvertedPendulum-v2"
    elif env == "reacher":
        return "Reacher-v2"
    elif env == "swimmer":
        return "Swimmer-v4"
    else:  # env == "walker_2d":
        return "Walker2d-v4"


def get_env_info(env):
    """
    .
    """
    env_name = get_env_name(env)

    env = gymnasium.make(env_name)

    d_input = env.observation_space.shape[0]

    if isinstance(env.action_space, gymnasium.spaces.Discrete):
        d_output = env.action_space.n
        discrete_output = True
        absolute_output_bound = None

    else:  # isinstance(env.action_space, gymnasium.spaces.Box):
        d_output = env.action_space.shape[0]
        discrete_output = False
        absolute_output_bound = env.action_space.high[0]

        if (
            not np.all(env.action_space.high == env.action_space.high[0])
            or not np.all(env.action_space.low == env.action_space.low[0])
            or -env.action_space.high[0] != env.action_space.low[0]
        ):
            raise Exception("Task absolute output bound issue.")

    env.close()

    return d_input, d_output, discrete_output, absolute_output_bound
