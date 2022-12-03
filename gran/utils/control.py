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

import copy

import gym


def set_control_emulator_state(emulator, state):
    emulator = state


def get_control_emulator_state(emulator):
    return copy.deepcopy(emulator)


def get_control_score_tasks():

    return [
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
        "humanoid_standup",
        "inverted_double_pendulum",
        "inverted_pendulum",
        "reacher",
        "swimmer",
        "walker_2d",
    ]


def get_control_imitation_tasks():

    return [
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


def get_control_task_name(task):

    if task == "acrobot":
        return "Acrobot-v1"
    elif task == "cart_pole":
        return "CartPole-v1"
    elif task == "mountain_car":
        return "MountainCar-v0"
    elif task == "mountain_car_continuous":
        return "MountainCarContinuous-v0"
    elif task == "pendulum":
        return "Pendulum-v1"
    elif task == "bipedal_walker":
        return "BipedalWalker-v3"
    elif task == "bipedal_walker_hardcore":
        return "BipedalWalkerHardcore-v3"
    elif task == "lunar_lander":
        return "LunarLander-v2"
    elif task == "lunar_lander_continuous":
        return "LunarLanderContinuous-v2"
    elif task == "ant":
        return "Ant-v4"
    elif task == "half_cheetah":
        return "HalfCheetah-v4"
    elif task == "hopper":
        return "Hopper-v4"
    elif task == "humanoid":
        return "Humanoid-v4"
    elif task == "humanoid_standup":
        return "HumanoidStandup-v2"
    elif task == "inverted_double_pendulum":
        return "InvertedDoublePendulum-v2"
    elif task == "inverted_pendulum":
        return "InvertedPendulum-v2"
    elif task == "reacher":
        return "Reacher-v2"
    elif task == "swimmer":
        return "Swimmer-v4"
    elif task == "walker_2d":
        return "Walker2d-v4"
    else:
        raise Exception("Task: " + task + " not supported.")


def get_control_task_info(task):

    task_name = get_control_task_name(task)

    emulator = gym.make(task_name)

    d_input = emulator.observation_space.shape[0]

    if isinstance(emulator.action_space, gym.spaces.Discrete):

        d_output = emulator.action_space.n
        discrete_output = True
        output_bound = None

    else:  # isinstance(emulator.action_space, gym.spaces.Box):

        d_output = emulator.action_space.shape[0]
        discrete_output = False
        output_bound = emulator.action_space.high.item()

    emulator.close()

    return d_input, d_output, discrete_output, output_bound
