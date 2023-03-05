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

import random

import numpy as np
from omegaconf import OmegaConf
import torch

config = OmegaConf.load(".hydra/config.yaml")


def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def update_config_for_fork(time, stage, curr_iter):

    config = OmegaConf.load(".hydra/config.yaml")

    if time == "pre-fork":

        config["curr_iter"] = curr_iter

        if stage == "collect":

            config["stage"] = "collect"

        elif stage == "train":

            config["agent"]["prev_num_gens"] = (
                config.agent.num_gens_per_iter * curr_iter
            )

    else:  # time == "post-fork":

        del config["curr_iter"]

        if stage == "collect":

            config["stage"] = "train"

        elif stage == "train":

            del config["agent"]["prev_num_gens"]

    OmegaConf.save(config, ".hydra/config.yaml")


def get_env_domain(env):

    gym_fb_control = [
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
        "humanoid_standup",
        "inverted_double_pendulum",
        "inverted_pendulum",
        "reacher",
    ]

    if env in gym_fb_control:
        return "gym_fb_control"
