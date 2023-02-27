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

from sb3_contrib import TQC
from stable_baselines3 import DQN, PPO, SAC, TD3

from gran.ne.IO.imitation.base import BaseImitationIO, BaseTarget
from gran.util.wrapper.gym_fb_control import get_task_name
from gran.util.misc import config


class Target(BaseTarget):
    def __init__(self, model):
        self.model = model
        self.is_done = False

    def reset(self, seed, step_num):
        pass

    def __call__(self, x):
        return self.model.predict(x)[0]


class IO(BaseImitationIO):
    def __init__(self):
        super().__init__()

        if config.task in ["acrobot", "mountain_car"]:
            self.model = DQN
            self.model_name = "dqn"

        elif config.task in ["cart_pole", "lunar_lander"]:
            self.model = PPO
            self.model_name = "ppo"

        elif config.task in [
            "mountain_car_continuous",
            "lunar_lander_continuous",
            "humanoid",
        ]:
            self.model = SAC
            self.model_name = "sac"

        elif config.task in ["ant", "swimmer", "walker_2d"]:
            self.model = TD3
            self.model_name = "td3"

        elif config.task in [
            "pendulum",
            "bipedal_walker",
            "bipedal_walker_hardcore",
            "half_cheetah",
            "hopper",
        ]:
            self.model = TQC
            self.model_name = "tqc"

    def load_target(self):
        dict = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

        dir_path = (
            self.args.data_path
            + "rl-trained-agents/"
            + self.model_name
            + "/"
            + get_task_name(config.task)
            + "_1/"
        )

        return Target(
            self.model.load(
                dir_path + get_task_name(config.task) + ".zip",
                custom_objects=dict,
                device="cpu",
            )
        )
