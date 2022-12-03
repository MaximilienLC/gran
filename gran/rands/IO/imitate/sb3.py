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

from sb3_contrib import TQC
from stable_baselines3 import DQN, PPO, SAC, TD3

from gran.rands.IO.imitate.base import ImitateIOBase, TargetBase
from gran.utils.atari import get_atari_task_name, get_atari_imitation_tasks
from gran.utils.control import (
    get_control_task_name,
    get_control_imitation_tasks,
)


class Target(TargetBase):
    def __init__(self, model):
        self.model = model
        self.is_done = False

    def reset(self, seed, step_nb):
        pass

    def __call__(self, x):
        return self.model.predict(x)[0]


class IO(ImitateIOBase):
    def __init__(self, args):

        super().__init__(args)

        self.task = args.extra_arguments["task"]

        if self.task in get_atari_imitation_tasks():
            self.task_type = "atari"
        else:  # self.task in get_control_imitation_tasks():
            self.task_type = "control"

        if self.task in [
            "acrobot",
            "mountain_car",
            "beam_rider",
            "enduro",
            "seaquest",
        ]:

            self.model = DQN
            self.model_name = "dqn"

        elif self.task in [
            "cart_pole",
            "lunar_lander",
            "asteroids",
            "breakout",
            "pong",
            "qbert",
            "space_invaders",
        ]:

            self.model = PPO
            self.model_name = "ppo"

        elif self.task in [
            "mountain_car_continuous",
            "lunar_lander_continuous",
            "humanoid",
        ]:

            self.model = SAC
            self.model_name = "sac"

        elif self.task in ["ant", "swimmer", "walker_2d"]:

            self.model = TD3
            self.model_name = "td3"

        elif self.task in [
            "pendulum",
            "bipedal_walker",
            "bipedal_walker_hardcore",
            "half_cheetah",
            "hopper",
        ]:

            self.model = TQC
            self.model_name = "tqc"

    def load_target(self):

        if self.task_type == "atari":
            task_name = get_atari_task_name(self.task)
        else:  # self.task_type == "control":
            task_name = get_control_task_name(self.task)

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
            + task_name
            + "_1/"
        )

        return Target(
            self.model.load(
                dir_path + task_name + ".zip",
                custom_objects=dict,
                device="cpu",
            )
        )
