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

import gym

from gran.rands.envs.multistep.score.base import ScoreMultistepEnvBase
from gran.utils.control import (
    get_control_emulator_state,
    get_control_score_tasks,
    get_control_task_name,
    set_control_emulator_state,
)


class Env(ScoreMultistepEnvBase):
    def __init__(self, args):

        (
            self.get_valid_tasks,
            self.get_emulator_state,
            self.set_emulator_state,
        ) = (
            get_control_score_tasks,
            get_control_emulator_state,
            set_control_emulator_state,
        )

        super().__init__(args)

        for key in args.extra_arguments:
            if key not in ["task", "seeding", "steps", "transfer", "trials"]:
                raise Exception("Extra argument", key, "not supported.")

        self.emulator = gym.make(
            get_control_task_name(args.extra_arguments["task"])
        )
