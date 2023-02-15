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

import gymnasium

from gran.rand.env.score.base import BaseScoreEnv
from gran.util.gym_state_control import (
    imitation_tasks,
    hide_score,
    get_emulator_state,
    get_task_name,
    reset_emulator_state,
    run_emulator_step,
    set_emulator_state,
)
from gran.util.misc import cfg


class Env(BaseImitateEnv):
    def __init__(self):
        self.valid_tasks = imitation_tasks
        self.hide_score = hide_score
        self.get_emulator_state = get_emulator_state
        self.set_emulator_state = set_emulator_state
        self.reset_emulator_state = reset_emulator_state
        self.run_emulator_step = run_emulator_step

        self.emulators = [gymnasium.make(get_task_name(cfg.task))]

        super().__init__()

        self.target = self.io.load_target()
