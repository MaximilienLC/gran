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

import gymnasium

from gran.rands.envs.cpu.imitate.base import CPUImitateEnvBase
from gran.utils.control import (
    get_control_emulator_state,
    get_control_imitate_tasks,
    get_control_task_name,
    hide_score,
    reset_emulator_state,
    run_emulator_step,
    set_control_emulator_state,
)


class Env(CPUImitateEnvBase):
    def __init__(self, cfg):

        self.get_valid_tasks = get_control_imitate_tasks
        self.get_emulator_state = get_control_emulator_state
        self.set_emulator_state = set_control_emulator_state
        self.reset_emulator_state = reset_emulator_state
        self.run_emulator_step = run_emulator_step
        self.hide_score = hide_score

        super().__init__(cfg)

        self.emulators = [
            gymnasium.make(get_control_task_name(cfg.rands.env.task))
        ]

        self.target = self.io.load_target()
