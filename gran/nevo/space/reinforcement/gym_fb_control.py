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

from gran.nevo.space.reinforcement.base import BaseReinforcementSpace
from gran.util.gym_fb_control import (
    score_tasks,
    get_env_state,
    get_task_name,
    reset_env_state,
    run_env_step,
    set_env_state,
)
from gran.util.misc import cfg


class Space(BaseReinforcementSpace):
    """
    Gymnasium Feature-Based Control Reinforcement Space.
    """

    def __init__(self):
        """
        .
        """
        self.valid_tasks = score_tasks
        self.get_env_state = get_env_state
        self.set_env_state = set_env_state
        self.reset_env_state = reset_env_state
        self.run_env_step = run_env_step

        super().__init__()

        self.env = gymnasium.make(get_task_name(cfg.ecosystem.task))
