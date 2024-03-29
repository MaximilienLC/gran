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

from gran.nevo.space.reinforcement.base import BaseReinforcementSpace
from gran.util.wrapper.gym_fb_control import GymFeatureBasedControlWrapper


class Space(BaseReinforcementSpace):
    """
    Gymnasium Feature-Based Control Reinforcement Space.
    """

    def __init__(self):

        self.env = GymFeatureBasedControlWrapper()

        super().__init__()
