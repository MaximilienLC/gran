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

import numpy as np

from gran.rands.IO.imitate.base import ImitateIOBase, TargetBase


class Target(TargetBase):
    def __init__(self, data):

        self.data = data
        self.data_index = 0
        self.frame_index = 0
        self.is_done = False

    def reset(self, seed, step_nb):

        self.data_index, self.frame_index = seed, step_nb

    def __call__(self, x):

        frame = self.data[self.data_index][self.frame_index]
        self.frame_index += 1

        return frame


class IO(ImitateIOBase):
    def load_target(self):
        data = np.load("data/behaviour/gravity/11k.npy")[:10000]
        return Target(data)
