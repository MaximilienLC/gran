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

from glob import glob

import numpy as np

from gran.rands.IO.imitate import ImitateIOBase, TargetBase


class Target(TargetBase):
    def __init__(self, data):

        self.data = data
        self.data_index = 0
        self.frame_index = 0
        self.is_done = False

    def reset(self, y, z):

        self.data_index, self.frame_index = y % len(self.data), z
        self.is_done = False

    def __call__(self, x):

        action = self.data[self.data_index][self.frame_index]

        self.frame_index += 1
        if len(self.data[self.data_index]) == self.frame_index:
            self.is_done = True

        return action


class IO(ImitateIOBase):
    def __init__(self, args):

        super().__init__(args)

        path = (
            "data/behaviour/subject/"
            + args.extra_arguments["task"]
            + "/sub_0"
            + str(args.extra_arguments["subject"])
        )

        if args.extra_arguments["task"] in ["shinobi", "mario"]:
            path += "/" + args.extra_arguments["level"]

        path += "/*.npy"

        self.behaviour = []
        for f in glob(path):
            self.behaviour.append(np.load(f))

        if len(self.behaviour) == 0:
            raise Exception("No *.npy data in " + path[:-5])

    def load_target(self):
        return Target(self.behaviour)
