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

import argparse
from gym import wrappers
import numpy as np
import os
import retro
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

parser = argparse.ArgumentParser()
parser.add_argument("file")
parser.add_argument("--x", default="0")
parser.add_argument("--level", default="1-0")
parser.add_argument("--to_mp4", action="store_true")
parser.add_argument("--to_npy", action="store_true")
args = parser.parse_args()

if ".npy" in args.file:
    file = args.file.replace(".npy", "")

    args.to_npy = False

    npy_key_log = np.load(args.file)

if ".bk2" in args.file:
    file = args.file.replace(".bk2", "")

    key_log = retro.Movie(args.file)

if args.to_npy:
    npy_key_log = np.empty((0, 12), dtype=np.bool)

env = retro.make(
    "ShinobiIIIReturnOfTheNinjaMaster-Genesis", state="Level" + args.level
)

if args.to_mp4:
    env = wrappers.Monitor(env, file, force=True)

state = env.reset()

np.random.seed(0)

i = 0
more = True
while True:
    if ".bk2" in args.file:
        more = key_log.step()
        action = [key_log.get_key(i, 0) for i in range(env.num_buttons)]
    if not more:
        done = True
        break
    if ".npy" in args.file:
        action = npy_key_log[i]
    i += 1

    state, _, done, info = env.step(action)
    if done:
        break

    # if info['section'] == 1:
    # break

    if args.to_npy:
        npy_key_log = np.concatenate((npy_key_log, np.array(action)[None, :]))

    if not (args.to_npy or args.to_mp4):
        time.sleep(0.001)
        env.render()

if args.to_npy:
    np.save(file + ".npy", npy_key_log)
