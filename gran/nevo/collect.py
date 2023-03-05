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

import os
import sys
import warnings

from mpi4py import MPI
import numpy as np

from gran.util.misc import config, seed_everything
from gran.util.wrapper.gym_fb_control import GymFeatureBasedControlWrapper

np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")
sys.setrecursionlimit(2**31 - 1)


def collect():
    """
    Initialization of various MPI & EA-specific variables.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    """
    Initialize environment
    """
    env = GymFeatureBasedControlWrapper()

    episodes_per_proc = config.collection_episodes // size

    assigned_episodes = np.arange(
        rank * episodes_per_proc, (rank + 1) * episodes_per_proc
    ).tolist()

    obs_list, rew_list, done_list, action_list = [], [], [], []

    for episode in assigned_episodes:

        print(f"{rank}: {episode}")

        seed_everything(episode)

        obs = env.reset(episode)
        done, rew = False, np.nan

        while True:

            obs_list.append(obs)
            rew_list.append(rew)
            done_list.append(done)

            if done:
                action_list.append([np.nan, np.nan])
                break

            action = env.action_space.sample()
            action_list.append([1, 0] if action == 0 else [0, 1])

            obs, rew, done = env.step(action)

    obs_array = np.array(obs_list, dtype=np.float32)
    rew_array = np.array(rew_list, dtype=np.float32)
    done_array = np.array(done_list, dtype=np.float32)
    action_array = np.array(action_list, dtype=np.float32)

    dir_path = f"data/{config.curr_iter}/{rank}"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    np.save(f"{dir_path}/obs.npy", obs_array)
    np.save(f"{dir_path}/rew.npy", rew_array)
    np.save(f"{dir_path}/done.npy", done_array)
    np.save(f"{dir_path}/action.npy", action_array)
