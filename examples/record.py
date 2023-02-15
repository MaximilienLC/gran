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
import glob
import numpy as np
import os
import pickle
import random
import sys
import torch
import warnings

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

warnings.filterwarnings("ignore", category=UserWarning)
sys.setrecursionlimit(2**31 - 1)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--state_path",
    "-s",
    type=str,
    required=True,
    help="Path to the saved state <=> "
    "data/states/<env_path>/<extra_arguments>/"
    "<bot_path>/<pop_size>/<gen>/",
)

parser.add_argument(
    "--num_tests",
    "-t",
    type=int,
    default=1,
    help="Number of tests to record the agent on.",
)

parser.add_argument(
    "--num_obs",
    "-o",
    type=int,
    default=2**31 - 1,
    help="Number of observations to record the agent on.",
)

parser.add_argument(
    "--record_rewards",
    "-r",
    action="store_true",
    help="Whether to record the agent's reward history.",
)

args = parser.parse_args()

MAX_INT = 2**31 - 1


# Backward Compatibility for Control Task experiments
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "bots.static.rnn.control":
            renamed_module = "bots.netted.static.rnn.control"
        elif module == "bots.dynamic.rnn.control":
            renamed_module = "bots.netted.dynamic.rnn.control"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


"""
Process arguments
"""

if args.state_path[-1] == "/":
    args.state_path = args.state_path[:-1]

split_path = args.state_path.split("/")

env_path = split_path[-5]
extra_arguments = split_path[-4]
bot_path = split_path[-3]
pop_size = int(split_path[-2])
gen = int(split_path[-1])

split_extra_arguments = extra_arguments.split("~")


if "score" in env_path:
    steps = split_extra_arguments[0].split(".")[1]
    task = split_extra_arguments[1].split(".")[1]
    transfer = split_extra_arguments[2].split(".")[1]
    trials = split_extra_arguments[3].split(".")[1]

elif "imitate.retro" in env_path:
    level = split_extra_arguments[0].split(".")[1]
    merge = split_extra_arguments[1].split(".")[1]
    steps = split_extra_arguments[2].split(".")[1]
    subject = split_extra_arguments[3].split(".")[1]
    task = split_extra_arguments[4].split(".")[1]
    transfer = split_extra_arguments[5].split(".")[1]

else:  # 'imitate' in env_path:
    merge = split_extra_arguments[0].split(".")[1]
    steps = split_extra_arguments[1].split(".")[1]
    task = split_extra_arguments[2].split(".")[1]
    transfer = split_extra_arguments[3].split(".")[1]

"""
Initialize environment
"""

if "control" in env_path:
    import gym
    from gym import wrappers
    from gran.utils.functions.control import get_task_name

    emulator = gym.make(get_task_name(task))

    hide_score = lambda x: x

    if args.record_rewards == False:
        emulator = wrappers.RecordVideo(emulator, args.state_path)

elif "atari" in env_path:
    import gym
    from gym import wrappers
    import ale_py

    from utils.functions.atari import get_task_name, hide_score, wrap

    emulator = wrap(
        gym.make(get_task_name(task), frameskip=1, repeat_action_probability=0)
    )

    if args.record_rewards == False:
        emulator = wrappers.RecordVideo(emulator, args.state_path)

elif "retro" in env_path:
    from gym import wrappers
    import retro
    from utils.functions.retro import get_task_name, hide_score
    from utils.functions.retro import get_state_name

    emulator = retro.make(
        game=get_task_name(task), state=get_state_name(level)
    )

    if args.record_rewards == False:
        emulator = wrappers.RecordVideo(emulator, args.state_path)

else:  # 'gravity' in env_path:
    data = np.load("data/behaviour/gravity/11k.npy")[-1000:]

    output = np.empty((7, 16, 16))

"""
Import bots
"""

if "gym" in env_path:
    if "dynamic" in bot_path:
        from bots.netted.dynamic.rnn.control import Bot
    else:  # 'static' in bot_path:
        from bots.netted.static.rnn.control import Bot

elif "atari" in env_path:
    if "dynamic" in bot_path:
        from bots.netted.dynamic.conv_rnn.atari import Bot
    else:  # 'static' in bot_path:
        from bots.netted.static.conv_rnn.atari import Bot

elif "retro" in env_path:
    if "dynamic" in bot_path:
        from bots.netted.dynamic.conv_rnn.retro import Bot
    else:  # 'static' in bot_path:
        from bots.netted.static.conv_rnn.retro import Bot

else:  # 'gravity' in env_path:
    if "dynamic.conv_rnn" in bot_path:
        from bots.netted.dynamic.conv_rnn.gravity import Bot
    else:  # 'static.conv_rnn' in bot_path:
        from bots.netted.static.conv_rnn.gravity import Bot

"""
Distribute workload
"""

pkl_files = [
    os.path.basename(x) for x in glob.glob(args.state_path + "/*.pkl")
]

state_files = []

for pkl_file in pkl_files:
    if pkl_file[:-4].isdigit():
        state_files.append(pkl_file)

if len(state_files) == 0:
    raise Exception("Directory '" + args.state_path + "/' empty.")

try:
    with open(args.state_path + "/0.pkl", "rb") as f:
        state = RenameUnpickler(f).load()

except Exception:
    print("File '" + args.state_path + "/0.pkl' doesn't exist / is corrupted.")

if len(args.state_path) == 3:
    full_seed_list, _, _ = state

else:  # len(state) == 4:
    _, _, latest_fitnesses_and_bot_sizes, bots = state

    for i in range(1, len(state_files)):
        try:
            with open(args.state_path + "/" + str(i) + ".pkl", "rb") as f:
                bots += RenameUnpickler(f).load()[0]

        except Exception:
            print(
                "File '"
                + args.state_path
                + "/"
                + str(i)
                + ".pkl' doesn't exist / is corrupted."
            )

    fitnesses_sorting_indices = latest_fitnesses_and_bot_sizes[
        :, :, 0
    ].argsort(axis=0)
    fitnesses_rankings = fitnesses_sorting_indices.argsort(axis=0)
    selected = np.greater_equal(fitnesses_rankings, pop_size // 2)
    selected_indices = np.where(selected[:, 0] == True)[0]

scores = np.load(args.state_path + "/scores.npy")

i = scores.mean(axis=1).argmax()

if len(state) == 3:
    bot = Bot(0)
    bot.build(full_seed_list[i][0])

else:  # len(state) == 4:
    bot = bots[selected_indices[i]][0]

bot.setup_to_run()

rewards = np.empty((args.num_tests, 0)).tolist()

if "dynamic.rnn" in bot_path:
    with open(args.state_path + "/net.pkl", "wb") as f:
        pickle.dump(str(bot.nets[0].nodes["layered"]), f)

if hasattr(bot, "n"):
    print(bot.n)

for i in range(args.num_tests):
    print("Test #" + str(i))

    bot.reset()

    np.random.seed(MAX_INT - i)
    torch.manual_seed(MAX_INT - i)
    random.seed(MAX_INT - i)

    if "gravity" not in env_path:
        emulator.seed(MAX_INT - i)
        obs = emulator.reset()
        done = False

    else:  # 'gravity' in env_path:
        num_obs_fed_to_generator = 3 if task == "predict" else 1  # 'generate'
        data_point = np.random.choice(data)
        output[i, 3:] = data_point

    score = 0

    for j in range(args.num_obs):
        if "gravity" not in env_path:
            if "imitate" in env_path:
                obs = hide_score(obs)

            obs, rew, done, _ = emulator.step(bot(obs))
            score += rew
            rewards[i].append(rew)

            if done:
                print(score)
                break

        else:  # 'gravity' in env_path:
            if j < num_obs_fed_to_generator:
                obs = data_point[j]

            obs = bot(obs)
            score += np.sum((obs - data_point[j + 1]) ** 2)

            output[i, j] = obs

            if j == 2:
                print(score)
                break

if "gravity" in env_path:
    np.save(args.state_path + "/output.npy", output)

if args.record_rewards:
    with open(args.state_path + "/rewards.pkl", "wb") as f:
        pickle.dump(rewards, f)
