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

import os

import numpy as np
from omegaconf import OmegaConf

cfg = OmegaConf.load(os.getcwd() + "/.hydra/config.yaml")


def update_running_mean_std(x, old_mean, old_var, n):

    new_mean = old_mean + (x - old_mean) / n
    new_var = old_var + (x - old_mean) * (x - new_mean)
    new_std = np.sqrt(new_var / n)

    return new_mean, new_var, new_std
