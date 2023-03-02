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

import sys
from typing import Union

import numpy as np
import torch


def normalize(
    X: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    return (X - X_min) / (X_max - X_min)


def standardize(
    X: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)

    return (X - X_mean) / X_std


class RunningNormalization:
    """
    Normalize data using running max and min.
    """

    def __init__(self) -> None:

        self.min, self.max = sys.float_info.min, sys.float_info.max

    def __call__(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:

        if isinstance(x, np.ndarray):
            minimum, maximum = np.minimum, np.maximum
        else:  # isinstance(x, torch.Tensor):
            minimum, maximum = np.minimum, torch.maximum

        self.min = minimum(x, self.min)
        self.max = maximum(x, self.max)

        # Default value is 0.5 & no division by zero
        numerator = (x - self.min) + 0.5 * ((self.max - self.min) == 0)
        denominator = self.max - self.min + ((self.max - self.min) == 0)

        return numerator / denominator


class RunningStandardization:
    """
    Standardize data using running mean and standard deviation.
    """

    def __init__(self) -> None:

        self.mean = self.var = self.std = self.n = 0

    def __call__(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:

        self.n += 1

        new_mean = self.mean + (x - self.mean) / self.n
        new_var = self.var + (x - self.mean) * (x - new_mean)
        new_std = np.sqrt(new_var / self.n)

        self.mean, self.var, self.std = new_mean, new_var, new_std

        return (x - self.mean) / (self.std + (self.std == 0))
