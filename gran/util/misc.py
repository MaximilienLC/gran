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

from typing import Union

import numpy as np
import torch


def standardize(
    X: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)

    return (X - X_mean) / X_std


def normalize(
    X: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    return (X - X_min) / (X_max - X_min)


class RunningStandardization:
    """
    Standardize data using running mean and standard deviation.
    """

    def __init__(self) -> None:
        """
        Initialize running variables.
        """
        self.mean = self.var = self.std = self.n = 0

    def __call__(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Update running mean and standard deviation using the datapoint and
        return an updated datapoint.
        """
        self.n += 1

        new_mean = self.mean + (x - self.mean) / self.n
        new_var = self.var + (x - self.mean) * (x - new_mean)
        new_std = np.sqrt(new_var / self.n)

        self.mean, self.var, self.std = new_mean, new_var, new_std

        return (x - self.mean) / self.std
