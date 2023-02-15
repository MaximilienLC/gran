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

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, dimensions, recurrent=True):
        super().__init__()

        self.dimensions = dimensions
        self.recurrent = recurrent

        if recurrent:
            if len(dimensions) == 2:
                self.recurrent_layer_idx = 0
            else:
                self.recurrent_layer_idx = len(dimensions) - 3

        self.fc = nn.ModuleList()

        for i in range(len(dimensions) - 1):
            if recurrent and i == self.recurrent_layer_idx:
                self.rnn = nn.RNN(dimensions[i], dimensions[i + 1])
                self.h = torch.zeros(1, 1, dimensions[i + 1])
            else:
                self.fc.append(nn.Linear(dimensions[i], dimensions[i + 1]))

    def reset(self):
        if self.recurrent:
            self.h *= 0

    def forward(self, x):
        for i in range(len(self.dimensions) - 1):
            if self.recurrent == True:
                if i + 3 == len(self.dimensions) or len(self.dimensions) == 2:
                    x, self.h = self.rnn(x[None, :], self.h)

                elif i + 2 == len(self.dimensions):
                    x = torch.tanh(self.fc[-1](x[0, :]))

                else:
                    x = torch.tanh(self.fc[i](x))

            else:
                x = torch.tanh(self.fc[i](x))

        return x
