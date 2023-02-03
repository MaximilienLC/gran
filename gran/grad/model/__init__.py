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

from omegaconf import DictConfig
import pytorch_lightning as pl
import torch


class BaseModel(pl.LightningModule):
    def __init__(cfg: DictConfig) -> None:

        super().__init__()

        cfg = cfg

    def step(self, batch: torch.Tensor):
        raise NotImplementedError

    def training_step(self, batch: torch.Tensor, batch_idx: int):

        loss = self.step(batch)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):

        loss = self.step(batch)
        self.log("val/loss", loss)

    def test_step(self, batch: torch.Tensor, batch_idx: int):

        loss = self.step(batch)
        self.log("test/loss", loss)

    def configure_optimizers(self):

        return torch.optim.AdamW(self.parameters())