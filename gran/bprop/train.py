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

from hydra.utils import instantiate
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from gran.bprop.util.data.ar import ARDataModule
from gran.util.misc import seed_everything, config


def train():

    seed_everything(0)

    # W&B

    with open("../../../../../../../wandb_key.txt", "r") as f:
        key = f.read()

    wandb.login(key=key)

    wandb_logger = WandbLogger()

    # Train

    # dm = AEDataModule(data=obs_array, batch_size=10000)
    # model = MLPAE(x_size)

    # trainer = pl.Trainer(
    #     max_epochs=1000,
    #     accelerator="gpu",
    #     devices=1,
    #     logger=wandb_logger,
    #     enable_progress_bar=False,
    # )
    # trainer.fit(model, dm)

    import glob

    folders = glob.glob("data/0/*")

    obs_list, action_list, rew_list, done_list = [], [], [], []

    for folder in folders:
        obs_list.append(np.load(f"{folder}/obs.npy"))
        action_list.append(np.load(f"{folder}/action.npy"))
        rew_list.append(np.load(f"{folder}/rew.npy"))
        done_list.append(np.load(f"{folder}/done.npy"))

    obs_array = np.concatenate(obs_list)
    action_array = np.concatenate(action_list)
    rew_array = np.concatenate(rew_list)
    done_array = np.concatenate(done_list)

    data = [obs_array, action_array, rew_array, done_array]

    dm = ARDataModule(data=data, batch_size=4)

    model = instantiate(config.autoregressor)
    print(1)
    import os

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        # logger=wandb_logger,
        enable_progress_bar=True,
        default_root_dir="ar/",
    )
    print(2)
    trainer.fit(model, dm)

    wandb.finish()
