import gymnasium
import numpy as np

from gran.util.wrapper.gym_fb_control import (
    reset_env_state,
    run_env_step,
    get_task_info,
    get_task_name,
)
from gran.util.misc import standardize

task = "cart_pole"
env = gymnasium.make(get_task_name(task))
x_size, _, _, _ = get_task_info(task)

states = []

for i in range(10000):
    if i == 0 or done:
        obs, done = reset_env_state(env, 0), False

    states.append(obs)

    obs, rew, done = run_env_step(env, env.action_space.sample())

states = standardize(np.array(states))

env.close()

import torch
import pytorch_lightning as pl

import wandb
from pytorch_lightning.loggers import WandbLogger

with open("../../wandb_key.txt", "r") as f:
    key = f.read()

wandb.login(key=key)


class CartPoleDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray):
        self.X = torch.tensor(data, dtype=torch.float)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index: int):
        return self.X[index]


class CartPoleDataModule(pl.LightningDataModule):
    def __init__(self, data: str, batch_size: int):
        super().__init__()
        self.data = data
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.cartpole_train = CartPoleDataset(self.data)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cartpole_train, batch_size=self.batch_size
        )


from gran.bp.model.ae.var.mlp import MLPVAE

pl.seed_everything(0)
wandb_logger = WandbLogger()

model = MLPVAE(
    x_size=x_size, hidden_size=x_size * 100, latent_size=x_size * 100
)
dm = CartPoleDataModule(np.array(states), 10000)

trainer = pl.Trainer(
    max_epochs=10000, accelerator="gpu", devices=-1, logger=wandb_logger
)
trainer.fit(model, dm)
