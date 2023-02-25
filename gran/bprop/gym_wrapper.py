import gymnasium

from gran.util.misc import cfg

from hydra import instantiate


class Wrapper(gymnasium.Wrapper):
    """
    .
    """

    def __init__(self, env: gymnasium.Env):
        """
        .
        """
        super().__init__(env)

        if cfg.compressor.is_used:
            self.compressor = instantiate(cfg.compressor).load_from_checkpoint(
                cfg.predictor.checkpoint
            )

        if cfg.predictor.is_used:
            self.predictor = instantiate(cfg.predictor).load_from_checkpoint(
                cfg.predictor.checkpoint
            )

    def reset(self, **kwargs):

        obs, info = self.env.reset(**kwargs)

        if self.compressor != None:

            self.compressor.eval()

            with torch.no_grad():
                obs = self.compressor(obs)

        if self.predictor != None:

            self.predictor.eval()
            self.predictor.reset()

            self.obs = obs

        return obs, info

    def step(self, action):
        """
        .
        """
        if self.predictor != None:

            obs_action = torch.cat((self.obs, action)).view(1, 1, -1)

            with torch.no_grad():
                self.obs, rew, done = self.predictor(obs_action)

            self.obs = self.obs.cpu().squeeze().numpy()
            self.rew = self.rew.cpu().squeeze().numpy()
            self.done = bool(self.done.cpu().squeeze().numpy())

            return self.obs, rew, done, done, None

        else:

            obs, rew, term, trunc, info = self.env.step(action)

            if self.compressor != None:

                with torch.no_grad():
                    obs = self.compressor(obs)

            return obs, rew, term, trunc, info

    def render(self):
        pass
