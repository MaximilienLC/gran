import gymnasium


class BaseWrapper(gymnasium.Wrapper):
    """
    .
    """

    def __init__(self, config, env):
        """
        .
        """
        super().__init__(env)

        if config.autoencoder.in_use:
            self.autoencoder = instantiate(
                config.autoencoder
            ).load_from_checkpoint(config.autoencoder.checkpoint)

        if config.autoregressor.in_use:
            self.autoregressor = instantiate(
                config.autoregressor
            ).load_from_checkpoint(config.autoregressor.checkpoint)

    def reset(self, **kwargs):

        obs, info = self.env.reset(**kwargs)

        if self.autoencoder != None:

            self.autoencoder.eval()

            with torch.no_grad():
                obs = self.autoencoder(obs)

        if self.autoregressor != None:

            self.autoregressor.eval()
            self.autoregressor.reset()

            self.obs = obs

        return obs, info

    def step(self, action):
        """
        .
        """
        if self.autoregressor != None:

            obs_action = torch.cat((self.obs, action)).view(1, 1, -1)

            with torch.no_grad():
                self.obs, rew, done = self.autoregressor(obs_action)

            self.obs = self.obs.cpu().squeeze().numpy()
            self.rew = self.rew.cpu().squeeze().numpy()
            self.done = bool(self.done.cpu().squeeze().numpy())

            return self.obs, rew, done, done, None

        else:

            obs, rew, term, trunc, info = self.env.step(action)

            if self.autoencoder != None:

                with torch.no_grad():
                    obs = self.autoencoder(obs)

            return obs, rew, term, trunc, info

    def render(self):
        pass
