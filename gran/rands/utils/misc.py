def import(env_path):

    env = getattr(import_module(cfg.env_path), "Env")(cfg)