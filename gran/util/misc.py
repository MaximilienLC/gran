from omegaconf import OmegaConf

config = OmegaConf.load(".hydra/config.yaml")


def update_config_for_nevo(curr_iter):

    config = OmegaConf.load(".hydra/config.yaml")

    config["curr_iter"] = curr_iter

    config["ecosystem"]["prev_num_gens"] = (
        config.ecosystem.num_gens_per_iter * curr_iter
    )

    OmegaConf.save(config, ".hydra/config.yaml")


def get_task_domain(task):

    gym_fb_control = [
        "acrobot",
        "cart_pole",
        "mountain_car",
        "mountain_car_continuous",
        "pendulum",
        "bipedal_walker",
        "bipedal_walker_hardcore",
        "lunar_lander",
        "lunar_lander_continuous",
        "ant",
        "half_cheetah",
        "hopper",
        "humanoid",
        "swimmer",
        "walker_2d",
        "humanoid_standup",
        "inverted_double_pendulum",
        "inverted_pendulum",
        "reacher",
    ]

    if task in gym_fb_control:
        return "gym_fb_control"
