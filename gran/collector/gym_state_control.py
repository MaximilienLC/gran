import gymnasium

from gran.collector.base import BaseCollector
from gran.util.gym_state_control import (
    reset_emulator_state,
    run_emulator_step,
    get_task_name,
    get_task_info,
)
from gran.util.misc import cfg


class GymStateControlCollector(BaseCollector):
    def __init__(self, rank: int = -1) -> None:
        super().__init__(rank)

        self.emulator = gymnasium.make(cfg.emulator.name)
