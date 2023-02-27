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

from abc import ABC, abstractmethod
from typing import Any

from gran.ne.IO.base import BaseIO


class BaseTarget(ABC):
    """
    Base Target class.
    Concrete subclasses need to be named *Target*.
    """

    @abstractmethod
    def reset(self, seed: int, step_num: int) -> None:
        """
        Reset the target's state

        Args:
            seed - Seed.
            step_num - Current step number.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, x: Any) -> Any:
        """
        Reset the target's state

        Args:
            x - Input value.
        Returns:
            Any - Output value.
        """
        raise NotImplementedError


class BaseImitationIO(BaseIO):
    @abstractmethod
    def load_target(self) -> BaseTarget:
        """
        Load a target to imitate.

        Returns:
            BaseTarget - The target.
        """
        raise NotImplementedError
