# Copyright 2022 Maximilien Le Clei.
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

import argparse

from gran.rands.envs.base import EnvBase


class MultistepEnvBase(EnvBase):
    """
    Multistep Env Base class. Concrete subclasses need to be named *Env* and
    create the attribute `valid_tasks`: a function returning a list of valid
    tasks.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        io_path: str,
        nb_pops: int,
    ):

        if not hasattr(self, "get_valid_tasks"):
            raise NotImplementedError(
                "Multistep Environments require the attribute "
                "'get_valid_tasks': a function returning a list of all "
                "valid tasks."
            )

        if "task" not in args.extra_arguments:
            raise Exception(
                "Extra argument 'task' missing. It needs to "
                "be chosen from one of " + str(self.get_valid_tasks())
            )
        elif args.extra_arguments["task"] not in self.get_valid_tasks():
            raise Exception(
                "Extra argument 'task' invalid. It needs to "
                "be chosen from one of " + str(self.valid_tasks)
            )

        if "seeding" not in args.extra_arguments:
            args.extra_arguments["seeding"] = "reg"
        elif (
            not isinstance(args.extra_arguments["seeding"], int)
            and args.extra_arguments["seeding"] != "reg"
        ):
            raise Exception(
                "Extra argument 'seed' is of wrong type. "
                "It needs to be an integer >= 0 or string 'reg'."
            )

        if "steps" not in args.extra_arguments:
            args.extra_arguments["steps"] = 0
        elif not isinstance(args.extra_arguments["steps"], int):
            raise Exception(
                "Extra argument 'steps' is of wrong type. "
                "It needs to be an integer >= 0."
            )
        elif args.extra_arguments["steps"] < 0:  # 0 : infinite
            raise Exception("Extra argument 'steps' needs to be >= 0.")

        transfer_options = ["no", "fit", "env+fit", "mem+env+fit"]
        if "transfer" not in args.extra_arguments:
            args.extra_arguments["transfer"] = "no"
        elif args.extra_arguments["transfer"] not in transfer_options:
            raise Exception(
                "Extra argument 'transfer' invalid. It "
                "needs be chosen from one of " + str(transfer_options)
            )

        super().__init__(args, io_path, nb_pops)
