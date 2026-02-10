# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# Use local NVFlare source for MPS/hardware acceleration support
import sys
sys.path.insert(0, "/Users/angel/code/nvflare_exectorch_trial/NVFlare")

"""
Simulator version of et_job.py - runs locally without needing NVFlare server.
Use this for local testing without connectivity/authentication issues.
"""

import argparse
import os

from nvflare.edge.tools.et_fed_buff_recipe import (
    DeviceManagerConfig,
    ETFedBuffRecipe,
    EvaluatorConfig,
    ModelManagerConfig,
    SimulationConfig,
)
from nvflare.recipe.sim_env import SimEnv

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--num_clients", type=int, default=2, help="Number of simulated clients")
parser.add_argument("--total_num_of_devices", type=int, default=4)
parser.add_argument("--num_of_simulated_devices_on_each_leaf", type=int, default=1)
args = parser.parse_args()

total_num_of_devices = args.total_num_of_devices
num_of_simulated_devices_on_each_leaf = args.num_of_simulated_devices_on_each_leaf

if args.dataset == "cifar10":
    from processors.cifar10_et_task_processor import Cifar10ETTaskProcessor
    from processors.models.cifar10_model import TrainingNet

    dataset_root = "/tmp/nvflare/cifar10"
    job_name = "cifar10_et_sim"
    device_model = TrainingNet()
    batch_size = 4
    input_shape = (batch_size, 3, 32, 32)
    output_shape = (batch_size,)
    task_processor = Cifar10ETTaskProcessor(
        data_path=dataset_root,
        training_config={
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": 0,
        },
        subset_size=100,
    )
    evaluator_config = EvaluatorConfig(
        torchvision_dataset={"name": "CIFAR10", "path": dataset_root},
        eval_frequency=1,
    )
elif args.dataset == "xor":
    from processors.models.xor_model import TrainingNet
    from processors.xor_et_task_processor import XorETTaskProcessor

    job_name = "xor_et_sim"
    device_model = TrainingNet()
    batch_size = 1
    input_shape = (batch_size, 2)
    output_shape = (batch_size,)
    task_processor = XorETTaskProcessor(
        training_config={
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": 0,
        },
    )
    evaluator_config = None


recipe = ETFedBuffRecipe(
    job_name=job_name,
    device_model=device_model,
    input_shape=input_shape,
    output_shape=output_shape,
    model_manager_config=ModelManagerConfig(
        max_model_version=3,
        update_timeout=1000,
        num_updates_for_model=total_num_of_devices,
    ),
    device_manager_config=DeviceManagerConfig(
        device_selection_size=total_num_of_devices,
        min_hole_to_fill=total_num_of_devices,
    ),
    evaluator_config=evaluator_config,
    simulation_config=(
        SimulationConfig(
            task_processor=task_processor,
            num_devices=num_of_simulated_devices_on_each_leaf,
        )
        if num_of_simulated_devices_on_each_leaf > 0
        else None
    ),
    device_training_params={"epoch": 3, "lr": 0.0001, "batch_size": batch_size},
)

print("Running in SIMULATOR mode - no server connection needed!")
print(f"Number of clients: {args.num_clients}")
print()

# Use SimEnv instead of ProdEnv - runs locally without server
env = SimEnv(num_clients=args.num_clients)
result = recipe.execute(env)

print()
print("Simulation complete!")
print("Result workspace:", result)
print()
