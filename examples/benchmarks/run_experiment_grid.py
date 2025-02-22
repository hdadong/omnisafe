# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
# ==============================================================================
"""Example of training a policy from exp-x config with OmniSafe."""

import os
import sys

import torch

import omnisafe
from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.typing import NamedTuple, Tuple


def train(
    exp_id: str, algo: str, env_id: str, custom_cfgs: NamedTuple
) -> Tuple[float, float, float]:
    """Train a policy from exp-x config with OmniSafe.

    Args:
        exp_id (str): Experiment ID.
        algo (str): Algorithm to train.
        env_id (str): The name of test environment.
        custom_cfgs (NamedTuple): Custom configurations.
        num_threads (int, optional): Number of threads. Defaults to 6.
    """
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f'exp-x: {exp_id} is training...')
    USE_REDIRECTION = True
    if USE_REDIRECTION:
        if not os.path.exists(custom_cfgs['logger_cfgs']['log_dir']):
            os.makedirs(custom_cfgs['logger_cfgs']['log_dir'])
        sys.stdout = open(
            f'{custom_cfgs["logger_cfgs"]["log_dir"]}terminal.log', 'w', encoding='utf-8'
        )
        sys.stderr = open(
            f'{custom_cfgs["logger_cfgs"]["log_dir"]}error.log', 'w', encoding='utf-8'
        )
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    reward, cost, ep_len = agent.learn()
    return reward, cost, ep_len


if __name__ == '__main__':
    eg = ExperimentGrid(exp_name='Safety_Gymnasium_Goal')
    base_policy = ['PolicyGradient', 'NaturalPG', 'TRPO', 'PPO']
    naive_lagrange_policy = ['PPOLag', 'TRPOLag', 'RCPO', 'OnCRPO', 'PDO']
    first_order_policy = ['CUP', 'FOCOPS']
    second_order_policy = ['CPO', 'PCPO']
    eg.add('algo', base_policy + naive_lagrange_policy + first_order_policy + second_order_policy)
    eg.add('env_id', ['SafetyAntVelocity-v4'])
    eg.add('logger_cfgs:use_wandb', [False])
    eg.add('num_envs', [16])
    eg.add('num_threads', [1])
    # eg.add('logger_cfgs:wandb_project', ['omnisafe_jiaming'])
    # eg.add('train_cfgs:total_steps', 2000)
    # eg.add('algo_cfgs:update_cycle', 1000)
    # eg.add('train_cfgs:vector_env_nums', 1)
    eg.add('seed', [0])
    eg.run(train, num_pool=13)
