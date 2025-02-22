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
"""Implementation of VCritic."""

from typing import List

import torch
import torch.nn as nn

from omnisafe.models.base import Critic
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network


class VCritic(Critic):
    """Implementation of VCritic.

    A V-function approximator that uses a multi-layer perceptron (MLP) to map observations to V-values.
    This class is an inherit class of :class:`Critic`.
    You can design your own V-function approximator by inheriting this class or :class:`Critic`.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: List[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        num_critics: int = 1,
    ) -> None:
        """Initialize the critic network.

        Args:
            obs_dim (int): Observation dimension.
            act_dim (int): Action dimension.
            hidden_sizes (list): Hidden layer sizes.
            activation (Activation): Activation function.
            weight_initialization_mode (InitFunction): Weight initialization mode.
            shared (nn.Module): Shared network.
        """
        super().__init__(
            obs_space,
            act_space,
            hidden_sizes,
            activation,
            weight_initialization_mode,
            num_critics,
            use_obs_encoder=False,
        )
        self.net_lst: List[nn.Module] = []
        for idx in range(self._num_critics):
            net = build_mlp_network(
                sizes=[self._obs_dim, *self._hidden_sizes, 1],
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
            )
            self.net_lst.append(net)
            self.add_module(f'critic_{idx}', net)

    def forward(
        self,
        obs: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Forward function.

        Specifically, V function approximator maps observations to V-values.

        Args:
            obs (torch.Tensor): Observations.
        """
        res = []
        for critic in self.net_lst:
            res.append(torch.squeeze(critic(obs), -1))
        return res
