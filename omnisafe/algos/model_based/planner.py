# Copyright 2022 OmniSafe Team. All Rights Reserved.
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
"""Safe controllers which do a black box optimization incorporating the constraint costs."""

import copy

import numpy as np
import scipy.stats as stats
import torch

class Planner:  # pylint: disable=too-many-instance-attributes
    """Model Predictive Control"""

    # pylint: disable-next=too-many-locals, too-many-arguments
    def __init__(
        self,
        device,
        env,
        models,
        actor_critic,
        horizon,
        popsize,
        particles,
        max_iters,
        alpha,
        mixture_coefficient,
        kappa,
        safety_threshold,
        minimal_elites,
        obs_clip,
    ):
        self.device = device
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.env = copy.deepcopy(env)
        self.models = models
        self.actor_critic = actor_critic
        self.termination_function = default_termination_function
        self.horizon = horizon
        self.sol_dim = self.env.action_space.shape[0] * horizon
        self.action_max = np.repeat(self.env.action_space.high, self.horizon, axis=0)
        self.action_min = np.repeat(self.env.action_space.low, self.horizon, axis=0)
        self.mean = np.zeros((self.sol_dim,))
        # Shape: [ H * action_dim, 1 ]
        self.num_gaussian_traj = popsize
        self.mixture_coefficient = mixture_coefficient
        self.num_actor_traj = int(self.mixture_coefficient * self.num_gaussian_traj)

        self.particles = particles
        self.max_iters = max_iters
        self.alpha_plan = alpha
        self.kappa = kappa
        self.safety_threshold = safety_threshold
        self.minimal_elites = minimal_elites
        self.state_start_dim = 2 if self.env.env_type == 'mujoco-speed' else 1
        self.obs_clip = obs_clip
        
    def planner_reset(self):
        """Reset planner when the episode end."""
        self.mean = np.zeros((self.sol_dim,))
        
    def generate_actor_action(self, curr_state):
        """Generate H steps deterministic and stochastic actor action trajectory using dynamics model."""
        # Set the reward of initial state to zero.
        actor_state = np.array(
            [np.concatenate(([0]*self.state_start_dim, curr_state.copy()), axis=0)] * (self.num_actor_traj)
        )
        # Shape: [actor_traj, reward_dim (+ cost_dim) + state_dim]

        # Add trajectories using actions suggested by actors
        actor_action_traj = np.zeros((self.num_actor_traj, self.sol_dim))
        # Shape: [actor_traj, H * action_dim]

        actor_state = torch.FloatTensor(actor_state).to(self.device)
        # Shape: [actor_traj, reward_dim (+ cost_dim) + state_dim]

        actor_state_m = actor_state[0, :].reshape(1, -1)
        # Shape: [1, reward_dim (+ cost_dim) + state_dim]

        actor_state_m2 = actor_state[1, :].reshape(1, -1)
        # Shape: [1, reward_dim (+ cost_dim) + state_dim]

        for current_horizon in range(self.horizon):
            # Use deterministic policy to plan a action trajectory
            actor_actions_m = self.actor_critic.act_batch(
                actor_state_m.reshape(1, -1)[:, self.state_start_dim:], deterministic=True
            )
            # Shape: [1, action_dim]

            # Use dynamics model to plan
            actor_state_m, _ = self.models.get_forward_prediction_random_t(
                actor_state_m[:, self.state_start_dim:], actor_actions_m, repeat_network=True,
            )
            # Shape: [1, reward_dim + state_dim]
            
            # protection for producing nan
            actor_state_m = torch.clamp(actor_state_m, -self.obs_clip, self.obs_clip)
            actor_state_m = torch.nan_to_num(actor_state_m)

            # Store a planning action to action buffer
            actor_action_traj[
                0, current_horizon * self.action_dim : (current_horizon + 1) * self.action_dim
            ] = (actor_actions_m.detach().cpu().numpy())

            # Using Stochastic policy to plan a action trajectory
            actor_actions = self.actor_critic.act_batch(actor_state_m2[:, self.state_start_dim:])
            # Shape: [1, action_dim]

            # Use dynamics model to plan
            actor_state_m2, _ = self.models.get_forward_prediction_random_t(
                actor_state_m2[:, self.state_start_dim:], actor_actions, repeat_network=True,
            )
            # Shape: [1, reward_dim + state_dim]

            # protection for producing nan
            actor_state_m2 = torch.clamp(actor_state_m2, -self.obs_clip, self.obs_clip)
            actor_state_m2 = torch.nan_to_num(actor_state_m2)

            # Copy the planning action of stochastic actor (actor_traj-1) times, and store to action buffer
            actor_action_traj[
                1:, current_horizon * self.action_dim : (current_horizon + 1) * self.action_dim
            ] = (actor_actions.detach().cpu().numpy())
        return actor_action_traj
    
    def compute_terminal_reward(self,action_traj,state_traj):
        """ Compute the terminal reward behind H horizon"""
        # This is the final action for evaluating terminated reward and cost
        final_action = (
            torch.from_numpy(
                action_traj[
                    :,
                    (self.horizon - 1)
                    * self.action_dim : (self.horizon)
                    * self.action_dim,
                ].reshape((self.num_gaussian_traj + self.num_actor_traj) * self.particles, -1)
            )
            .float()
            .to(self.device)
        )
        # Shape: [ (num_gau_traj + num_actor_traj) * particles,  action_dim ) ,
        # action_traj is [ (num_gau_traj + num_actor_traj) * particles, H * action_dim]

        final_action = final_action.repeat(self.models.model.network_size, 1)
        # Shape: [ (num_gau_traj + num_actor_traj) * particles, network_size , action_dim )

        # This is the final state for evaluating terminated reward and cost
        final_state = (
            torch.from_numpy(
                state_traj[self.horizon, :, :, self.state_start_dim:].reshape(
                    (self.num_gaussian_traj + self.num_actor_traj) * self.particles * state_traj.shape[1],
                    -1,
                )
            )
            .float()
            .to(self.device)
        )
        # [ (num_gau_traj + num_actor_traj) * particles, state_dim ]
    
        terminal_reward = self.actor_critic.q1(final_state, final_action).cpu().detach().numpy()
        terminal_reward = terminal_reward.reshape(state_traj.shape[1], -1)
        # [ (num_gau_traj + num_actor_traj) * particles, 1]
        
        return terminal_reward

    def compute_cost_from_state(self, state_traj):
        """compute cost from state that dyamics model predict"""
        states_flatten = state_traj[:, :, :, self.state_start_dim:].reshape(-1, self.obs_dim)
        # [ horizon+1 * network_size * (num_gau_traj + num_actor_traj) * particles, state_dim]
        
        all_safety_costs = np.zeros((states_flatten.shape[0],))
        # [ horizon+1 * network_size * (num_gau_traj + num_actor_traj) * particles, 1]

        all_safety_costs = self.env.get_observation_cost(states_flatten)
        # [ horizon+1 * network_size * (num_gau_traj + num_actor_traj) * particles, 1]

        all_safety_costs = all_safety_costs.reshape(
            state_traj.shape[0], state_traj.shape[1], state_traj.shape[2], 1
        )
        # [ horizon+1, network_size, (num_gau_traj + num_actor_traj) * particles, 1]
        return all_safety_costs
    
    
    def get_action(
        self, curr_state
    ):  # pylint: disable=too-many-statements, too-many-locals
        """Select action when interact with environment."""
        # sample action from actor
        if self.num_actor_traj != 0.0 :
            actor_action_traj = self.generate_actor_action(curr_state)
            # Shape: [actor_traj, H * action_dim]

        curr_state = np.array(
            [np.concatenate(([0]*self.state_start_dim, curr_state.copy()), axis=0)]
            * ((self.num_gaussian_traj + self.num_actor_traj) * self.particles)
        )
        # Shape: [(num_gau_traj + num_actor_traj) * particles, reward_dim (+ cost_dim) + state_dim]

        curr_state = np.expand_dims(curr_state, axis=0)
        # Shape: [1, (num_gau_traj + num_actor_traj) * particles, reward_dim (+ cost_dim) +state_dim]

        curr_state = np.repeat(curr_state, self.models.model.network_size, 0)
        # Shape: [network_size, (num_gau_traj + num_actor_traj) * particles, reward_dim (+ cost_dim) + state_dim]
        
        # initial mean and var of the sampling normal dist
        # shift the current array to the left, clear the used action
        self.mean[: -self.action_dim] = self.mean[self.action_dim :]
        # Shape: [ H * action_dim, 1 ]

        # fill the last position with the last second  action
        self.mean[-self.action_dim :] = self.mean[-2 * self.action_dim : -self.action_dim]
        mean = self.mean
        # Shape: [ H * action_dim, 1 ]

        var = np.tile(
            np.square(self.env.action_space.high[0] - self.env.action_space.low[0]) / 16,
            [self.sol_dim],
        )
        # Shape: [ H * action_dim, 1 ]

        # Create gaussian distribution.
        # mean is the zero vector, var is Unit Matrix
        gaussian = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean))

        current_iter = 0
        while current_iter < self.max_iters:
            lb_dist, ub_dist = mean - self.action_min, self.action_max - mean

            constrained_var = np.minimum(
                np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var
            )

            # Generate random normal gaussian variable and multiply by the var, then add to the mean
            action_traj = (
                gaussian.rvs(size=(self.num_gaussian_traj, self.sol_dim)) * np.sqrt(constrained_var)
                + mean
            ).astype(np.float32)
            # Shape: [ N , H * action_dim]

            if self.num_actor_traj != 0:
                # Combine the actor action with gaussian action
                action_traj = np.concatenate((action_traj, actor_action_traj), axis=0)
                # Shape: [ num_gau_traj + num_actor_traj, H * action_dim]

            # Multiple particles go through the same action sequence
            action_traj = np.repeat(action_traj, self.particles, axis=0)
            # Shape: [ particles, num_gau_traj + num_actor_traj, H * action_dim]

            # actions clipped between -1 and 1
            action_traj = np.clip(action_traj, -1, 1)
            # Shape: [ particles, num_gau_traj + num_actor_traj, H * action_dim]

            state_traj = (
                torch.from_numpy(np.expand_dims(curr_state.copy(), axis=0)).float().to(self.device)
            )
            # Shape: [1, network_size, (num_gau_traj + num_actor_traj) * particles, reward_dim (+ cost_dim) + state_dim]

            var_traj = torch.zeros([1,curr_state.shape[0],curr_state.shape[1],1]).float().to(self.device)
            # Shape: [1, network_size, (num_gau_traj + num_actor_traj) * particles, 1]
            actions = np.repeat(
                np.expand_dims(action_traj, axis=0), self.models.model.network_size, axis=0
            )
            # Shape: [ network_size, particles, num_gau_traj + num_actor_traj, H * action_dim]

            actions = torch.FloatTensor(actions).to(self.device)
            # Shape: [ network_size, particles, num_gau_traj + num_actor_traj, H * action_dim]
            for current_horizon in range(self.horizon):
                states_h = state_traj[current_horizon, :, :, self.state_start_dim:]
                # [ network_size, (num_gau_traj + num_actor_traj) * particles, state_dim]
                # use all dynamics model to predict next state (all_model=True)
                next_states, next_var = self.models.get_forward_prediction_random_t(
                    states_h,
                    actions[
                        :,
                        :,
                        current_horizon * self.action_dim : (current_horizon + 1) * self.action_dim,
                    ],
                    all_model=True,
                    repeat_network=False,
                )
                # next_states and var shape are [ network_size, (num_gau_traj + num_actor_traj) * particles, reward_dim (+ cost_dim) + state_dim]

                #protection for producing nan in rare cases
                next_states = torch.clamp(next_states,-self.obs_clip, self.obs_clip)
                next_states = torch.nan_to_num(next_states)

                state_traj = torch.cat((state_traj, next_states.unsqueeze(0)), axis=0)
                # [ horizon + 1, network_size, (num_gau_traj + num_actor_traj) * particles, reward_dim (+ cost_dim) + state_dim]
                
                next_var = next_var[:,:,self.state_start_dim:].sqrt().norm(dim=2).unsqueeze(2)
                # [network_size, (num_gau_traj + num_actor_traj) * particles,1]
                
                var_traj = torch.cat((var_traj, next_var.unsqueeze(0)), axis=0)
                # [ horizon + 1, network_size, (num_gau_traj + num_actor_traj) * particles, 1]
                
            state_traj = state_traj.cpu().detach().numpy()

            # [ horizon + 1, network_size, (num_gau_traj + num_actor_traj) * particles, reward_dim (+ cost_dim) + state_dim]
            var_traj_numpy = var_traj.detach().cpu().numpy()
            del var_traj

            if self.env.env_type == 'mujoco-terminated':
                done = np.zeros((state_traj.shape[1], state_traj.shape[2], 1))
                # Shape :[network_size, (num_gau_traj + num_actor_traj) * particles, 1]
                
                # Set the reward of terminated states to zero
                for current_horizon in range(1, self.horizon + 1):
                    for ens in range(state_traj.shape[1]):
                        # check the state whether terminate
                        done[ens, :, :] = np.logical_or(
                            done[ens, :, :],
                            self.termination_function(None, None, state_traj[current_horizon, ens, :, self.state_start_dim:]),
                        )
                        not_done = 1 - done[ens, :, :]
                        # Set the reward of terminated states to zero
                        state_traj[current_horizon, ens, :, 0] *= not_done.astype(np.float32).reshape(-1)

            # Find average cost of each trajectory
            returns = np.zeros((self.num_gaussian_traj + self.num_actor_traj,))
            safety_costs = np.zeros((self.num_gaussian_traj + self.num_actor_traj,))
            trajectory_max_vars = np.zeros((self.num_gaussian_traj + self.num_actor_traj,))

            # Shape: [ num_gau_traj + num_actor_traj,  1 ]
            if self.algo == 'safe-loop':
                terminal_reward = self.compute_terminal_reward(action_traj,state_traj)
                # [ (num_gau_traj + num_actor_traj) * particles, 1]

            if self.env.env_type == 'gym':
                all_safety_costs = self.compute_cost_from_state(state_traj)
                # [ horizon+1, network_size, (num_gau_traj + num_actor_traj) * particles, 1]

            # Calculate the average reward and max cost of N action trajectory,
            # each action trajectory have generated (network_size * particles) state-action trajectory
            # using (network_size * particles) ensemble models.
            for ensemble in self.models.model.elite_model_idxes:
                if self.algo=='safe-loop':
                    if self.env.env_type == 'mujoco-terminated':
                        done[ensemble, :, :] = np.logical_or(
                            done[ensemble, :, :],
                            self.termination_function(
                                None, None, state_traj[self.horizon - 1, ensemble, :, self.state_start_dim:]
                            ),
                        )
                        not_done = 1 - done[ensemble, :, :]
                        q_rews = terminal_reward[ensemble, :] * not_done.reshape(-1)
                    else:
                        q_rews = terminal_reward[ensemble, :]

                traj_indices = np.arange(0, self.num_gaussian_traj + self.num_actor_traj, 1).astype(int)
                for particle in range(self.particles):
                    returns[traj_indices] += \
                        np.sum(
                            state_traj[
                                1:self.horizon+1,
                                ensemble,
                                traj_indices * self.particles + particle,
                                0,
                            ],
                            axis=0,
                        )
                    if self.algo=='safe-loop':
                        returns[traj_indices] += q_rews.reshape(-1)[traj_indices * self.particles + particle]
                    if self.env.env_type=='gym':
                        # use state that dynamics predict to compute cost
                        safety_costs[traj_indices] = np.maximum(
                            safety_costs,
                            np.sum(
                                all_safety_costs[
                                    0 : self.horizon,
                                    ensemble,
                                    traj_indices * self.particles + particle,
                                    0,
                                ],
                                axis=0,
                            ),
                        )
                    elif self.env.env_type=='mujoco-speed':
                        # use cost that dynamics predict at dimension one
                        safety_costs[traj_indices] += (
                            np.sum(
                                state_traj[
                                    1: self.horizon+1,
                                    ensemble,
                                    traj_indices * self.particles + particle,
                                    1,
                                ],
                                axis=0,
                            )
                        )
                    if self.algo == 'cap':
                        trajectory_max_vars[traj_indices] += np.maximum(
                            trajectory_max_vars,
                            np.sum(
                                var_traj_numpy[
                                    1 : self.horizon+1,
                                    ensemble,
                                    traj_indices * self.particles + particle,
                                    0,
                                ],
                                axis=0,
                            ),
                        )
            returns /= state_traj.shape[1] * self.particles
            # [ num_gau_traj + num_actor_traj, 1]
            
            if self.algo == 'safe-loop':
                new_mean, new_var, safety_costs_mean = self.safe_loop_elite_select(returns, safety_costs, action_traj)
                if new_mean:
                    mean = new_mean
                else: # rare case for protecting bug
                    break
            elif self.algo == 'cap':
                safety_costs /= state_traj.shape[1] * self.particles
                if self.cfgs['cost_gamma'] == 1.0:
                    c_gamma_discount = self.cfgs['max_ep_len'] / self.horizon
                    # Extend the cost to the entire trajectory
                else:
                    c_gamma_discount = (1/self.horizon) * ((1 - self.cfgs['cost_gamma']**self.cfgs['max_ep_len']) / (1 - self.cfgs['cost_gamma']))
                    # average the cost, then view it as the starting element of the arithmetic progression
                safety_costs = c_gamma_discount * safety_costs
                
                penalty = self.lambda_range_projection(self.lagrangian_multiplier).item()
                safety_costs = safety_costs  #+ penalty * trajectory_max_vars
                mean, new_var, safety_costs_mean = self.safe_loop_elite_select(returns, safety_costs, action_traj) #self.cap_elite_select2(returns, safety_costs, action_traj)
            var = (self.alpha_plan) * var + (1 - self.alpha_plan) * new_var
            current_iter += 1
            
            del state_traj, action_traj
            
            # Initialize the var every 6 times
            if (current_iter + 1) % 6 == 0:
                var = np.tile(
                    np.square(self.env.action_space.high[0] - self.env.action_space.low[0]) / 16.0,
                    [self.sol_dim],
                ) * (1.5 ** ((current_iter + 1) // 6))

            # If safe trajectory not enough and t>5  or t>25 ,then break
            if (
                ((safety_costs < self.safety_threshold).sum() >= self.minimal_elites)
                and current_iter > 5
            ) or current_iter > 25:
                break

        # Store the mean and use it in next plan
        self.mean = mean

        # Return [1, action_dim], that is the first action of H horizon action mean, which shape is [1, H * action_dim]
        return mean[: self.action_dim], safety_costs_mean

    def cap_elite_select_TODO1(self, returns, safety_costs, action_traj):
        """TODO"""
        # returns shape : [ num_gau_traj + num_actor_traj, 1]
        # safety_costs shape : [ num_gau_traj + num_actor_traj, 1]
        # action_traj shape : [ (num_gau_traj + num_actor_traj) * particle,  H * action_dim]
        safety_costs_mean = np.mean(safety_costs)
        if((safety_costs<self.safety_threshold).sum()<self.minimal_elites):
            indices = np.argsort(safety_costs)
            indices*=self.particles
            elites = action_traj[indices][:self.minimal_elites]
        else:                 
            costs = -returns * (safety_costs<self.safety_threshold) + (safety_costs>=self.safety_threshold)*1e4
            indices = np.argsort(costs)
            indices = np.array([idx for idx in indices if costs[idx]<1e3])
            indices*=self.particles
            elites = action_traj[indices][:min(self.minimal_elites,indices.shape[0])]
        mean = np.mean(elites, axis=0)
        new_var = np.var(elites, axis=0)
        return mean, new_var, safety_costs_mean

    def cap_elite_select_TODO2(self, returns, safety_costs, action_traj):
        """TODO"""
        # returns shape : [ num_gau_traj + num_actor_traj, 1]
        # safety_costs shape : [ num_gau_traj + num_actor_traj, 1]
        # action_traj shape : [ (num_gau_traj + num_actor_traj) * particle,  H * action_dim]        
        all_action = action_traj[
            np.arange(0, self.num_gaussian_traj + self.num_actor_traj, 1).astype(int)
            * self.particles,
            :,
        ]
        # all_action is [ num_gau_traj + num_actor_traj, H * action_dim]
        
        # find the index for safe traj
        feasible_ids = (safety_costs <= self.safety_threshold).nonzero()[0]
        if feasible_ids.shape[0] < self.minimal_elites:
            # if safe traj not enough
            elite_ids = np.argsort(safety_costs)[:self.minimal_elites]
        else:  
            # if have enough safe trajectory
            # select the top k reward in safe action traj
            elite_ids = feasible_ids[np.argsort(-returns[feasible_ids])][:self.minimal_elites]
      
        elite_action = all_action[elite_ids]
        # [ elite_ids, H * action_dim]
        
        mean = np.mean(elite_action, axis=0)
        # [ 1, H * action_dim]

        var = np.var(elite_action, axis=0)
        # [ 1,  H * action_dim]

        return mean, var

    def safe_loop_elite_select(self, returns, safety_costs, action_traj):
        """Update mean and var using reward and cost"""
        # returns shape : [ num_gau_traj + num_actor_traj, 1]
        # safety_costs shape : [ num_gau_traj + num_actor_traj, 1]
        # action_traj shape : [ (num_gau_traj + num_actor_traj) * particle,  H * action_dim]
        safety_costs_mean = np.mean(safety_costs)

        if (safety_costs < self.safety_threshold).sum() < self.minimal_elites:
            safety_rewards = -safety_costs
            # [ num_gau_traj + num_actor_traj, 1]

            max_safety_reward = np.max(safety_rewards)
            # [1,1]

            score = np.exp(self.kappa * (safety_rewards - max_safety_reward))
            # [ num_gau_traj + num_actor_traj, 1]

            indices = np.argsort(safety_costs)
            # [ num_gau_traj + num_actor_traj, 1]

            mean = np.sum(
                action_traj[
                    np.arange(0, self.num_gaussian_traj + self.num_actor_traj, 1).astype(int)
                    * self.particles,
                    :,
                ]
                * score.reshape(-1, 1),
                axis=0,
            ) / (np.sum(score) + 1e-10)
            # [1, H * action_dim],  action_traj is [ num_gau_traj + num_actor_traj, H * action_dim], score is  [ num_gau_traj + num_actor_traj, 1]

            new_var = np.average(
                (
                    action_traj[
                        np.arange(0, self.num_gaussian_traj + self.num_actor_traj, 1).astype(int)
                        * self.particles,
                        :,
                    ]
                    - mean
                )
                ** 2,
                weights=score.reshape(-1),
                axis=0,
            )
        # [ 1,  H * action_dim]

        else:  # if have enough safe trajectory
            # safe trajectory's costs is -reward, unsafe trajectory's costs is 1e4
            costs = -returns * (safety_costs < self.safety_threshold) + (
                safety_costs >= self.safety_threshold
            ) * 1e4
            # [ num_gau_traj + num_actor_traj, 1]

            # select indices of safe trajectory
            indices = np.arange(costs.shape[0])
            indices = np.array([idx for idx in indices if costs[idx] < 1e3])
            # [ num_safe_traj, 1]

            # rare case
            if indices.shape[0] == 0 or action_traj.shape[0] == 0:
                return False,False,False
            
            safe_action_traj = action_traj[
                np.arange(0, self.num_gaussian_traj + self.num_actor_traj, 1).astype(int)
                * self.particles,
                :,
            ][indices, :]
            # [ num_safe_traj, H * action_dim]

            # use safe trajectory and its reward as weight to update
            rewards = -costs[indices]
            # [ num_safe_traj, 1 ]

            max_reward = np.max(rewards)
            # [ 1, 1 ]

            score = np.exp(self.kappa * (rewards - max_reward))
            # [ num_safe_traj, 1 ]

            mean = np.sum(safe_action_traj * score.reshape(-1, 1), axis=0) / (
                np.sum(score) + 1e-10
            )
            # [1, H * action_dim] = [1, H * action_dim] / [1,1]

            new_var = np.average(
                (safe_action_traj - mean) ** 2, weights=score.reshape(-1), axis=0
            )
            # [ 1,  H * action_dim]
        return mean, new_var, safety_costs_mean

def default_termination_function(state, action, next_state):  # pylint: disable=unused-argument
    '# Default termination function that outputs done=False'
    if torch.is_tensor(next_state):
        done = torch.zeros((next_state.shape[0], 1))
    else:
        done = np.zeros((next_state.shape[0], 1))
    return done


class CCEM_Planner:
    def __init__(self,
                device,
                env,
                models,
                horizon,
                popsize,
                particles,
                max_iters,
                alpha,
                mixture_coefficient,
                minimal_elites,
                obs_clip,
                cost_constrained=True,
                penalize_uncertainty=True,
                ):
        self.dO, self.dU = env.observation_space.shape[0], env.action_space.shape[0]
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low
        self.gamma = self.cfgs['gamma']
        self.c_gamma = self.cfgs['cost_gamma']
        self.cost_limit =  self.cfgs['lagrange_cfgs']['cost_limit']
        self.cost_constrained = cost_constrained
        self.penalize_uncertainty = penalize_uncertainty
        self.device = device
        self.obs_clip = obs_clip
        # CEM parameters
        self.particles = particles
        self.horizon = horizon
        self.popsize = popsize
        self.minimal_elites = minimal_elites
        self.max_iters = max_iters
        self.alpha = alpha
        self.epsilon = 0.001
        self.lb = np.tile(self.ac_lb, [self.horizon])
        self.ub = np.tile(self.ac_ub, [self.horizon])
        self.decay = 1.25
        self.elite_fraction = 0.3
        self.elites = None

        self.ac_buf = np.array([]).reshape(0, self.dU)
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.horizon])
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.horizon])

        self.models = models
        self.step = 0

    def get_action(self, obs):
        if self.models is None:
            return np.random.uniform(self.ac_lb, self.ac_ub, self.ac_lb.shape)
        if self.ac_buf.shape[0] > 0:
            action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
            return action

        soln = self.obtain_solution(obs, self.prev_sol, self.init_var)
        self.prev_sol = np.concatenate([np.copy(soln)[self.dU:], np.zeros(self.dU)])
        self.ac_buf = soln[:self.dU].reshape(-1, self.dU)

        return self.get_action(obs)

    def obtain_solution(self, obs, init_mean, init_var):
        mean, var, t = init_mean, init_var, 0
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))

        while (t < self.max_iters) and np.max(var) > self.epsilon:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            noise = X.rvs(size=[self.popsize, self.horizon * self.dU])

            samples = noise * np.sqrt(constrained_var) + mean
            samples = samples.astype(np.float32)
            
            rewards, costs, eps_lens = self.rollout(obs, samples)
            epoch_ratio = np.ones_like(eps_lens) * self.cfgs['max_ep_len'] / self.horizon
            terminated = eps_lens != self.horizon
            if self.c_gamma == 1:
                c_gamma_discount = epoch_ratio
            else:
                c_gamma_discount = (1 - self.c_gamma ** (epoch_ratio * self.horizon)) / (1 - self.c_gamma) / self.horizon
            rewards = rewards * epoch_ratio
            costs = costs * c_gamma_discount

            feasible_ids = ((costs <= self.cost_limit) & (~terminated)).nonzero()[0]
            if self.cost_constrained:
                if feasible_ids.shape[0] >= self.minimal_elites:
                    elite_ids = feasible_ids[np.argsort(-rewards[feasible_ids])][:self.minimal_elites]
                else:
                    elite_ids = np.argsort(costs)[:self.minimal_elites]
            else:
                elite_ids = np.argsort(-rewards)[:self.minimal_elites]
            self.elites = samples[elite_ids]
            new_mean = np.mean(self.elites, axis=0)
            new_var = np.var(self.elites, axis=0)
            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var
            t += 1
        
        self.step += 1
        return mean

    @torch.no_grad()
    def rollout(self, obs, ac_seqs):
        # obs: [obs_dim,]
        # ac_seqs: [popsize, horizon * action_dim] 

        ac_seqs = torch.from_numpy(ac_seqs).float().to(self.device)
        ac_seqs = ac_seqs.view(-1, self.horizon, self.dU)
        transposed = ac_seqs.transpose(0, 1)
        expanded = transposed[:, :, None]
        tiled = expanded.expand(-1, -1, self.particles, -1)
        ac_seqs = tiled.contiguous().view(self.horizon, -1, self.dU)
        
        # Expand current observation
        cur_obs = torch.from_numpy(obs).float().to(self.device)
        cur_obs = cur_obs[None]
        cur_obs = cur_obs.expand(self.popsize * self.particles, -1)
        # cur_obs: [popsize * npart, obs_dim] 
        rewards = torch.zeros(self.popsize, self.particles, device=self.device)
        costs = torch.zeros(self.popsize, self.particles, device=self.device)
        length = torch.zeros(self.popsize, self.particles, device=self.device)

        for t in range(self.horizon):
            cur_acs = ac_seqs[t]
            cur_obs, reward, cost = self._predict_next(cur_obs, cur_acs)
            # Clip state value
            cur_obs = torch.clamp(cur_obs, -self.obs_clip, self.obs_clip)
            reward = reward.view(-1, self.particles)
            cost = cost.view(-1, self.particles)

            rewards += reward
            costs += cost
            length += 1

        # Replace nan with high cost
        rewards[rewards != rewards] = -1e6
        costs[costs != costs] = 1e6

        return rewards.mean(dim=1).detach().cpu().numpy(), costs.mean(dim=1).detach().cpu().numpy(), length.mean(dim=1).detach().cpu().numpy()

    def _predict_next(self, obs, acs):
        # obs: [popsize * npart, obs_dim] 
        proc_obs = self._expand_to_ts_format(obs)
        # [network_size, popsize*npart/network_size, state_dim]
        proc_acs = self._expand_to_ts_format(acs)
        output = self.models.predic_cap(proc_obs, proc_acs)
        next_obs, var = output["state"]
        # [network_size, popsize*npart/network_size, state_dim]
        reward, _ = output["reward"]
        cost, _ = output["cost"]
        next_obs = self._flatten_to_matrix(next_obs)
        # [popsize*npart, state_dim]
        reward = self._flatten_to_matrix(reward)
        cost = self._flatten_to_matrix(cost)

        obs = obs.detach().cpu().numpy()
        acs = acs.detach().cpu().numpy()

        if self.cost_constrained and self.penalize_uncertainty:
            # var shape: [network_size, popsize*npart/network_size, state_dim]
            var_penalty = var.sqrt().norm(dim=2).max(0)[0]
            # cost_penalty: [popsize*npart/network_size]
            var_penalty = var_penalty.repeat_interleave(self.models.network_size).view(cost.shape)
            # cost_penalty: [popsize*npart, 1]
            penalty = self.lambda_range_projection(self.lagrangian_multiplier).item()
            cost += penalty * var_penalty

        return next_obs, reward, cost

    def _expand_to_ts_format(self, mat):
        dim = mat.shape[-1] 
        # eg:state_dim
        reshaped = mat.view(-1, self.models.network_size, self.particles // self.models.network_size, dim)
        # [popsize, network_size, npart // network_size, state_dim]
        transposed = reshaped.transpose(0, 1)
        # [network_size, popsize, npart // network_size, state_dim]
        reshaped = transposed.contiguous().view(self.models.network_size, -1, dim)
        # [network_size, popsize * npart / network_size, state_dim]
        
        return reshaped

    def _flatten_to_matrix(self, ts_fmt_arr):
        dim = ts_fmt_arr.shape[-1]
        reshaped = ts_fmt_arr.view(self.models.network_size, -1, self.particles // self.models.network_size, dim)
        transposed = reshaped.transpose(0, 1)
        reshaped = transposed.contiguous().view(-1, dim)
        return reshaped