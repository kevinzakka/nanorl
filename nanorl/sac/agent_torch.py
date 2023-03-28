"""Soft Actor-Critic implementation in torch."""
import os

import random
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from nanorl import SACConfig
from nanorl.specs import EnvironmentSpec, zeros_like
from nanorl.torch_networks import (
    MLP,
    Ensemble,
    Normal,
    StateActionValue,
)
from nanorl.types import LogDict, Transition


class Temperature(nn.Module):
    def __init__(self, initial_temperature: float = 1.0) -> None:
        super(Temperature, self).__init__()
        self.initial_temperature = initial_temperature
        self.log_temp = nn.Parameter(torch.tensor([initial_temperature]).log())

    def forward(self) -> torch.Tensor:
        return torch.exp(self.log_temp)



class SAC(nn.Module):
    @staticmethod
    def initialize(
        spec: EnvironmentSpec,
        config: SACConfig,
        seed: int = 0,
        discount: float = 0.99,
    ) -> Tuple["SAC", torch.optim.Optimizer, torch.optim.Optimizer]:
        """Create a SAC agent."""
        has_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if has_cuda else "cpu")

        # set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # get shapes
        action_dim = spec.action.shape[-1]
        observations = torch.zeros(*spec.observation.shape)
        actions = torch.zeros(*spec.action.shape)

        if config.target_entropy is None:
            target_entropy = -0.5 * action_dim
        else:
            target_entropy = config.target_entropy

        # create actor
        actor = Normal(
            MLP(
                input_dim=observations.shape[-1],
                hidden_dims=config.hidden_dims,
                activation=getattr(nn, config.activation)(),
                activate_final=True,
            ),
            module_output_dim=config.hidden_dims[-1],
            action_dim=action_dim,
            squash_tanh=True,
        )

        # create critic
        _critic_constructor = lambda: StateActionValue(
            MLP(
                input_dim=observations.shape[-1] + actions.shape[-1],
                hidden_dims=config.hidden_dims,
                activation=getattr(nn, config.activation)(),
                activate_final=True,
                dropout_rate=config.critic_dropout_rate,
                use_layer_norm=config.critic_layer_norm,
            ),
            module_output_dim=config.hidden_dims[-1],
        )
        critic = Ensemble(_critic_constructor, num=config.num_qs)
        target_critic = Ensemble(_critic_constructor, num=config.num_qs)
        target_critic.load_state_dict(critic.state_dict())

        # create temperature
        temperature = Temperature(config.init_temperature)

        actor.to(device)
        critic.to(device)
        target_critic.to(device)
        temperature.to(device)

        if os.getenv("TORCH_COMPILE", "0") == "1":
            # actor = torch.compile(actor)
            critic = torch.compile(critic)
            target_critic = torch.compile(target_critic)
            # temperature = torch.compile(temperature)

        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
        temp_optimizer = torch.optim.Adam(temperature.parameters(), lr=config.temp_lr)

        agent = SAC(
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temp=temperature,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            temp_optimizer=temp_optimizer,
            tau=config.tau,
            discount=discount,
            target_entropy=target_entropy,
            critic_utd_ratio=config.critic_utd_ratio,
            actor_utd_ratio=config.actor_utd_ratio,
            num_qs=config.num_qs,
            num_min_qs=config.num_min_qs,
            backup_entropy=config.backup_entropy,
            device=device,
        )
        return agent

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        target_critic: nn.Module,
        temp: Temperature,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        temp_optimizer: torch.optim.Optimizer,
        tau: float,
        discount: float,
        target_entropy: Optional[float],
        critic_utd_ratio: int,
        actor_utd_ratio: int,
        num_qs: int,
        num_min_qs: Optional[int],
        backup_entropy: bool,
        device: torch.device,
    ):
        super(SAC, self).__init__()
        self._actor = actor
        self._critic = critic
        self._target_critic = target_critic
        self._temp = temp
        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer
        self._temp_optimizer = temp_optimizer

        self._tau = tau
        self._discount = discount
        self._target_entropy = target_entropy
        self._critic_utd_ratio = critic_utd_ratio
        self._actor_utd_ratio = actor_utd_ratio
        self._num_qs = num_qs
        self._num_min_qs = num_min_qs
        self._backup_entropy = backup_entropy

        self._device = device

    @property
    def critic_utd_ratio(self) -> int:
        return self._critic_utd_ratio

    @property
    def actor_utd_ratio(self) -> int:
        return self._actor_utd_ratio

    def update_actor(self, transitions: Transition) -> LogDict:
        """Update the actor."""
        dist = self._actor(transitions.observation)
        # actions, log_probs = dist.sample_and_log_prob()
        actions = dist.rsample()
        log_probs = dist.log_prob(actions).sum(axis=-1)
        qs = self._critic(transitions.observation, actions)
        q = qs.mean(axis=0)  # why mean and not min
        actor_loss = (log_probs * self._temp() - q).mean()

        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()

        log_dict = {
            "actor_loss": actor_loss.item(),
            "entropy": -log_probs.mean().item(),
        }
        return log_dict

    def update_temperature(self, entropy: float) -> LogDict:
        temp = self._temp()
        temp_loss = temp * (entropy - self._target_entropy) # TODO why no mean
        self._temp_optimizer.zero_grad()
        temp_loss.backward()
        self._temp_optimizer.step()
        return {"temperature": temp.item(), "temperature_loss": temp_loss.item()}

    def update_critic(self, transitions: Transition) -> LogDict:
        """
        Update the critic.
        """
        with torch.no_grad():
            dist = self._actor(transitions.next_observation)
            next_actions = dist.rsample()
            next_log_probs = dist.log_prob(next_actions).sum(axis=-1)
            next_qs = self._target_critic(transitions.next_observation, next_actions)
            next_q = next_qs.min(axis=0)[0]
            target_q = transitions.reward + self._discount * transitions.discount * next_q
            if self._backup_entropy:
                target_q -= (
                    self._discount * transitions.discount * self._temp() * next_log_probs
                )

        qs = self._critic(transitions.observation, transitions.action)
        critic_loss = ((qs - target_q) ** 2).mean()

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        def incremental_update(ac, ac_targ, polyak_tau):
            with torch.no_grad():
                for param, target_param in zip(ac.parameters(), ac_targ.parameters()):
                    target_param.data.copy_(
                        polyak_tau * param.data + (1 - polyak_tau) * target_param.data
                    )

        incremental_update(self._critic, self._target_critic, self._tau)
        return {"critic_loss": critic_loss.item(), "q": qs.mean().item()}

    def update(self, transitions: Transition) -> Tuple["SAC", LogDict]:
        """Perform one update step using pytorch"""

        # convert to torch
        transitions = Transition(
            *[torch.as_tensor(x, dtype=torch.float32, device=self._device) for x in transitions]
        )

        # Update critic.
        for i in range(self._critic_utd_ratio):
            def slice(x):
                batch_size = x.shape[0] // self._critic_utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_transition = Transition(*[slice(x) for x in transitions])
            critic_info = self.update_critic(mini_transition)

        # Update actor.
        for i in range(self._actor_utd_ratio):

            def slice(x):
                batch_size = x.shape[0] // self._actor_utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_transition = Transition(*[slice(x) for x in transitions])
            actor_info = self.update_actor(mini_transition)

        # Update temperature.
        temp_info = self.update_temperature(actor_info["entropy"])

        return self, {**actor_info, **critic_info, **temp_info}

    @torch.no_grad()
    def sample_actions(self, observations: np.ndarray) -> Tuple["SAC", np.ndarray]:
        t_observations = torch.from_numpy(observations).float().to(self._device)
        dist = self._actor(t_observations)
        actions = dist.sample()
        return self, actions.cpu().detach().numpy()

    @torch.no_grad()
    def eval_actions(self, observations: np.ndarray) -> Tuple["SAC", np.ndarray]:
        t_observations = torch.from_numpy(observations).float().to(self._device)
        dist = self._actor(t_observations)
        actions = dist.mean()
        return self, actions.cpu().detach().numpy()
