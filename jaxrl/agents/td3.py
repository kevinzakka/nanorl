"""Twin Delayed Deep Deterministic policy gradient implementation."""

from functools import partial
from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState

from jaxrl.agents import base
from jaxrl.distributions import TanhDeterministic
from jaxrl.networks import MLP, Ensemble, StateActionValue, subsample_ensemble
from jaxrl.specs import EnvironmentSpec, zeros_like
from jaxrl.types import LogDict, Transition


@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(
    rng,
    apply_fn,
    params,
    observations: np.ndarray,
    sigma: float,
    action_min: np.ndarray,
    action_max: np.ndarray,
):
    key, rng = jax.random.split(rng)
    action = apply_fn({"params": params}, observations)
    noise = jax.random.normal(key, shape=action.shape) * sigma
    return jnp.clip(action + noise, action_min, action_max), rng


@partial(jax.jit, static_argnames="apply_fn")
def _eval_actions(apply_fn, params, observations: np.ndarray):
    return apply_fn({"params": params}, observations)


class TD3(base.Agent):
    """Twin Delayed Deep Deterministic policy gradient (TD3)."""

    actor: TrainState
    target_actor: TrainState
    rng: Any
    critic: TrainState
    target_critic: TrainState
    tau: float
    discount: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(pytree_node=False)
    sigma: float
    target_sigma: float
    noise_clip: float
    action_min: np.ndarray
    action_max: np.ndarray

    @staticmethod
    def initialize(
        spec: EnvironmentSpec,
        seed: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_layer_norm: bool = False,
        sigma: float = 0.2,
        target_sigma: float = 0.2,
        noise_clip: float = 0.5,
    ) -> "TD3":

        action_dim = spec.action.shape[-1]
        observations = zeros_like(spec.observation)
        actions = zeros_like(spec.action)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        # Actor.
        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        actor_def = TanhDeterministic(actor_base_cls, action_dim)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        # Target actor.
        target_actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        # Critic.
        critic_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
            dropout_rate=critic_dropout_rate,
            use_layer_norm=critic_layer_norm,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )

        # Target critic.
        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        return TD3(
            actor=actor,
            target_actor=target_actor,
            rng=rng,
            critic=critic,
            target_critic=target_critic,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            sigma=sigma,
            target_sigma=target_sigma,
            noise_clip=noise_clip,
            action_min=spec.action.minimum,
            action_max=spec.action.maximum,
        )

    def update_actor(self, transitions: Transition):
        key, rng = jax.random.split(self.rng)

        def actor_loss_fn(actor_params):
            actions = self.actor.apply_fn(
                {"params": actor_params}, transitions.observation
            )
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                transitions.observation,
                actions,
                True,  # training.
                rngs={"dropout": key},
            )
            q = qs.mean(axis=0)
            actor_loss = -q.mean()
            return actor_loss, {"actor_loss": actor_loss}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        target_actor_params = optax.incremental_update(
            actor.params, self.target_actor.params, self.tau
        )
        target_actor = self.target_actor.replace(params=target_actor_params)

        return self.replace(actor=actor, target_actor=target_actor, rng=rng), actor_info

    def update_critic(self, transitions: Transition):
        rng = self.rng

        next_actions = self.target_actor.apply_fn(
            {"params": self.target_actor.params}, transitions.observation
        )

        key, rng = jax.random.split(rng)
        target_noise = jax.random.normal(key, next_actions.shape) * self.target_sigma
        target_noise = target_noise.clip(-self.noise_clip, self.noise_clip)
        next_actions = jnp.clip(
            next_actions + target_noise, self.action_min, self.action_max
        )

        # Used only for REDQ.
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key=key,
            params=self.target_critic.params,
            num_sample=self.num_min_qs,
            num_qs=self.num_qs,
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            transitions.next_observation,
            next_actions,
            True,  # training.
            rngs={"dropout": key},
        )
        next_q = next_qs.min(axis=0)

        target_q = transitions.reward + self.discount * transitions.discount * next_q

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params):
            qs = self.critic.apply_fn(
                {"params": critic_params},
                transitions.observation,
                transitions.action,
                True,  # training.
                rngs={"dropout": key},
            )
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    @partial(jax.jit, static_argnames=["critic_utd_ratio", "actor_utd_ratio"])
    def update(
        self, transitions: Transition, critic_utd_ratio: int, actor_utd_ratio: int
    ) -> tuple["TD3", LogDict]:
        new_agent = self

        # Update critic.
        for i in range(critic_utd_ratio):

            def slice(x: np.ndarray) -> np.ndarray:
                batch_size = x.shape[0] // critic_utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_transition = jax.tree_util.tree_map(slice, transitions)
            new_agent, critic_info = new_agent.update_critic(mini_transition)

        # Update actor.
        for i in range(actor_utd_ratio):

            def slice(x: np.ndarray) -> np.ndarray:
                batch_size = x.shape[0] // actor_utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_transition = jax.tree_util.tree_map(slice, transitions)
            new_agent, actor_info = new_agent.update_actor(mini_transition)

        return new_agent, {**critic_info, **actor_info}

    def sample_actions(self, observations: np.ndarray) -> tuple["TD3", np.ndarray]:
        actions, new_rng = _sample_actions(
            self.rng,
            self.actor.apply_fn,
            self.actor.params,
            observations,
            self.sigma,
            self.action_min,
            self.action_max,
        )
        return self.replace(rng=new_rng), np.asarray(actions)

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = _eval_actions(self.actor.apply_fn, self.actor.params, observations)
        return np.asarray(actions)
