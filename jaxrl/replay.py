"""Replay module."""

import pickle
import random
from pathlib import Path
from typing import Optional

import dm_env
import numpy as np
import psutil

from jaxrl.specs import EnvironmentSpec
from jaxrl.types import Transition


class _CircularBuffer:
    """A circular data buffer."""

    def __init__(
        self,
        capacity: int,
        offline_buffer: Optional[list] = None,
        offline_pct: float = 0.5,
    ) -> None:
        self._buffer: list = [None] * capacity
        self._prev: Optional[dm_env.TimeStep] = None
        self._action: Optional[np.ndarray] = None
        self._latest: Optional[dm_env.TimeStep] = None
        self._index: int = 0
        self._size: int = 0
        self._capacity = capacity
        self._offline_buffer = offline_buffer or []
        self._offline_pct = offline_pct
        self._n_offline = len(self._offline_buffer)

    @property
    def buffer(self):
        return self._buffer[: self._size]

    @property
    def offline_buffer(self):
        return self._offline_buffer

    def __len__(self) -> int:
        return self._size

    def is_ready(self, batch_size: int) -> bool:
        return batch_size <= len(self)

    def insert(self, timestep: dm_env.TimeStep, action: Optional[np.ndarray]) -> None:
        self._prev = self._latest
        self._action = action
        self._latest = timestep

        if action is not None:
            self._buffer[self._index] = (
                self._prev.observation,  # type: ignore
                self._action,
                self._latest.reward,
                self._latest.discount,
                self._latest.observation,
            )
            self._size = min(self._size + 1, self._capacity)
            self._index = (self._index + 1) % self._capacity

    def sample(self, batch_size: int) -> Transition:
        if self._n_offline > 0:
            n_offline = int(batch_size * self._offline_pct)
            n_online = batch_size - n_offline
            offline_indices = random.sample(range(self._n_offline), n_offline)
            online_indices = random.sample(range(len(self)), n_online)
            offline_data = [self._offline_buffer[i] for i in offline_indices]
            online_data = [self._buffer[i] for i in online_indices]
            data = offline_data + online_data
            random.shuffle(data)
            obs_tm1, a_tm1, r_t, d_t, obs_t = zip(*data)
        else:
            indices = random.sample(range(len(self)), batch_size)
            obs_tm1, a_tm1, r_t, d_t, obs_t = zip(*[self._buffer[i] for i in indices])
        return Transition(
            observation=np.stack(obs_tm1),
            action=np.asarray(a_tm1),
            reward=np.asarray(r_t),
            discount=np.asarray(d_t),
            next_observation=np.stack(obs_t),
        )


class ReplayBuffer:
    """A replay buffer that stores environment transitions in memory."""

    def __init__(
        self,
        capacity: int,
        batch_size: int,
        spec: EnvironmentSpec,
        offline_dataset: Optional[Path] = None,
        offline_pct: float = 0.5,
    ) -> None:
        """Constructor.

        Args:
            capacity: The maximum capacity of the replay buffer.
            batch_size: The batch size for sampling transitions.
            spec: An instance of `EnvironmentSpec`.
            offline_dataset: The path to a file on disk containing a previously saved
                replay buffer. If specified, the replay buffer will be initialized from
                the file.
            offline_pct: The percentage of transitions to sample from the offline
                dataset, between 0 and 1. The remaining transitions will be sampled from
                the online dataset.

        Raises:
            ValueError: If `offline_pct` is not in [0, 1].
            ValueError: If `offline_dataset` is specified but it is empty.
            ValueError: If the total size of the replay buffer, and that of the offline
                dataset if specified, exceeds the available memory.
        """
        if offline_dataset is not None:
            with open(offline_dataset, "rb") as f:
                offline_buffer = pickle.load(f)
            if any(transition is None for transition in offline_buffer):
                raise ValueError("The offline dataset contains empty transitions.")
        else:
            offline_buffer = None

        # A circular buffer to store transitions.
        self._data = _CircularBuffer(
            capacity=capacity, offline_buffer=offline_buffer, offline_pct=offline_pct
        )
        self._capacity = capacity
        self._batch_size = batch_size
        self._spec = spec

        if not 0 <= offline_pct <= 1:
            raise ValueError("`offline_pct` must be in [0, 1].")
        self._offline_pct = offline_pct

        self._check_memory_usage()

    def sample(self) -> Transition:
        """Samples a batch of transitions from the replay buffer.

        Returns:
            A batch of transitions.
        """
        return self._data.sample(self._batch_size)

    def insert(self, timestep: dm_env.TimeStep, action: Optional[np.ndarray]) -> None:
        """Inserts a transition into the replay buffer.

        Args:
            timestep: The timestep to insert.
            action: The action taken at the timestep.
        """
        self._data.insert(timestep, action)

    def is_ready(self) -> bool:
        """Returns True if the replay buffer has enough data to sample a batch."""
        return self._data.is_ready(self._batch_size)

    def _check_memory_usage(self) -> None:
        available_size = psutil.virtual_memory().available
        size = _get_size_in_bytes(self.capacity, self._spec)
        size += _get_size_in_bytes(len(self._data.offline_buffer), self._spec)
        if size > available_size:
            raise ValueError(
                f"The replay buffer of size {size / 1e9} GB exceeds the available "
                f"memory of {available_size / 1e9} GB."
            )

    # Public API.

    def size_in_bytes(self) -> int:
        """Returns the current size of the replay buffer in bytes."""
        return _get_size_in_bytes(len(self), self._spec)

    def __len__(self) -> int:
        """Returns the current size of the replay buffer.

        Can be smaller than the capacity if the buffer is not full.
        """
        return len(self._data)

    @property
    def capacity(self) -> int:
        """Returns the maximum capacity of the replay buffer."""
        return self._capacity

    @property
    def data(self):
        """Returns the data stored in the replay buffer."""
        return self._data.buffer

    @property
    def spec(self) -> EnvironmentSpec:
        """Returns the environment spec."""
        return self._spec

    @property
    def batch_size(self) -> int:
        """Returns the batch size."""
        return self._batch_size


def _get_size_in_bytes(n_elements: int, spec: EnvironmentSpec) -> int:
    """Caclulates the total size of `n_elements` transitions, in bytes."""
    total_size: int = 0
    # Observation and next observation.
    obs_dtype = spec.observation.dtype
    total_size += (spec.observation.shape[0] * obs_dtype.itemsize) * 2
    # Action.
    action_dtype = spec.action.dtype
    total_size += spec.action.shape[0] * action_dtype.itemsize
    # Reward, discount.
    total_size += np.float32().itemsize * 2
    return total_size * n_elements
