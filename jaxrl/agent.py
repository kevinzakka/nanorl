"""Base agent class."""

import abc
from typing import Tuple

import numpy as np
from flax import struct

from jaxrl.types import LogDict, Transition


class Agent(abc.ABC, struct.PyTreeNode):
    """Base agent abstraction."""

    @staticmethod
    @abc.abstractmethod
    def initialize(*args, **kwargs) -> "Agent":
        """Initializes the agent."""

    @abc.abstractmethod
    def update(self, transitions: Transition) -> tuple["Agent", LogDict]:
        """Updates the agent's parameters given a batch of transitions."""

    @abc.abstractmethod
    def sample_actions(self, observations: np.ndarray) -> Tuple["Agent", np.ndarray]:
        """Selects actions given observations during training."""

    @abc.abstractmethod
    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        """Selects actions given observations during evaluation."""
