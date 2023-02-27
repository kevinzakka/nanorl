"""Base agent class."""

import abc
from typing import Tuple

import numpy as np
from flax import struct

from jaxrl.types import LogDict
from jaxrl.types import Transition


class Agent(abc.ABC, struct.PyTreeNode):
    """Base agent abstraction."""

    @staticmethod
    @abc.abstractmethod
    def initialize(*args, **kwargs) -> "Agent":
        ...

    @abc.abstractmethod
    def update(
        self, transitions: Transition, *args, **kwargs
    ) -> tuple["Agent", LogDict]:
        ...

    @abc.abstractmethod
    def sample_actions(self, observations: np.ndarray) -> Tuple["Agent", np.ndarray]:
        ...

    @abc.abstractmethod
    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        ...
