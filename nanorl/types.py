"""Type definitions."""

from typing import Any, NamedTuple

import jax
import numpy as np

Params = Any
PRNGKey = jax.random.KeyArray

LogDict = dict[str, float]


class Transition(NamedTuple):
    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    discount: np.ndarray
    next_observation: np.ndarray
