from typing import Optional, Tuple

import numpy as np
import pymc as pm
from pytensor.tensor import TensorVariable


def logp(value: TensorVariable, mu: TensorVariable) -> TensorVariable:
    return -(value - mu) ** 2


def random(
        mu: np.ndarray | float,
        rng: Optional[np.random.Generator] = None,
        size: Optional[Tuple[int]] = None,
) -> np.ndarray | float:
    return rng.normal(loc=mu, scale=1, size=size)


with pm.Model():
    mu = pm.Normal('mu', 0, 1)
    pm.CustomDist(
        'custom_dist',
        mu,
        logp=logp,
        random=random,
        observed=np.random.randn(100, 3),
        size=(100, 3),
    )
    prior = pm.sample_prior_predictive(10)
