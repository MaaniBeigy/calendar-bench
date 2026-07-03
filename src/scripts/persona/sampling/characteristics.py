"""Draw per-person characteristic values from declared distributions."""

from __future__ import annotations

import random
import warnings

import numpy as np

from src.scripts.persona.config.schema import (
    BooleanDist,
    CategoricalDist,
    CharacteristicDist,
    ScipyDist,
)
from src.scripts.persona.sampling.seeds import mix_axis_seed

_REJECTION_RETRIES = 100


def draw_axis_values(
    axis: str,
    dist: CharacteristicDist,
    persona_id: str,
    sample_size: int,
    root_seed: int,
) -> list[str | bool | int | float]:
    """Return `sample_size` values for `axis` drawn from `dist`."""
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}")
    seed = mix_axis_seed(root_seed, persona_id, axis)

    if isinstance(dist, CategoricalDist):
        rng = random.Random(seed)
        return _draw_categorical(dist.values, sample_size, rng)
    if isinstance(dist, BooleanDist):
        rng = random.Random(seed)
        weights = {True: dist.p_true, False: 1.0 - dist.p_true}
        return _draw_categorical(weights, sample_size, rng)
    if isinstance(dist, ScipyDist):
        return _draw_scipy(axis, dist, sample_size, seed)
    raise TypeError(f"unsupported distribution type: {type(dist).__name__}")


def _draw_categorical(weights: dict, sample_size: int, rng: random.Random) -> list:
    """Build a length-`sample_size` list whose realized counts match `weights`."""
    labels = list(weights.keys())
    counts = [int(round(weights[k] * sample_size)) for k in labels]
    diff = sample_size - sum(counts)
    if diff != 0:
        target = max(range(len(counts)), key=lambda i: counts[i])
        counts[target] += diff
    out: list = []
    for label, count in zip(labels, counts):
        out.extend([label] * count)
    rng.shuffle(out)
    return out


def _draw_scipy(
    axis: str, dist: ScipyDist, sample_size: int, seed: int
) -> list[int | float]:
    """Draw `sample_size` values from a `scipy.stats` distribution."""
    import scipy.stats as _ss

    rng = np.random.default_rng(seed)
    dist_obj = getattr(_ss, dist.type)(**dist.params)
    samples = np.asarray(dist_obj.rvs(size=sample_size, random_state=rng))
    if dist.clip is not None:
        samples = _resample_outside_clip(
            samples, dist_obj, dist.clip.min, dist.clip.max, rng, axis
        )
    if dist.dtype == "int":
        return [int(round(float(v))) for v in samples]
    if dist.dtype == "float":
        return [float(v) for v in samples]
    return [v.item() if hasattr(v, "item") else v for v in samples]


def _resample_outside_clip(
    samples: np.ndarray,
    dist_obj,
    clip_min: float,
    clip_max: float,
    rng: np.random.Generator,
    axis: str,
) -> np.ndarray:
    """Rejection-resample any out-of-range value; clip on exhaustion."""
    out = samples.astype(float, copy=True)
    for i in range(len(out)):
        retries = 0
        while not (clip_min <= out[i] <= clip_max):
            if retries >= _REJECTION_RETRIES:
                warnings.warn(
                    f"characteristic {axis!r}: clip exhausted after "
                    f"{_REJECTION_RETRIES} retries; clamping to bound",
                    stacklevel=3,
                )
                out[i] = min(max(out[i], clip_min), clip_max)
                break
            out[i] = float(dist_obj.rvs(size=1, random_state=rng)[0])
            retries += 1
    return out
