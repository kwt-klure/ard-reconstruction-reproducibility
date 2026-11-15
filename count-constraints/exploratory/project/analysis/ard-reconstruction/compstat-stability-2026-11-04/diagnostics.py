"""Chunked, NumPy-only MCMC diagnostics for Stage A retained draws.

Methods: Vehtari et al. (2021), Bayesian Analysis 16(2), 667--718,
https://arxiv.org/abs/1903.08008. Algorithm conventions were checked against
ArviZ 0.22.0 on 2026-11-04 (not an installed runtime dependency):
https://github.com/arviz-devs/arviz/blob/v0.22.0/arviz/stats/diagnostics.py
https://github.com/arviz-devs/arviz/blob/v0.22.0/arviz/stats/stats_utils.py

R-hat is the maximum of rank-normalized and folded rank-normalized split
R-hat. Both ESS measures use split chains, biased (divisor N) FFT
autocovariances, between-chain variance, Geyer's initial-positive/monotone
sequence, and the ArviZ/Stan final-even-lag correction and log10(S) ESS cap.
Mean ESS uses original values, not normal scores. Mean MCSE is the original
pooled sample SD divided by sqrt(mean ESS), as in ArviZ's mean MCSE.

An invalid variable is explicitly marked and returns R-hat/MCSE = inf and
ESS = 0. Unlike ArviZ's constant-input special case, constant split chains
are conservatively flagged instead of assigned a full effective sample.
These are finite-run screens, not convergence or calibration guarantees.
"""

from functools import lru_cache
from statistics import NormalDist

import numpy as np


@lru_cache(maxsize=4)
def _normal_rank_lookup(size):
    """Exact stdlib inverse CDF at every possible average-rank score.

    Average ranks are half integers. Index 2*r-2 addresses rank r; this
    avoids a Python inverse-CDF call for every value in every variable.
    """
    normal = NormalDist()
    table = np.fromiter(
        (normal.inv_cdf((twice_rank / 2 - 0.375) / (size + 0.25))
         for twice_rank in range(2, 2 * size + 1)),
        dtype=np.float64,
        count=2 * size - 1,
    )
    table.setflags(write=False)
    return table


def _rank_normalize(values):
    """Pool chains/draws separately per variable, with average-rank ties."""
    flat = values.reshape(-1, values.shape[-1])
    size = flat.shape[0]
    order = np.argsort(flat, axis=0, kind="stable")
    sorted_values = np.take_along_axis(flat, order, axis=0)
    distinct = sorted_values[1:] != sorted_values[:-1]
    positions = np.arange(size, dtype=np.int64)[:, None]
    starts = np.maximum.accumulate(
        np.where(np.vstack((np.ones((1, flat.shape[1]), dtype=bool), distinct)),
                 positions, 0), axis=0,
    )
    ends = np.minimum.accumulate(
        np.where(np.vstack((distinct, np.ones((1, flat.shape[1]), dtype=bool))),
                 positions, size - 1)[::-1], axis=0,
    )[::-1]
    # 2*average_rank-2 equals the sum of the zero-based tie endpoints.
    scores = _normal_rank_lookup(size)[starts + ends]
    result = np.empty_like(flat, dtype=np.float64)
    np.put_along_axis(result, order, scores, axis=0)
    return result.reshape(values.shape)


def _split(values):
    half = values.shape[1] // 2
    return np.concatenate((values[:, :half], values[:, -half:]), axis=0)


def _rhat(values):
    n = values.shape[1]
    within = values.var(axis=1, ddof=1).mean(axis=0)
    between_over_n = values.mean(axis=1).var(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = np.sqrt((n - 1) / n + between_over_n / within)
    return np.where(np.isfinite(result), result, np.inf)


def _autocovariance(values):
    """Biased per-chain autocovariances, zero-padded to avoid circular lags."""
    n = values.shape[1]
    fft_length = 1 << (2 * n - 1).bit_length()
    centered = values - values.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered, n=fft_length, axis=1)
    spectrum *= spectrum.conjugate()
    return np.fft.irfft(spectrum, n=fft_length, axis=1)[:, :n] / n


def _ess(values):
    """Multichain ESS; the input already contains split chains."""
    chains, n, variables = values.shape
    covariance = _autocovariance(values).mean(axis=0)
    within = covariance[0] * n / (n - 1)
    variance_plus = covariance[0] + values.mean(axis=1).var(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rho = 1 - (within[None, :] - covariance) / variance_plus[None, :]
    rho[0] = 1

    # Include pairs through the same last candidate used by ArviZ/Stan.
    # The final candidate contributes its even lag only. Earlier pairs
    # stop at the first nonpositive pair and are made monotone decreasing.
    last_candidate = max((n - 3) // 2, 0)
    pair_sums = rho[:2 * last_candidate + 2:2] + rho[1:2 * last_candidate + 2:2]
    positive_prefix = np.logical_and.accumulate(pair_sums > 0, axis=0)
    stopping_pair = np.minimum(positive_prefix.sum(axis=0), last_candidate)
    included = np.arange(last_candidate + 1)[:, None] < stopping_pair[None, :]
    monotone_pairs = np.minimum.accumulate(pair_sums, axis=0)
    pair_total = np.where(included, monotone_pairs, 0).sum(axis=0)
    columns = np.arange(variables)
    final_even = rho[2 * stopping_pair, columns]
    final_pair = pair_sums[stopping_pair, columns]
    correction = np.where((final_pair >= 0) | (final_even > 0), final_even, 0)
    tau = np.maximum(-1 + 2 * pair_total + correction,
                     1 / np.log10(chains * n))
    return chains * n / tau


def diagnose(draws, *, chunk_size=32):
    """Diagnose a (chains, draws, variables) real array without mutating it.

    Require >=2 chains, >=8 retained draws per chain and >=1 variable.
    Odd draw counts discard the middle draw for split R-hat/ESS; MCSE's
    pooled SD uses all original draws (the ArviZ convention). Stage A has
    even counts, so no retained draw is discarded there.

    Returns one-dimensional arrays ``rhat``, ``bulk_ess``, ``mean_ess``,
    ``mean_mcse``, plus boolean ``nonfinite``, ``constant``,
    ``folded_constant`` and ``invalid`` arrays. ``constant`` means any
    split chain is exactly constant; no arbitrary small-range threshold
    is imposed. ``folded_constant`` means the pooled folded transform is
    constant and cannot supply a variance diagnostic. ``invalid`` also
    catches numerical overflow/underflow. Callers must preserve these
    masks if replacing infinite sentinels with JSON nulls.

    Working arrays are bounded by chunk_size, apart from the supplied
    input and O(variables) result. A NumPy memmap is accepted without
    converting the complete input to a floating-point copy.
    """
    draws = np.asanyarray(draws)
    if (draws.ndim != 3 or draws.shape[0] < 2 or draws.shape[1] < 8
            or draws.shape[2] < 1):
        raise ValueError("draws must have shape (chains>=2, draws>=8, variables>=1)")
    if not np.issubdtype(draws.dtype, np.number) or np.iscomplexobj(draws):
        raise TypeError("draws must contain real numeric values")
    if not isinstance(chunk_size, (int, np.integer)) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    variables = draws.shape[2]
    result = {name: np.full(variables, sentinel, dtype=np.float64)
              for name, sentinel in (("rhat", np.inf), ("bulk_ess", 0),
                                     ("mean_ess", 0), ("mean_mcse", np.inf))}
    for name in ("nonfinite", "constant", "folded_constant", "invalid"):
        result[name] = np.zeros(variables, dtype=bool)

    for start in range(0, variables, chunk_size):
        end = min(start + chunk_size, variables)
        block = np.asarray(draws[:, :, start:end], dtype=np.float64)
        finite = np.isfinite(block).all(axis=(0, 1))
        result["nonfinite"][start:end] = ~finite
        result["invalid"][start:end] = ~finite
        if not finite.any():
            continue
        indices = start + np.flatnonzero(finite)
        original = block[:, :, finite]
        split = _split(original)
        constant = (split.max(axis=1) == split.min(axis=1)).any(axis=0)
        result["constant"][indices] = constant
        result["invalid"][indices] |= constant
        if constant.all():
            continue
        indices = indices[~constant]
        original = original[:, :, ~constant]
        split = split[:, :, ~constant]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            folded = np.abs(split - np.median(split, axis=(0, 1)))
            folded_constant = folded.max(axis=(0, 1)) == folded.min(axis=(0, 1))
            z = _rank_normalize(split)
            rhat = np.maximum(_rhat(z), _rhat(_rank_normalize(folded)))
            bulk = _ess(z)
            mean = _ess(split)
            mcse = original.std(axis=(0, 1), ddof=1) / np.sqrt(mean)
        valid = (np.isfinite(rhat) & np.isfinite(bulk) & (bulk > 0)
                 & np.isfinite(mean) & (mean > 0) & np.isfinite(mcse)
                 & (mcse > 0) & ~folded_constant)
        result["folded_constant"][indices] = folded_constant
        result["invalid"][indices] |= ~valid
        for name, values in (("rhat", rhat), ("bulk_ess", bulk),
                             ("mean_ess", mean), ("mean_mcse", mcse)):
            result[name][indices[valid]] = values[valid]
    return result
