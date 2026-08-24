#!/usr/bin/env python3
"""Paired significance tools for the v2 candidate gate.

The v1 gate (`wins >= 2 and worst <= 5e-4`) accepted effects two orders of
magnitude smaller than its own threshold, so it degenerated into a 3-fold sign
test that passes 50% of the time under the null.  E16 cleared it with a
*positive* mean delta and the leaderboard then confirmed it was worse.  These
helpers replace that gate with an explicit paired confidence interval.
"""

from __future__ import annotations

import numpy as np

RANDOM_SEED = 42


def paired_brier_delta(
    y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray
) -> np.ndarray:
    """Per-row Brier difference. Negative favours the candidate."""
    y = np.asarray(y, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    c = np.asarray(candidate, dtype=np.float64)
    if not (y.shape == b.shape == c.shape):
        raise ValueError(f"Shape mismatch: y={y.shape}, base={b.shape}, cand={c.shape}")
    return np.square(c - y) - np.square(b - y)


def paired_bootstrap_brier_ci(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    iterations: int = 2000,
    seed: int = RANDOM_SEED,
    reference_brier: float | None = None,
    clusters: np.ndarray | None = None,
    confidence: float = 0.95,
) -> dict[str, float | int | bool | str]:
    """Paired percentile CI for mean Brier difference.

    When ``clusters`` is supplied, whole clusters are resampled.  The rolling
    harness uses pitcher_id because plate appearances from one pitcher are not
    independent observations.  This avoids the over-confident row bootstrap
    previously used by the v2 gate.
    """
    per_row = paired_brier_delta(y, baseline, candidate)
    n = len(per_row)
    if n == 0:
        raise ValueError("Empty fold")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    cluster_count = 0
    if clusters is None:
        for index in range(iterations):
            draws[index] = per_row[rng.integers(0, n, n)].mean()
        bootstrap_unit = "row"
    else:
        clusters = np.asarray(clusters)
        if clusters.shape != per_row.shape:
            raise ValueError(
                f"Cluster shape mismatch: clusters={clusters.shape}, rows={per_row.shape}"
            )
        _, inverse = np.unique(clusters.astype(str), return_inverse=True)
        cluster_count = int(inverse.max()) + 1
        sums = np.bincount(inverse, weights=per_row, minlength=cluster_count)
        counts = np.bincount(inverse, minlength=cluster_count)
        for index in range(iterations):
            selected = rng.integers(0, cluster_count, cluster_count)
            draws[index] = sums[selected].sum() / counts[selected].sum()
        bootstrap_unit = "cluster"
    tail = 100.0 * (1.0 - confidence) / 2.0
    low = float(np.percentile(draws, tail))
    high = float(np.percentile(draws, 100.0 - tail))
    point = float(per_row.mean())
    result: dict[str, float | int | bool | str] = {
        "n": int(n),
        "bootstrap_unit": bootstrap_unit,
        "cluster_count": cluster_count,
        "confidence": float(confidence),
        "iterations": int(iterations),
        "point": point,
        "ci_low": low,
        "ci_high": high,
        "bootstrap_standard_error": float(draws.std(ddof=1)),
        "significant": bool(high < 0.0),
    }
    if reference_brier is None:
        rate = float(np.asarray(y, dtype=np.float64).mean())
        reference_brier = rate * (1.0 - rate)
    if reference_brier > 0:
        scale = 100_000.0 / reference_brier
        result["score_point"] = -point * scale
        result["score_ci_low"] = -high * scale
        result["score_ci_high"] = -low * scale
    return result


def aggregate_gate(
    fold_intervals: dict[int, dict],
    primary_season: int = 2024,
    secondary_season: int = 2022,
) -> dict:
    """v2 candidate gate.

    Adopt when the primary fold (closest regime to 2025) improves significantly
    and the secondary fold does not degrade significantly.  The 2023 fold is
    recorded but never decides: it is the one-off futures-league label-regime
    break documented in EDA section 20.2.
    """
    primary = fold_intervals.get(primary_season)
    secondary = fold_intervals.get(secondary_season)
    if primary is None:
        raise ValueError(f"Primary fold {primary_season} is missing")
    primary_pass = bool(primary["significant"])
    secondary_block = bool(secondary is not None and secondary["ci_low"] > 0.0)
    return {
        "primary_season": primary_season,
        "secondary_season": secondary_season,
        "primary_point": primary["point"],
        "primary_ci": [primary["ci_low"], primary["ci_high"]],
        "primary_significant": primary_pass,
        "secondary_significantly_worse": secondary_block,
        "recorded_only": [s for s in fold_intervals if s not in (primary_season, secondary_season)],
        "gate_pass": bool(primary_pass and not secondary_block),
        "gate_definition": (
            f"{primary_season} fold clustered CI upper bound < 0 "
            f"AND {secondary_season} fold clustered CI lower bound <= 0"
        ),
        "confirmatory": False,
        "interpretation": (
            "Exploratory gate: the 2024 development fold can also be used for "
            "configuration selection; final confirmation is leaderboard-only."
        ),
    }
