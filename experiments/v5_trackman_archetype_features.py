#!/usr/bin/env python3
"""Target-free, fixed-basis TrackMan pitcher archetype features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from experiments.run_e20r_rolling import (
    RICH_TRACKMAN_COLUMNS,
    rich_profile_table,
)


ARCHETYPE_CELL = "e82_trackman_archetype"
GROUPS = ("fastball", "breaking", "offspeed")
COMPONENTS = 8
CLUSTERS = 16
PC_COLUMNS = tuple(f"e82_archetype_pc{index + 1}" for index in range(COMPONENTS))
NUMERIC_COLUMNS = (
    *PC_COLUMNS,
    "e82_archetype_distance",
    "e82_archetype_margin",
    "e82_archetype_unseen",
)


@dataclass
class ArchetypeBasis:
    imputer: SimpleImputer
    scaler: StandardScaler
    pca: PCA
    kmeans: KMeans
    source_feature_names: list[str]
    source_pitchers: int


def _profile_matrix(profile: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    values: dict[str, pd.Series] = {}
    for group in GROUPS:
        rate = pd.to_numeric(profile[f"e58_{group}_rate"], errors="coerce").clip(
            0.0, 1.0
        )
        values[f"rate_{group}"] = rate
        weight = np.sqrt(rate.fillna(0.0))
        for metric in RICH_TRACKMAN_COLUMNS:
            overall = pd.to_numeric(profile[f"e58_{metric}_mean"], errors="coerce")
            group_mean = pd.to_numeric(
                profile[f"e58_{group}_{metric}_mean"], errors="coerce"
            )
            # A missing group mean means that the pitcher did not throw that
            # family in the completed history.  Its rate already records the
            # absence, so the weighted physical deviation is exactly zero.
            difference = (group_mean - overall).where(group_mean.notna(), 0.0)
            values[f"weighted_{group}_{metric}"] = weight * difference
    result = pd.DataFrame(values, index=profile.index, dtype=np.float64)
    return result, list(result.columns)


def fit_archetype_basis(joined_trackman: pd.DataFrame) -> ArchetypeBasis:
    """Fit the immutable representation on official 2019 TrackMan only."""
    source = joined_trackman.loc[
        joined_trackman["season"].eq(2019)
        & joined_trackman["game_type"].eq("R")
    ]
    profile = rich_profile_table(source)
    matrix, names = _profile_matrix(profile)
    if len(matrix) < CLUSTERS:
        raise ValueError(f"Only {len(matrix)} source pitchers for {CLUSTERS} clusters")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    pca = PCA(n_components=COMPONENTS, whiten=True, random_state=20260821)
    kmeans = KMeans(
        n_clusters=CLUSTERS,
        n_init=50,
        max_iter=500,
        random_state=20260821,
    )
    imputed = imputer.fit_transform(matrix)
    standardized = scaler.fit_transform(imputed)
    coordinates = pca.fit_transform(standardized)
    kmeans.fit(coordinates)
    return ArchetypeBasis(
        imputer=imputer,
        scaler=scaler,
        pca=pca,
        kmeans=kmeans,
        source_feature_names=names,
        source_pitchers=int(len(matrix)),
    )


def transform_profile_table(
    profile: pd.DataFrame,
    basis: ArchetypeBasis,
) -> pd.DataFrame:
    if profile.empty:
        return pd.DataFrame(columns=[*PC_COLUMNS, ARCHETYPE_CELL, *NUMERIC_COLUMNS[8:]])
    matrix, names = _profile_matrix(profile)
    if names != basis.source_feature_names:
        raise ValueError("Archetype source feature order changed")
    coordinates = basis.pca.transform(
        basis.scaler.transform(basis.imputer.transform(matrix))
    )
    distances = basis.kmeans.transform(coordinates)
    order = np.partition(distances, kth=1, axis=1)[:, :2]
    clusters = np.argmin(distances, axis=1)
    result = pd.DataFrame(
        coordinates.astype(np.float32), columns=PC_COLUMNS, index=profile.index
    )
    result[ARCHETYPE_CELL] = np.asarray(
        [f"a{value:02d}" for value in clusters], dtype=object
    )
    result["e82_archetype_distance"] = order[:, 0].astype(np.float32)
    result["e82_archetype_margin"] = (order[:, 1] - order[:, 0]).astype(np.float32)
    result["e82_archetype_unseen"] = np.zeros(len(result), dtype=np.int8)
    return result


def build_archetype_features(
    frame: pd.DataFrame,
    profiles_before: dict[int, pd.DataFrame],
    basis: ArchetypeBasis,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = pd.DataFrame(index=frame.index)
    for name in PC_COLUMNS:
        result[name] = np.full(len(frame), np.nan, dtype=np.float32)
    result[ARCHETYPE_CELL] = np.full(len(frame), "__unseen__", dtype=object)
    result["e82_archetype_distance"] = np.full(len(frame), np.nan, dtype=np.float32)
    result["e82_archetype_margin"] = np.full(len(frame), np.nan, dtype=np.float32)
    result["e82_archetype_unseen"] = np.ones(len(frame), dtype=np.int8)

    seasons = frame["season"].to_numpy(dtype=np.int16, copy=False)
    pitchers = frame["pitcher_id"].to_numpy(dtype=np.int64, copy=False)
    transformed_cache: dict[int, pd.DataFrame] = {}
    for season in sorted(set(int(value) for value in seasons)):
        profile = profiles_before.get(season)
        if profile is None or profile.empty:
            continue
        transformed = transform_profile_table(profile, basis)
        transformed_cache[season] = transformed
        mask = seasons == season
        row_positions = np.flatnonzero(mask)
        lookup = transformed.reindex(pitchers[mask])
        known = lookup[ARCHETYPE_CELL].notna().to_numpy(dtype=bool)
        if not np.any(known):
            continue
        selected_positions = row_positions[known]
        selected_index = frame.index[selected_positions]
        for name in (*PC_COLUMNS, "e82_archetype_distance", "e82_archetype_margin"):
            result.loc[selected_index, name] = lookup.loc[known, name].to_numpy()
        result.loc[selected_index, ARCHETYPE_CELL] = lookup.loc[
            known, ARCHETYPE_CELL
        ].astype(str).to_numpy()
        result.loc[selected_index, "e82_archetype_unseen"] = 0
    known_rows = result["e82_archetype_unseen"].eq(0)
    return result, {
        "source_year": 2019,
        "source_pitchers": basis.source_pitchers,
        "pca_components": COMPONENTS,
        "pca_explained_variance_ratio": basis.pca.explained_variance_ratio_.tolist(),
        "pca_explained_variance_total": float(
            basis.pca.explained_variance_ratio_.sum()
        ),
        "clusters": CLUSTERS,
        "known_rows": int(known_rows.sum()),
        "unseen_rows": int((~known_rows).sum()),
        "cutoff": "profile values use matched R TrackMan seasons strictly before row season",
        "basis_cutoff": "immutable scaler/PCA/KMeans fit on 2019 official TrackMan only",
        "target_free": True,
        "row_independent": True,
        "source_features": basis.source_feature_names,
    }
