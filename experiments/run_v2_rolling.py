#!/usr/bin/env python3
"""Unified leakage-safe rolling harness for the v2 plan.

One script covers every v2 experiment so the pipeline has a single entry point:

    model family   linear | hgb | lgbm | catboost
    feature set    base [+ e14] [+ platoon] [+ pitcher_te] [+ trackman]
    protocol       outer history season < Y, validation season == Y

All state assets (E14 season-end counters, platoon encoders, priors) are built
from the outer history only.  Training rows receive season-wise out-of-fold
encodings so the training matrix never sees its own target, and the validation
season receives the frozen full-history encoder.  Every derived feature is a
per-row lookup, so batch composition can never change a row's prediction.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import gc
import json
from math import gcd
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

# TabICLv2 imports PyTorch.  On Windows its CUDA DLLs must be loaded before
# NumPy/pandas/sklearn native runtimes or WinError 1114 can occur.  Keep the
# preload opt-in so all established CatBoost/LightGBM recipes retain their
# original import path.
if os.environ.get("V2_PRELOAD_TORCH", "0") == "1":
    import torch as _preloaded_torch  # noqa: F401

import numpy as np
try:
    # Import before pandas/sklearn and before the 400+ MiB training frame is
    # loaded.  On Windows, loading this DLL later intermittently raised an
    # access violation on the first categorical fit in LightGBM 4.7.0.
    from lightgbm import LGBMClassifier as _LGBMClassifier
except ImportError:  # LightGBM is optional for non-LGBM experiments.
    _LGBMClassifier = None
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MPL_CACHE = ROOT / "experiments" / "_cache" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

from experiments.run_baselines import (  # noqa: E402
    FEATURES as BASE_FEATURES,
    HGB_CATEGORICAL,
    HGB_DROPPED,
    LINEAR_CATEGORICAL,
    RANDOM_SEED,
    STRING_CATEGORICAL,
    TARGET,
    load_train,
)
from experiments.run_e14_rolling import (  # noqa: E402
    E14_K,
    SEASON,
    build_e14_features,
    metric,
    prior_before_each_season,
    season_end_state,
)
from experiments.run_e15_pseudo_forward import candidate_priors  # noqa: E402
from experiments.stats import aggregate_gate, paired_bootstrap_brier_ci  # noqa: E402

PITCHER = "pitcher_id"
BATTER_HAND = "batter_hand"

PLATOON_FEATURES = [
    "e30_pitcher_rate",
    "e30_platoon_delta",
    "e30_platoon_n_log",
    "e30_platoon_unseen",
]
PLATOON_K_PITCHER = 200.0
PLATOON_K_PLATOON = 200.0
PITCHER_TE_FEATURES = ["b2_pitcher_id"]
PITCHER_HAND_CELL = "c33_pitcher_batter_hand"
HAND_MATCHUP_CELL = "c36_pitcher_hand_batter_hand"
COUNT_STATE_CELL = "c37_count_state"
TYPE_COUNT_CELL = "c45_game_type_count"
TYPE_MONTH_CELL = "c38_game_type_month"
E14_RATE_HAND_BIN_CELL = "c42_e14_rate_hand_bin"
E14_N_HAND_BIN_CELL = "c43_e14_n_hand_bin"
TEAM_MATCHUP_CELL = "c44_pitcher_batter_team_type"
HOME_TEAM_CELL = "c46_home_team"
AWAY_TEAM_CELL = "c46_away_team"
VENUE_CELL = "c46_game_type_home_team"
PITCHER_ROLE_CELL = "c47_pitcher_role"
TRACKMAN_ARCHETYPE_CELL = "e82_trackman_archetype"
FINE_PITCH_TYPES = (
    "Fastball", "Slider", "Curveball", "ChangeUp",
    "Splitter", "Sinker", "Cutter", "Other",
)
PHYSICS_AUX_COLUMNS = (
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
    "extension", "rel_height", "rel_side", "zone_speed",
)
FINE_PITCH_PROBABILITY_COLUMNS = [
    f"e90_p_{label.lower()}" for label in FINE_PITCH_TYPES
]
FINE_PITCH_PROFILE_SPECS = (
    ("pitcher", ("pitcher_id",), 50.0),
    ("pitcher_count", ("pitcher_id", "balls_before", "strikes_before"), 40.0),
    ("pitcher_hand", ("pitcher_id", "batter_hand"), 40.0),
    (
        "pitcher_hand_count",
        ("pitcher_id", "batter_hand", "balls_before", "strikes_before"),
        60.0,
    ),
)
E14_MULTI_KS = (20.0, 50.0, 250.0, 500.0, 1000.0)
COMPONENT_RATE_COLUMNS = {
    "reverse": "asof_pitcher_reverse_rate",
    "middle": "asof_pitcher_middle_rate",
    "ball": "asof_pitcher_ball_rate",
    "strike": "asof_pitcher_strike_rate",
}
# Columns a gradient booster should treat as unordered categories.
BOOSTER_CATEGORICAL = [
    *STRING_CATEGORICAL,
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_hand",
    "batter_hand",
    "b2_pitcher_id",
    PITCHER_HAND_CELL,
    HAND_MATCHUP_CELL,
    COUNT_STATE_CELL,
    TYPE_COUNT_CELL,
    TYPE_MONTH_CELL,
    E14_RATE_HAND_BIN_CELL,
    E14_N_HAND_BIN_CELL,
    TEAM_MATCHUP_CELL,
    HOME_TEAM_CELL,
    AWAY_TEAM_CELL,
    VENUE_CELL,
    PITCHER_ROLE_CELL,
    TRACKMAN_ARCHETYPE_CELL,
]

MODEL_CHOICES = (
    "linear", "hgb", "lgbm", "lgbm_outcome", "xgboost", "catboost", "catboost_numeric",
    "catboost_outcome", "catboost_count_moe", "catboost_pitchtype_moe",
    "catboost_dense_pitchtype_moe", "catboost_dense_pitch_joint",
    "catboost_fine_pitch_joint",
    "catboost_physics_joint",
    "catboost_hier_pitch_joint",
    "catboost_auto_pitch_joint",
    "catboost_fine_pitch_moe",
    "catboost_fine_pitch_binary_moe",
    "catboost_dense_multitask",
    "tabm_dense_multitask",
    "tabm_pitch_gated",
    "catboost_component_pattern_moe",
    "catboost_game_centered_brier",
    "catboost_game_pairwise_rank",
    "catboost_state_residual", "catboost_group_soft",
    "catboost_brier", "catboost_multi_brier", "catboost_failure_decomp",
    "catboost_failure_chain",
    "catboost_teacher", "ebm",
    "tabicl",
    "deep_mlp", "deepfm", "tabtransformer",
    "deep_mlp_outcome", "deepfm_outcome", "tabtransformer_outcome",
    "tabm", "tabm_periodic", "tabm_piecewise",
    "tabm_outcome", "tabm_periodic_outcome", "tabm_piecewise_outcome",
    "realmlp", "realmlp_outcome", "tabr", "tabr_outcome",
    "realtabr", "realtabr_outcome",
    "catboost_leaf_refit",
)
OUTCOME_SCHEMES = (
    "five", "drop_overlap", "reverse_any", "middle_any", "coarse3",
    "success_count", "all_count", "success_type", "all_type", "binary_count",
    "reverse_count", "middle_count", "wide_count", "reverse_type",
    "reverse_hand", "all_hand", "success_call", "all_call",
    "component15", "component15_type", "component15_count",
)
FEATURE_CHOICES = (
    "base",
    "e14",
    "e14_multi",
    "platoon",
    "pitcher_te",
    "trackman",
    "trackman_rich",
    "trackman_stability",
    "trackman_group_stability",
    "trackman_game_repeatability",
    "trackman_inning_physics",
    "trackman_trend",
    "trackman_platoon",
    "trackman_count",
    "trackman_workload",
    "trackman_teacher",
    "trackman_lupi",
    "trackman_archetype",
    "trackman_batter_rich",
    "expanded_trackman_profiles",
    "e22_probs",
    "e22_cat",
    "fine_pitch_latent",
    "auto_pitch_latent",
    "auto_pitch_profile_latent",
    "expanded_auto_pitch_latent",
    "partial_expanded_auto_pitch_latent",
    "matchup_hand_auto_pitch_latent",
    "components",
    "reverse_component",
    "platoon_centered",
    "pitcher_hand_cat",
    "f_regime",
    "hand_matchup",
    "semantic_row",
    "count_state",
    "type_count",
    "type_month",
    "e14_hand_cells",
    "e14_count_cells",
    "e14_type_count_cells",
    "reverse_hand_cells",
    "fastball_hand_cells",
    "e14_rate_hand_bin",
    "e14_n_hand_bin",
    "team_matchup",
    "venue",
    "pitcher_profile",
    "batter_e14",
    "hierarchical_e14",
    "hierarchical_batter_e14",
    "recent_form",
    "recent_form_count_cells",
    "recent_denominators",
    "recent_workload_decoder",
    "batter_e14_count_cells",
    "pitcher_batter_season_interactions",
    "batter_middle_e14",
    "pitchmix_e14",
    "current_state_full",
    "current_state_context",
    "current_state_level",
    "outcome_context",
    "history_month_rate",
    "history_count_rate",
    "history_hand_rate",
    "history_inning_rate",
    "history_pitcher_count_rate",
    "history_pitcher_type_count_rate",
    "history_batter_count_rate",
    "history_pitcher_batterhand_count_rate",
    "history_batter_pitcherhand_count_rate",
    "temporal_stable_joint",
    "pitcher_context_profile",
    "batter_context_profile",
    "consistent_prior",
)

FEATURE_VIEWS = ("all", "application", "behavioral")

# A50 separates information that is known for a pitch application from
# player-history/behavioural state.  Keeping the split here (after feature
# assembly) makes the rolling evaluator use exactly the same cutoff-correct
# upstream feature builders while preventing either expert from silently
# seeing the other view.
APPLICATION_VIEW_FEATURES = {
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "base_state",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "c36_pitcher_hand_batter_hand",
    "c36_same_hand",
}

BEHAVIORAL_CONTEXT_FEATURES = {
    "season",
    "game_month",
    "game_type",
    "balls_before",
    "strikes_before",
    "pitcher_hand",
    "batter_hand",
}


def apply_feature_view(frame: pd.DataFrame, view: str) -> pd.DataFrame:
    """Select an A50 expert view without changing any feature values."""
    if view == "all":
        return frame
    if view == "application":
        selected = [column for column in frame if column in APPLICATION_VIEW_FEATURES]
    elif view == "behavioral":
        behavioral_prefixes = (
            "asof_", "e14_", "e30_", "e31_", "e49_", "e52_", "e53_",
            "e58_", "c39_", "c40_", "c41_", "c48_",
        )
        selected = [
            column for column in frame
            if column in BEHAVIORAL_CONTEXT_FEATURES
            or column in {"pitcher_id", "batter_id"}
            or column.startswith(behavioral_prefixes)
        ]
    else:  # Defensive guard for direct callers outside argparse.
        raise ValueError(f"Unknown feature view: {view}")
    if not selected:
        raise ValueError(f"Feature view '{view}' selected no columns")
    return frame[selected]

HISTORICAL_GROUP_RATE_SPECS: dict[str, tuple[str, ...]] = {
    "history_month_rate": ("game_type", "game_month"),
    "history_count_rate": ("game_type", "balls_before", "strikes_before"),
    "history_hand_rate": ("game_type", "pitcher_hand", "batter_hand"),
    "history_inning_rate": ("game_type", "inning"),
    # High-cardinality, completed-season-only state.  These expose persistent
    # entity tendencies in the exact pre-pitch context without using any row
    # from the validation/test season to construct a lookup table.
    "history_pitcher_count_rate": (
        "pitcher_id", "balls_before", "strikes_before",
    ),
    "history_pitcher_type_count_rate": (
        "pitcher_id", "game_type", "balls_before", "strikes_before",
    ),
    "history_batter_count_rate": (
        "batter_id", "balls_before", "strikes_before",
    ),
    "history_pitcher_batterhand_count_rate": (
        "pitcher_id", "batter_hand", "balls_before", "strikes_before",
    ),
    "history_batter_pitcherhand_count_rate": (
        "batter_id", "pitcher_hand", "balls_before", "strikes_before",
    ),
}

OUTCOME_CONTEXT_SPECS: dict[str, tuple[str, ...]] = {
    "pc": ("pitcher_id", "balls_before", "strikes_before"),
    "ph": ("pitcher_id", "batter_hand"),
    "pg": ("pitcher_id", "game_type"),
    "bc": ("batter_id", "balls_before", "strikes_before"),
    "bh": ("batter_id", "pitcher_hand"),
}


def candidate_priors_before_each_season(
    frame: pd.DataFrame, mode: str
) -> dict[int, float]:
    """Apply the validation prior recipe season-wise without looking ahead."""
    result: dict[int, float] = {}
    for current_season in sorted(int(value) for value in frame[SEASON].unique()):
        completed = frame.loc[frame[SEASON] < current_season]
        result[current_season] = (
            0.5
            if completed.empty
            else float(candidate_priors(completed, current_season)[mode])
        )
    return result


def build_pitcher_hand_category(frame: pd.DataFrame) -> pd.DataFrame:
    pitcher = pd.to_numeric(frame[PITCHER], errors="coerce").astype("Int64").astype("string")
    hand = pd.to_numeric(frame[BATTER_HAND], errors="coerce").astype("Int64").astype("string")
    value = pitcher.fillna("__missing__") + "|" + hand.fillna("__missing__")
    return pd.DataFrame({PITCHER_HAND_CELL: value}, index=frame.index)


def build_f_regime_feature(frame: pd.DataFrame) -> pd.DataFrame:
    post_break = frame["game_type"].eq("F") & frame[SEASON].ge(2023)
    return pd.DataFrame(
        {"d34_f_post_2023": post_break.to_numpy(dtype=np.int8)}, index=frame.index
    )


def build_hand_matchup_features(frame: pd.DataFrame) -> pd.DataFrame:
    pitcher = pd.to_numeric(frame["pitcher_hand"], errors="coerce").astype("Int64")
    batter = pd.to_numeric(frame[BATTER_HAND], errors="coerce").astype("Int64")
    cell = pitcher.astype("string").fillna("__missing__") + "|" + batter.astype(
        "string"
    ).fillna("__missing__")
    same = (pitcher == batter).fillna(False).astype(np.int8)
    return pd.DataFrame(
        {HAND_MATCHUP_CELL: cell, "c36_same_hand": same.to_numpy()}, index=frame.index
    )


def build_semantic_row_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose deterministic arithmetic relations among official row statistics.

    Every value is computed from the current row only.  The companion
    ``c36_same_hand`` feature already comes from ``hand_matchup`` in component C,
    so this builder adds the ten non-duplicate members of the fixed semantic
    bundle.
    """
    numeric_columns = (
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    )
    value = {
        column: pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype=np.float64, copy=False
        )
        for column in numeric_columns
    }

    career_success = value["asof_pitcher_success_rate"]
    failure_rate = np.maximum(1.0 - career_success, 1e-6)
    pitch_mix = np.column_stack(
        [
            value["asof_pitcher_fastball_rate"],
            value["asof_pitcher_breaking_rate"],
            value["asof_pitcher_offspeed_rate"],
        ]
    )
    pitch_mix = np.where(np.isfinite(pitch_mix), np.maximum(pitch_mix, 0.0), 0.0)
    pitch_mix_total = pitch_mix.sum(axis=1, keepdims=True)
    pitch_mix_probability = np.divide(
        pitch_mix,
        pitch_mix_total,
        out=np.zeros_like(pitch_mix),
        where=pitch_mix_total > 0.0,
    )
    pitch_mix_entropy = -np.sum(
        np.where(
            pitch_mix_probability > 0.0,
            pitch_mix_probability * np.log(np.maximum(pitch_mix_probability, 1e-12)),
            0.0,
        ),
        axis=1,
    )

    values = {
        "e80_prev5_success_minus_career": (
            value["asof_pitcher_prev5_game_success_rate"] - career_success
        ),
        "e80_reverse_share_of_failure": (
            value["asof_pitcher_reverse_rate"] / failure_rate
        ),
        "e80_ball_share_of_failure": (
            value["asof_pitcher_ball_rate"] / failure_rate
        ),
        "e80_pitchmix_entropy": pitch_mix_entropy,
        "e80_log_pitcher_n": np.log1p(
            np.maximum(value["asof_pitcher_n"], 0.0)
        ),
        "e80_success_prev1_minus_prev3": (
            value["asof_pitcher_prev1_game_success_rate"]
            - value["asof_pitcher_prev3_game_success_rate"]
        ),
        "e80_success_prev3_minus_prev5": (
            value["asof_pitcher_prev3_game_success_rate"]
            - value["asof_pitcher_prev5_game_success_rate"]
        ),
        "e80_middle_prev1_minus_career": (
            value["asof_pitcher_prev1_game_middle_rate"]
            - value["asof_pitcher_middle_rate"]
        ),
        "e80_middle_prev1_minus_prev3": (
            value["asof_pitcher_prev1_game_middle_rate"]
            - value["asof_pitcher_prev3_game_middle_rate"]
        ),
        "e80_ball_minus_strike": (
            value["asof_pitcher_ball_rate"] - value["asof_pitcher_strike_rate"]
        ),
    }
    return pd.DataFrame(
        {name: array.astype(np.float32) for name, array in values.items()},
        index=frame.index,
    )


def build_count_state_feature(frame: pd.DataFrame) -> pd.DataFrame:
    value = frame["balls_before"].astype(str) + "|" + frame["strikes_before"].astype(str)
    return pd.DataFrame({COUNT_STATE_CELL: value}, index=frame.index)


def build_type_count_feature(frame: pd.DataFrame) -> pd.DataFrame:
    value = (
        frame["game_type"].astype(str)
        + "|"
        + frame["balls_before"].astype(str)
        + "|"
        + frame["strikes_before"].astype(str)
    )
    return pd.DataFrame({TYPE_COUNT_CELL: value}, index=frame.index)


def build_type_month_feature(frame: pd.DataFrame) -> pd.DataFrame:
    value = frame["game_type"].astype(str) + "|" + frame["game_month"].astype(str)
    return pd.DataFrame({TYPE_MONTH_CELL: value}, index=frame.index)


def build_e14_hand_cell_features(
    frame: pd.DataFrame, e14: pd.DataFrame
) -> pd.DataFrame:
    pitcher = pd.to_numeric(frame["pitcher_hand"], errors="coerce").fillna(-1).to_numpy()
    batter = pd.to_numeric(frame[BATTER_HAND], errors="coerce").fillna(-1).to_numpy()
    rate = e14["e14_rate_season"].to_numpy(dtype=np.float32, copy=False)
    values = {}
    for pitcher_hand in (1, 2):
        for batter_hand in (1, 2):
            mask = (pitcher == pitcher_hand) & (batter == batter_hand)
            values[f"c39_e14_ph{pitcher_hand}_bh{batter_hand}"] = (
                rate * mask.astype(np.float32)
            )
    return pd.DataFrame(values, index=frame.index)


def build_e14_count_cell_features(
    frame: pd.DataFrame, e14: pd.DataFrame, include_game_type: bool
) -> pd.DataFrame:
    """Expose separate numeric E14 slopes for legal pre-pitch count cells."""
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(-1).to_numpy()
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(-1).to_numpy()
    rate = e14["e14_rate_season"].to_numpy(dtype=np.float32, copy=False)
    types = frame["game_type"].astype(str).to_numpy()
    values: dict[str, np.ndarray] = {}
    type_values = ("R", "F") if include_game_type else (None,)
    for game_type in type_values:
        for ball_count in range(4):
            for strike_count in range(3):
                mask = (balls == ball_count) & (strikes == strike_count)
                if game_type is not None:
                    mask &= types == game_type
                    prefix = f"c48_e14_{game_type.lower()}"
                else:
                    prefix = "c48_e14"
                values[f"{prefix}_b{ball_count}s{strike_count}"] = (
                    rate * mask.astype(np.float32)
                )
    return pd.DataFrame(values, index=frame.index)


def build_rate_count_cell_features(
    frame: pd.DataFrame, rate: pd.Series, prefix: str
) -> pd.DataFrame:
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(-1).to_numpy()
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(-1).to_numpy()
    rate_values = rate.to_numpy(dtype=np.float32, copy=False)
    values: dict[str, np.ndarray] = {}
    for ball_count in range(4):
        for strike_count in range(3):
            mask = (balls == ball_count) & (strikes == strike_count)
            values[f"{prefix}_b{ball_count}s{strike_count}"] = (
                rate_values * mask.astype(np.float32)
            )
    return pd.DataFrame(values, index=frame.index)


def build_recent_form_features(
    frame: pd.DataFrame,
    e14: pd.DataFrame | None,
    include_count_cells: bool,
) -> pd.DataFrame:
    """Expose within-row 1/3/5-game trends hidden by raw tree thresholds."""
    success = {
        horizon: pd.to_numeric(
            frame[f"asof_pitcher_prev{horizon}_game_success_rate"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        for horizon in (1, 3, 5)
    }
    middle = {
        horizon: pd.to_numeric(
            frame[f"asof_pitcher_prev{horizon}_game_middle_rate"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        for horizon in (1, 3, 5)
    }
    career_success = pd.to_numeric(
        frame["asof_pitcher_success_rate"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    career_middle = pd.to_numeric(
        frame["asof_pitcher_middle_rate"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    season_rate = (
        e14["e14_rate_season"].to_numpy(dtype=np.float64, copy=False)
        if e14 is not None else career_success
    )
    values: dict[str, np.ndarray] = {}
    for name, series, career in (
        ("success", success, career_success),
        ("middle", middle, career_middle),
    ):
        matrix = np.column_stack([series[1], series[3], series[5]])
        valid_count = np.sum(np.isfinite(matrix), axis=1)
        safe = np.where(np.isfinite(matrix), matrix, 0.0)
        mean = np.divide(
            safe.sum(axis=1), valid_count,
            out=np.nan_to_num(career, nan=0.5).copy(), where=valid_count > 0,
        )
        maximum = np.max(np.where(np.isfinite(matrix), matrix, -np.inf), axis=1)
        minimum = np.min(np.where(np.isfinite(matrix), matrix, np.inf), axis=1)
        maximum[~np.isfinite(maximum)] = mean[~np.isfinite(maximum)]
        minimum[~np.isfinite(minimum)] = mean[~np.isfinite(minimum)]
        weighted = np.nan_to_num(series[1], nan=mean) * 0.5
        weighted += np.nan_to_num(series[3], nan=mean) * 0.3
        weighted += np.nan_to_num(series[5], nan=mean) * 0.2
        values[f"e62_{name}_recent_weighted"] = weighted.astype(np.float32)
        values[f"e62_{name}_recent_mean"] = mean.astype(np.float32)
        values[f"e62_{name}_prev1_minus_prev3"] = np.nan_to_num(
            series[1] - series[3], nan=0.0
        ).astype(np.float32)
        values[f"e62_{name}_prev3_minus_prev5"] = np.nan_to_num(
            series[3] - series[5], nan=0.0
        ).astype(np.float32)
        values[f"e62_{name}_prev1_minus_prev5"] = np.nan_to_num(
            series[1] - series[5], nan=0.0
        ).astype(np.float32)
        values[f"e62_{name}_recent_range"] = (maximum - minimum).astype(np.float32)
        values[f"e62_{name}_weighted_minus_career"] = np.nan_to_num(
            weighted - career, nan=0.0
        ).astype(np.float32)
        values[f"e62_{name}_missing_count"] = (3 - valid_count).astype(np.int8)
    values["e62_success_weighted_minus_season"] = (
        values["e62_success_recent_weighted"].astype(np.float64) - season_rate
    ).astype(np.float32)
    values["e62_recent_success_plus_middle"] = (
        values["e62_success_recent_weighted"]
        + values["e62_middle_recent_weighted"]
    ).astype(np.float32)
    result = pd.DataFrame(values, index=frame.index)
    if include_count_cells:
        result = pd.concat([
            result,
            build_rate_count_cell_features(
                frame, result["e62_success_recent_weighted"], "e62_recent_success"
            ),
            build_rate_count_cell_features(
                frame, result["e62_success_weighted_minus_season"],
                "e62_recent_delta",
            ),
        ], axis=1)
    return result


def build_recent_denominator_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover recent-window workload lower bounds from paired official rates.

    Success and middle rates for a given 1/3/5-game window share the same
    hidden pitch-count denominator.  The LCM of their reduced denominators is
    therefore a row-local lower bound on that workload and is often the exact
    denominator.  Only values already present in the row are used; no ordering
    or aggregation over evaluation rows is performed.
    """

    horizon_max = {1: 200, 3: 500, 5: 900}
    tolerance = 5.1e-7

    def denominator_map(series: pd.Series, maximum: int) -> dict[float, tuple[int, float]]:
        mapping: dict[float, tuple[int, float]] = {}
        for raw in series.dropna().unique():
            value = float(raw)
            if value <= 0.0 or value >= 1.0:
                mapping[value] = (1, 0.0)
                continue
            fraction = Fraction(value).limit_denominator(maximum)
            error = abs(float(fraction) - value)
            mapping[value] = (
                int(fraction.denominator) if error <= tolerance else 0,
                float(error),
            )
        return mapping

    values: dict[str, np.ndarray] = {}
    recovered_n: dict[int, np.ndarray] = {}
    recovered_success: dict[int, np.ndarray] = {}
    recovered_middle: dict[int, np.ndarray] = {}
    boundary: dict[int, np.ndarray] = {}
    for horizon in (1, 3, 5):
        success_column = f"asof_pitcher_prev{horizon}_game_success_rate"
        middle_column = f"asof_pitcher_prev{horizon}_game_middle_rate"
        success = pd.to_numeric(frame[success_column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        middle = pd.to_numeric(frame[middle_column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        success_mapping = denominator_map(frame[success_column], horizon_max[horizon])
        middle_mapping = denominator_map(frame[middle_column], horizon_max[horizon])
        success_den = np.asarray(
            [success_mapping.get(float(value), (0, np.nan))[0] if np.isfinite(value) else 0
             for value in success],
            dtype=np.int32,
        )
        middle_den = np.asarray(
            [middle_mapping.get(float(value), (0, np.nan))[0] if np.isfinite(value) else 0
             for value in middle],
            dtype=np.int32,
        )
        common = np.asarray(
            [
                int(left // gcd(int(left), int(right)) * right)
                if left > 0 and right > 0 else 0
                for left, right in zip(success_den, middle_den)
            ],
            dtype=np.int32,
        )
        valid = (common > 0) & (common <= horizon_max[horizon])
        common = np.where(valid, common, 0).astype(np.int32)
        success_count = np.where(
            valid, np.rint(np.nan_to_num(success, nan=0.0) * common), 0
        ).astype(np.int32)
        middle_count = np.where(
            valid, np.rint(np.nan_to_num(middle, nan=0.0) * common), 0
        ).astype(np.int32)
        ambiguous = valid & (
            np.isin(np.nan_to_num(success, nan=-1.0), (0.0, 1.0))
            | np.isin(np.nan_to_num(middle, nan=-1.0), (0.0, 1.0))
        )
        recovered_n[horizon] = common
        recovered_success[horizon] = success_count
        recovered_middle[horizon] = middle_count
        boundary[horizon] = ambiguous
        values[f"e74_prev{horizon}_n_lower"] = common
        values[f"e74_prev{horizon}_log_n_lower"] = np.log1p(common).astype(np.float32)
        values[f"e74_prev{horizon}_n_per_game_lower"] = (
            common / float(horizon)
        ).astype(np.float32)
        values[f"e74_prev{horizon}_success_count_lower"] = success_count
        values[f"e74_prev{horizon}_middle_count_lower"] = middle_count
        values[f"e74_prev{horizon}_decoded"] = valid.astype(np.int8)
        values[f"e74_prev{horizon}_boundary_ambiguous"] = ambiguous.astype(np.int8)
        career = pd.to_numeric(
            frame["asof_pitcher_success_rate"], errors="coerce"
        ).fillna(0.5).to_numpy(dtype=np.float64)
        posterior = (success_count + 20.0 * career) / (common + 20.0)
        values[f"e74_prev{horizon}_success_posterior_k20"] = posterior.astype(
            np.float32
        )
        values[f"e74_prev{horizon}_success_reliability_k20"] = (
            common / (common + 20.0)
        ).astype(np.float32)

    nested = (
        (recovered_n[1] > 0)
        & (recovered_n[1] <= recovered_n[3])
        & (recovered_n[3] <= recovered_n[5])
    )
    n_23 = np.where(nested, recovered_n[3] - recovered_n[1], 0)
    n_45 = np.where(nested, recovered_n[5] - recovered_n[3], 0)
    success_23 = np.where(
        nested, recovered_success[3] - recovered_success[1], 0
    )
    success_45 = np.where(
        nested, recovered_success[5] - recovered_success[3], 0
    )
    middle_23 = np.where(
        nested, recovered_middle[3] - recovered_middle[1], 0
    )
    middle_45 = np.where(
        nested, recovered_middle[5] - recovered_middle[3], 0
    )
    component_valid_23 = nested & (success_23 >= 0) & (middle_23 >= 0)
    component_valid_45 = nested & (success_45 >= 0) & (middle_45 >= 0)

    def rate(count: np.ndarray, n: np.ndarray, valid: np.ndarray) -> np.ndarray:
        return np.divide(
            count,
            n,
            out=np.full(len(frame), np.nan, dtype=np.float64),
            where=valid & (n > 0),
        ).astype(np.float32)

    success_rate_23 = rate(success_23, n_23, component_valid_23)
    success_rate_45 = rate(success_45, n_45, component_valid_45)
    middle_rate_23 = rate(middle_23, n_23, component_valid_23)
    middle_rate_45 = rate(middle_45, n_45, component_valid_45)
    prev1_success = pd.to_numeric(
        frame["asof_pitcher_prev1_game_success_rate"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    prev1_middle = pd.to_numeric(
        frame["asof_pitcher_prev1_game_middle_rate"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    values.update(
        {
            "e74_nested_denominators": nested.astype(np.int8),
            "e74_prev23_n_lower": n_23.astype(np.int32),
            "e74_prev45_n_lower": n_45.astype(np.int32),
            "e74_prev23_log_n_lower": np.log1p(n_23).astype(np.float32),
            "e74_prev45_log_n_lower": np.log1p(n_45).astype(np.float32),
            "e74_prev23_success_rate": success_rate_23,
            "e74_prev45_success_rate": success_rate_45,
            "e74_prev23_middle_rate": middle_rate_23,
            "e74_prev45_middle_rate": middle_rate_45,
            "e74_prev23_component_valid": component_valid_23.astype(np.int8),
            "e74_prev45_component_valid": component_valid_45.astype(np.int8),
            "e74_success_prev1_minus_prev23": (
                prev1_success - success_rate_23
            ).astype(np.float32),
            "e74_success_prev23_minus_prev45": (
                success_rate_23 - success_rate_45
            ).astype(np.float32),
            "e74_middle_prev1_minus_prev23": (
                prev1_middle - middle_rate_23
            ).astype(np.float32),
            "e74_middle_prev23_minus_prev45": (
                middle_rate_23 - middle_rate_45
            ).astype(np.float32),
            "e74_workload_prev1_minus_prev23_per_game": (
                recovered_n[1] - n_23 / 2.0
            ).astype(np.float32),
            "e74_workload_prev23_minus_prev45_per_game": (
                n_23 / 2.0 - n_45 / 2.0
            ).astype(np.float32),
        }
    )
    return pd.DataFrame(values, index=frame.index)


def build_pitcher_batter_season_interactions(
    pitcher_e14: pd.DataFrame, batter_e14: pd.DataFrame
) -> pd.DataFrame:
    pitcher_rate = pitcher_e14["e14_rate_season"].to_numpy(
        dtype=np.float32, copy=False
    )
    batter_rate = batter_e14["e49_batter_rate_season"].to_numpy(
        dtype=np.float32, copy=False
    )
    pitcher_n = pitcher_e14["e14_n_season"].to_numpy(dtype=np.float64, copy=False)
    batter_n = batter_e14["e49_batter_n_season"].to_numpy(dtype=np.float64, copy=False)
    total_n = pitcher_n + batter_n
    weighted = np.divide(
        pitcher_rate * pitcher_n + batter_rate * batter_n,
        total_n,
        out=((pitcher_rate + batter_rate) / 2.0).astype(np.float64),
        where=total_n > 0,
    )
    return pd.DataFrame(
        {
            "e51_rate_mean": ((pitcher_rate + batter_rate) / 2.0).astype(np.float32),
            "e51_rate_diff": (pitcher_rate - batter_rate).astype(np.float32),
            "e51_rate_product": (pitcher_rate * batter_rate).astype(np.float32),
            "e51_rate_min": np.minimum(pitcher_rate, batter_rate).astype(np.float32),
            "e51_rate_max": np.maximum(pitcher_rate, batter_rate).astype(np.float32),
            "e51_rate_n_weighted": weighted.astype(np.float32),
        },
        index=pitcher_e14.index,
    )


def build_rate_hand_cell_features(
    frame: pd.DataFrame, source_column: str, prefix: str
) -> pd.DataFrame:
    pitcher = pd.to_numeric(frame["pitcher_hand"], errors="coerce").fillna(-1).to_numpy()
    batter = pd.to_numeric(frame[BATTER_HAND], errors="coerce").fillna(-1).to_numpy()
    rate = pd.to_numeric(frame[source_column], errors="coerce").fillna(0.0).to_numpy(
        dtype=np.float32
    )
    values = {}
    for pitcher_hand in (1, 2):
        for batter_hand in (1, 2):
            mask = (pitcher == pitcher_hand) & (batter == batter_hand)
            values[f"{prefix}_ph{pitcher_hand}_bh{batter_hand}"] = (
                rate * mask.astype(np.float32)
            )
    return pd.DataFrame(values, index=frame.index)


def build_e14_hand_bin_features(
    frame: pd.DataFrame,
    e14: pd.DataFrame,
    include_rate: bool,
    include_n: bool,
) -> pd.DataFrame:
    hand = frame["pitcher_hand"].astype(str) + "|" + frame[BATTER_HAND].astype(str)
    values: dict[str, pd.Series] = {}
    if include_rate:
        rate_bin = pd.cut(
            e14["e14_rate_season"],
            bins=[-np.inf, 0.35, 0.40, 0.45, 0.475, 0.50, 0.525, 0.55, 0.60, 0.65, np.inf],
            labels=False,
        )
        values[E14_RATE_HAND_BIN_CELL] = hand + "|" + rate_bin.fillna(-1).astype(int).astype(str)
    if include_n:
        n_bin = pd.cut(
            e14["e14_n_season"],
            bins=[-1, 10, 25, 50, 100, 250, 500, 1000, 2000, np.inf],
            labels=False,
        )
        values[E14_N_HAND_BIN_CELL] = hand + "|" + n_bin.fillna(-1).astype(int).astype(str)
    return pd.DataFrame(values, index=frame.index)


def build_team_matchup_feature(frame: pd.DataFrame) -> pd.DataFrame:
    value = (
        frame["pitcher_team_id"].astype(str)
        + "|"
        + frame["batter_team_id"].astype(str)
        + "|"
        + frame["game_type"].astype(str)
    )
    return pd.DataFrame({TEAM_MATCHUP_CELL: value}, index=frame.index)


def build_venue_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive home/away team proxies from fields present in the current row."""
    pitcher_home = frame["top_bottom"].astype(str).eq("T")
    pitcher_team = pd.to_numeric(frame["pitcher_team_id"], errors="coerce").astype(
        "Int64"
    )
    batter_team = pd.to_numeric(frame["batter_team_id"], errors="coerce").astype(
        "Int64"
    )
    home = pitcher_team.where(pitcher_home, batter_team).astype("string").fillna(
        "__missing__"
    )
    away = batter_team.where(pitcher_home, pitcher_team).astype("string").fillna(
        "__missing__"
    )
    venue = frame["game_type"].astype("string").fillna("__missing__") + "|" + home
    return pd.DataFrame(
        {
            HOME_TEAM_CELL: home,
            AWAY_TEAM_CELL: away,
            VENUE_CELL: venue,
            "c46_pitcher_home": pitcher_home.to_numpy(dtype=np.int8),
        },
        index=frame.index,
    )


def build_pitcher_profile_state(frame: pd.DataFrame) -> dict[str, Any]:
    """Build target-free pitcher usage summaries from completed history."""
    if frame.empty:
        return {
            "table": pd.DataFrame(),
            "global": {
                "mean_inning": 5.0,
                "early_rate": 1.0 / 3.0,
                "late_rate": 1.0 / 3.0,
                "first_rate": 0.1,
            },
        }
    work = pd.DataFrame(
        {
            PITCHER: frame[PITCHER].to_numpy(),
            "inning": pd.to_numeric(frame["inning"], errors="coerce")
            .fillna(5.0)
            .to_numpy(dtype=np.float64),
        },
        index=frame.index,
    )
    work["early"] = work["inning"].le(3).astype(np.int8)
    work["late"] = work["inning"].ge(7).astype(np.int8)
    work["first"] = work["inning"].eq(1).astype(np.int8)
    table = work.groupby(PITCHER, sort=False, observed=True).agg(
        n=("inning", "size"),
        inning_sum=("inning", "sum"),
        early_sum=("early", "sum"),
        late_sum=("late", "sum"),
        first_sum=("first", "sum"),
    )
    return {
        "table": table,
        "global": {
            "mean_inning": float(work["inning"].mean()),
            "early_rate": float(work["early"].mean()),
            "late_rate": float(work["late"].mean()),
            "first_rate": float(work["first"].mean()),
        },
    }


def pitcher_profile_states_before_each_season(
    frame: pd.DataFrame,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Create season-wise OOF profile states plus the final frozen state."""
    states: dict[int, dict[str, Any]] = {}
    for current_season in sorted(int(value) for value in frame[SEASON].unique()):
        states[current_season] = build_pitcher_profile_state(
            frame.loc[frame[SEASON] < current_season]
        )
    return states, build_pitcher_profile_state(frame)


def apply_pitcher_profile_features(
    frame: pd.DataFrame, state: dict[str, Any], k: float
) -> pd.DataFrame:
    if k <= 0:
        raise ValueError("--pitcher-profile-k must be positive")
    table: pd.DataFrame = state["table"]
    global_values: dict[str, float] = state["global"]
    pitcher = frame[PITCHER]
    if table.empty:
        n = np.zeros(len(frame), dtype=np.float64)
        inning_sum = early_sum = late_sum = first_sum = n.copy()
    else:
        n = pitcher.map(table["n"]).fillna(0.0).to_numpy(dtype=np.float64)
        inning_sum = pitcher.map(table["inning_sum"]).fillna(0.0).to_numpy(
            dtype=np.float64
        )
        early_sum = pitcher.map(table["early_sum"]).fillna(0.0).to_numpy(
            dtype=np.float64
        )
        late_sum = pitcher.map(table["late_sum"]).fillna(0.0).to_numpy(
            dtype=np.float64
        )
        first_sum = pitcher.map(table["first_sum"]).fillna(0.0).to_numpy(
            dtype=np.float64
        )
    denominator = n + k
    mean_inning = (inning_sum + k * global_values["mean_inning"]) / denominator
    early_rate = (early_sum + k * global_values["early_rate"]) / denominator
    late_rate = (late_sum + k * global_values["late_rate"]) / denominator
    first_rate = (first_sum + k * global_values["first_rate"]) / denominator
    role = np.where(
        n <= 0,
        "unseen",
        np.where(first_rate >= 0.08, "starter", np.where(first_rate >= 0.02, "swing", "reliever")),
    )
    return pd.DataFrame(
        {
            "c47_profile_n_log": np.log1p(n).astype(np.float32),
            "c47_profile_mean_inning": mean_inning.astype(np.float32),
            "c47_profile_early_rate": early_rate.astype(np.float32),
            "c47_profile_late_rate": late_rate.astype(np.float32),
            "c47_profile_first_rate": first_rate.astype(np.float32),
            "c47_profile_unseen": (n <= 0).astype(np.int8),
            PITCHER_ROLE_CELL: pd.Series(role, index=frame.index, dtype="string"),
        },
        index=frame.index,
    )


def build_pitcher_profile_frame(
    frame: pd.DataFrame,
    states_before: dict[int, dict[str, Any]],
    fallback: dict[str, Any],
    k: float,
) -> pd.DataFrame:
    parts = []
    for current_season, block in frame.groupby(SEASON, sort=True, observed=True):
        parts.append(
            apply_pitcher_profile_features(
                block, states_before.get(int(current_season), fallback), k
            )
        )
    return pd.concat(parts).reindex(frame.index)


def entity_season_end_state(
    frame: pd.DataFrame,
    entity_column: str,
    n_column: str,
    rate_column: str,
) -> tuple[dict[int, dict[int, tuple[int, int]]], dict[int, tuple[int, int]]]:
    """Freeze cumulative target counters for an arbitrary official entity axis."""
    before: dict[int, dict[int, tuple[int, int]]] = {}
    state: dict[int, tuple[int, int]] = {}
    for current_season in sorted(int(value) for value in frame[SEASON].unique()):
        before[current_season] = dict(state)
        last_rows = frame.loc[frame[SEASON] == current_season].groupby(
            entity_column, sort=False, observed=True
        ).tail(1)
        for row in last_rows.itertuples(index=False):
            n = int(getattr(row, n_column) or 0)
            rate = getattr(row, rate_column)
            success_before = int(np.rint((0.0 if pd.isna(rate) else float(rate)) * n))
            state[int(getattr(row, entity_column))] = (
                n + 1,
                success_before + int(getattr(row, TARGET)),
            )
    return before, state


def build_entity_season_features(
    frame: pd.DataFrame,
    states_before: dict[int, dict[int, tuple[int, int]]],
    prior_by_season: dict[int, float],
    validation_prior: float,
    entity_column: str,
    n_column: str,
    rate_column: str,
    prefix: str,
    k: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if k <= 0:
        raise ValueError("entity season smoothing k must be positive")
    n_end = np.zeros(len(frame), dtype=np.int64)
    s_end = np.zeros(len(frame), dtype=np.int64)
    unseen = np.zeros(len(frame), dtype=np.int8)
    seasons = frame[SEASON].to_numpy(dtype=np.int16, copy=False)
    entities = pd.to_numeric(frame[entity_column], errors="coerce").fillna(-1).to_numpy(
        dtype=np.int64
    )
    for index, (current_season, entity) in enumerate(zip(seasons, entities)):
        state = states_before.get(int(current_season), {}).get(int(entity))
        if state is None:
            unseen[index] = 1
        else:
            n_end[index], s_end[index] = state
    n_asof = pd.to_numeric(frame[n_column], errors="coerce").fillna(0).to_numpy(
        dtype=np.int64
    )
    career_rate = pd.to_numeric(frame[rate_column], errors="coerce").fillna(
        validation_prior
    ).to_numpy(dtype=np.float64)
    s_asof = np.rint(career_rate * n_asof).astype(np.int64)
    n_delta = n_asof - n_end
    s_delta = s_asof - s_end
    invalid = (n_delta < 0) | (s_delta < 0) | (s_delta > n_delta)
    safe_n = np.where(invalid, 0, n_delta)
    safe_s = np.where(invalid, 0, s_delta)
    row_prior = np.asarray(
        [prior_by_season.get(int(value), validation_prior) for value in seasons],
        dtype=np.float64,
    )
    rate = (safe_s + k * row_prior) / (safe_n + k)
    result = pd.DataFrame(
        {
            f"{prefix}_n_season": safe_n.astype(np.int32),
            f"{prefix}_s_season": safe_s.astype(np.int32),
            f"{prefix}_log_n_season": np.log1p(safe_n).astype(np.float32),
            f"{prefix}_rate_season": rate.astype(np.float32),
            f"{prefix}_rate_delta": (rate - career_rate).astype(np.float32),
            f"{prefix}_n_season_zero": (safe_n == 0).astype(np.int8),
            f"{prefix}_unseen": unseen,
            f"{prefix}_counter_invalid": invalid.astype(np.int8),
        },
        index=frame.index,
    )
    return result, {
        "k": float(k),
        "invalid_rows": int(invalid.sum()),
        "unseen_rows": int(unseen.sum()),
        "n_positive_rate": float(np.mean(safe_n > 0)),
        "n_median": float(np.median(safe_n)),
    }


def build_hierarchical_entity_features(
    frame: pd.DataFrame,
    states_before: dict[int, dict[int, tuple[int, int]]],
    prior_by_season: dict[int, float],
    validation_prior: float,
    entity_column: str,
    n_column: str,
    rate_column: str,
    prefix: str,
    history_k: float = 200.0,
    current_ks: tuple[float, ...] = (20.0, 50.0, 100.0, 200.0),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Shrink season-to-date form toward the entity's frozen prior history.

    E14 shrinks every player toward the league prior.  This A10 arm instead
    builds a two-level posterior: completed-history ability is first shrunk to
    the league, then current-season evidence is shrunk to that player-specific
    ability.  Every state is frozen before the row's season.
    """
    if history_k <= 0 or any(value <= 0 for value in current_ks):
        raise ValueError("hierarchical shrinkage constants must be positive")
    seasons = frame[SEASON].to_numpy(dtype=np.int16, copy=False)
    entities = pd.to_numeric(frame[entity_column], errors="coerce").fillna(-1).to_numpy(
        dtype=np.int64
    )
    n_end = np.zeros(len(frame), dtype=np.int64)
    s_end = np.zeros(len(frame), dtype=np.int64)
    unseen = np.ones(len(frame), dtype=np.int8)
    for index, (current_season, entity) in enumerate(zip(seasons, entities)):
        state = states_before.get(int(current_season), {}).get(int(entity))
        if state is not None:
            n_end[index], s_end[index] = state
            unseen[index] = 0
    row_prior = np.asarray([
        prior_by_season.get(int(current_season), validation_prior)
        for current_season in seasons
    ], dtype=np.float64)
    history_rate = (s_end + history_k * row_prior) / (n_end + history_k)
    n_asof = pd.to_numeric(frame[n_column], errors="coerce").fillna(0).to_numpy(
        dtype=np.int64
    )
    career_rate = pd.to_numeric(frame[rate_column], errors="coerce").fillna(
        validation_prior
    ).to_numpy(dtype=np.float64)
    s_asof = np.rint(career_rate * n_asof).astype(np.int64)
    n_delta = n_asof - n_end
    s_delta = s_asof - s_end
    invalid = (n_delta < 0) | (s_delta < 0) | (s_delta > n_delta)
    safe_n = np.where(invalid, 0, n_delta)
    safe_s = np.where(invalid, 0, s_delta)
    raw_season = np.divide(
        safe_s, safe_n, out=history_rate.copy(), where=safe_n > 0
    )
    values: dict[str, np.ndarray] = {
        f"{prefix}_history_rate": history_rate.astype(np.float32),
        f"{prefix}_history_delta_league": (history_rate - row_prior).astype(np.float32),
        f"{prefix}_history_log_n": np.log1p(n_end).astype(np.float32),
        f"{prefix}_raw_season_rate": raw_season.astype(np.float32),
        f"{prefix}_raw_season_delta_history": (raw_season - history_rate).astype(
            np.float32
        ),
        f"{prefix}_unseen": unseen,
        f"{prefix}_invalid": invalid.astype(np.int8),
    }
    for current_k in current_ks:
        label = str(int(current_k))
        posterior = (safe_s + current_k * history_rate) / (safe_n + current_k)
        values[f"{prefix}_posterior_k{label}"] = posterior.astype(np.float32)
        values[f"{prefix}_posterior_delta_history_k{label}"] = (
            posterior - history_rate
        ).astype(np.float32)
        values[f"{prefix}_reliability_k{label}"] = (
            safe_n / (safe_n + current_k)
        ).astype(np.float32)
    return pd.DataFrame(values, index=frame.index), {
        "history_k": float(history_k),
        "current_ks": [float(value) for value in current_ks],
        "unseen_rows": int(unseen.sum()),
        "invalid_rows": int(invalid.sum()),
        "cutoff": "completed prior seasons only plus current row official as-of counters",
        "row_independent": True,
    }


def build_historical_group_rate_features(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    spec_names: list[str],
    k: float,
    window: int | None,
    fallback_prior: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build season-wise OOF target rates from completed prior seasons only."""
    if k <= 0:
        raise ValueError("--history-group-k must be positive")
    if window is not None and window < 1:
        raise ValueError("--history-group-window must be >= 1")
    unknown = sorted(set(spec_names) - set(HISTORICAL_GROUP_RATE_SPECS))
    if unknown:
        raise ValueError(f"Unknown historical group-rate specs: {unknown}")

    result_parts: list[pd.DataFrame] = []
    metadata: dict[str, Any] = {}
    for spec_name in spec_names:
        columns = list(HISTORICAL_GROUP_RATE_SPECS[spec_name])
        prefix = {
            "history_month_rate": "e54_month",
            "history_count_rate": "e55_count",
            "history_hand_rate": "e56_hand",
            "history_inning_rate": "e57_inning",
            "history_pitcher_count_rate": "e63_pitcher_count",
            "history_pitcher_type_count_rate": "e64_pitcher_type_count",
            "history_batter_count_rate": "e65_batter_count",
            "history_pitcher_batterhand_count_rate": "e66_pitcher_batterhand_count",
            "history_batter_pitcherhand_count_rate": "e67_batter_pitcherhand_count",
        }[spec_name]
        spec_parts: list[pd.DataFrame] = []
        state_sizes: dict[str, int] = {}
        for current_season, block in frame.groupby(SEASON, sort=True, observed=True):
            current_season = int(current_season)
            completed = source.loc[source[SEASON] < current_season]
            if window is not None:
                completed = completed.loc[completed[SEASON] >= current_season - window]
            prior = (
                float(completed[TARGET].mean())
                if not completed.empty
                else float(fallback_prior)
            )
            if completed.empty:
                count = np.zeros(len(block), dtype=np.float64)
                success = np.zeros(len(block), dtype=np.float64)
                state_sizes[str(current_season)] = 0
            else:
                table = completed.groupby(columns, sort=False, observed=True)[TARGET].agg(
                    ["sum", "count"]
                )
                state_sizes[str(current_season)] = int(len(table))
                if len(columns) == 1:
                    key = block[columns[0]]
                    count = key.map(table["count"]).fillna(0.0).to_numpy(
                        dtype=np.float64
                    )
                    success = key.map(table["sum"]).fillna(0.0).to_numpy(
                        dtype=np.float64
                    )
                else:
                    key = pd.MultiIndex.from_frame(block[columns])
                    aligned = table.reindex(key)
                    count = aligned["count"].fillna(0.0).to_numpy(dtype=np.float64)
                    success = aligned["sum"].fillna(0.0).to_numpy(dtype=np.float64)
            rate = (success + k * prior) / (count + k)
            spec_parts.append(
                pd.DataFrame(
                    {
                        f"{prefix}_rate": rate.astype(np.float32),
                        f"{prefix}_delta": (rate - prior).astype(np.float32),
                        f"{prefix}_n_log": np.log1p(count).astype(np.float32),
                        f"{prefix}_unseen": (count <= 0).astype(np.int8),
                    },
                    index=block.index,
                )
            )
        result_parts.append(pd.concat(spec_parts).reindex(frame.index))
        metadata[spec_name] = {
            "columns": columns,
            "state_sizes": state_sizes,
        }
    return pd.concat(result_parts, axis=1), {
        "enabled": spec_names,
        "k": float(k),
        "window": window,
        "specs": metadata,
        "protocol": "season-wise OOF; only completed prior seasons",
    }


def build_temporal_stable_joint_features(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    *,
    k_pitcher: float = 100.0,
    k_hand: float = 38.0,
    k_pressure: float = 30.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the preregistered V18-style stable conditional state.

    Training rows use their own season's regular-season table with the row's
    target removed at every hierarchy level.  A future validation season uses
    the latest completed regular-season table unchanged.  The lookup is
    therefore row independent and never consumes a validation-season label or
    statistic.
    """
    strengths = {
        "pitcher": float(k_pitcher),
        "hand": float(k_hand),
        "pressure_hand": float(k_pressure),
    }
    if any(value <= 0.0 for value in strengths.values()):
        raise ValueError("temporal-stable hierarchy strengths must be positive")

    def pressure_state(rows: pd.DataFrame) -> np.ndarray:
        balls = rows["balls_before"].to_numpy(dtype=np.int8, copy=False)
        strikes = rows["strikes_before"].to_numpy(dtype=np.int8, copy=False)
        state = np.zeros(len(rows), dtype=np.int8)
        state[(balls == 3) & (strikes < 2)] = 1
        state[(balls < 3) & (strikes == 2)] = 2
        state[(balls == 3) & (strikes == 2)] = 3
        return state

    def lookup(
        table: pd.DataFrame,
        rows: pd.DataFrame,
        keys: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        grouped = table.groupby(keys, sort=False, observed=True)[TARGET].agg(
            ["sum", "size"]
        )
        if len(keys) == 1:
            index = pd.Index(rows[keys[0]].to_numpy(), name=keys[0])
        else:
            index = pd.MultiIndex.from_frame(rows[keys])
        aligned = grouped.reindex(index)
        return (
            aligned["sum"].fillna(0.0).to_numpy(dtype=np.float64),
            aligned["size"].fillna(0.0).to_numpy(dtype=np.float64),
        )

    def posterior_features(
        rate: np.ndarray,
        parent: np.ndarray,
        count: np.ndarray,
        prefix: str,
        strength: float,
    ) -> dict[str, np.ndarray]:
        reliability = np.clip(count / (count + strength), 0.0, 1.0)
        delta = rate - parent
        return {
            f"e96_{prefix}_rate": rate.astype(np.float32),
            f"e96_{prefix}_delta": delta.astype(np.float32),
            f"e96_{prefix}_reliability": reliability.astype(np.float32),
            f"e96_{prefix}_log_n": np.log1p(np.maximum(count, 0.0)).astype(
                np.float32
            ),
            f"e96_{prefix}_trusted_delta": (reliability * delta).astype(
                np.float32
            ),
            f"e96_{prefix}_post_sd": np.sqrt(
                np.clip(
                    rate * (1.0 - rate) / (count + strength + 1.0),
                    0.0,
                    None,
                )
            ).astype(np.float32),
        }

    source_r = source.loc[source["game_type"].eq("R")].copy()
    source_r["_e96_pressure"] = pressure_state(source_r)
    parts: list[pd.DataFrame] = []
    fold_metadata: dict[str, Any] = {}
    for current_season, block_original in frame.groupby(
        SEASON, sort=True, observed=True
    ):
        current_season = int(current_season)
        block = block_original.copy()
        block["_e96_pressure"] = pressure_state(block)

        same_season = source_r.loc[source_r[SEASON].eq(current_season)]
        if not same_season.empty:
            table = same_season
            source_season = current_season
            self_mask = (
                block.index.isin(table.index)
                & block["game_type"].eq("R").to_numpy(dtype=bool)
            )
        else:
            completed = source_r.loc[source_r[SEASON].lt(current_season)]
            if completed.empty:
                table = completed
                source_season = None
            else:
                source_season = int(completed[SEASON].max())
                table = completed.loc[completed[SEASON].eq(source_season)]
            self_mask = np.zeros(len(block), dtype=bool)

        if table.empty:
            fallback = (
                float(source_r[TARGET].mean()) if not source_r.empty else 0.5
            )
            league = np.full(len(block), fallback, dtype=np.float64)
            own_target = np.zeros(len(block), dtype=np.float64)
            table_rows = 0
        else:
            own_target = block[TARGET].to_numpy(dtype=np.float64, copy=False)
            total_success = float(table[TARGET].sum())
            total_count = float(len(table))
            league = np.full(len(block), total_success / total_count, dtype=np.float64)
            if self_mask.any():
                league[self_mask] = (
                    total_success - own_target[self_mask]
                ) / max(total_count - 1.0, 1.0)
            table_rows = int(total_count)

        def level(
            keys: list[str], parent: np.ndarray, strength: float
        ) -> tuple[np.ndarray, np.ndarray]:
            if table.empty:
                count = np.zeros(len(block), dtype=np.float64)
                success = np.zeros(len(block), dtype=np.float64)
            else:
                success, count = lookup(table, block, keys)
                if self_mask.any():
                    success[self_mask] -= own_target[self_mask]
                    count[self_mask] = np.maximum(0.0, count[self_mask] - 1.0)
            rate = (success + strength * parent) / (count + strength)
            return rate, count

        pitcher_rate, pitcher_n = level(
            [PITCHER], league, strengths["pitcher"]
        )
        hand_rate, hand_n = level(
            [PITCHER, BATTER_HAND], pitcher_rate, strengths["hand"]
        )
        pressure_rate, pressure_n = level(
            [PITCHER, "_e96_pressure", BATTER_HAND],
            hand_rate,
            strengths["pressure_hand"],
        )
        values = {
            **posterior_features(
                pitcher_rate, league, pitcher_n, "pitcher", strengths["pitcher"]
            ),
            **posterior_features(
                hand_rate, pitcher_rate, hand_n, "hand", strengths["hand"]
            ),
            **posterior_features(
                pressure_rate,
                hand_rate,
                pressure_n,
                "pressure_hand",
                strengths["pressure_hand"],
            ),
        }
        part = pd.DataFrame(values, index=block.index)
        if part.shape[1] != 18 or not np.isfinite(part.to_numpy()).all():
            raise ValueError("invalid temporal-stable joint feature matrix")
        parts.append(part)
        fold_metadata[str(current_season)] = {
            "source_season": source_season,
            "source_R_rows": table_rows,
            "self_excluded_rows": int(self_mask.sum()),
        }

    result = pd.concat(parts).reindex(frame.index)
    return result, {
        "enabled": True,
        "strengths": strengths,
        "source_scope": "R",
        "pressure_definition": "normal/three_ball/two_strike/full",
        "training_encoding": "same-season leave-one-out",
        "validation_encoding": "latest completed season",
        "feature_count": int(result.shape[1]),
        "feature_columns": list(result.columns),
        "seasons": fold_metadata,
        "row_independent_validation": True,
    }


def build_outcome_context_features(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    component15_labels: pd.Series,
    k: float = 200.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build prior-season success and detailed-outcome context deviations.

    For a row in season S, every lookup table is built exclusively from season
    S-1. The reverse/middle/ball/strike labels are reconstructed only for
    historical source rows from their next same-pitcher as-of counter delta.
    The current row contributes keys only, keeping validation/test inference
    independent of batch composition and row order.
    """
    if k <= 0:
        raise ValueError("outcome-context smoothing k must be positive")
    labels = component15_labels.reindex(source.index).astype("string")
    valid_detail = labels.notna()
    pattern = labels.str.split("|", regex=False).str.get(1)
    targets = pd.DataFrame(index=source.index)
    targets["success"] = source[TARGET].astype(np.float32)
    for name, values in (
        ("reverse", pattern.str.startswith("r1", na=False)),
        ("middle", pattern.str.contains("m1", regex=False, na=False)),
        ("ball", pattern.str.contains("b1", regex=False, na=False)),
        ("strike", pattern.str.endswith("s1", na=False)),
    ):
        target = pd.Series(np.nan, index=source.index, dtype=np.float32)
        target.loc[valid_detail] = values.loc[valid_detail].astype(np.float32)
        targets[name] = target

    blocks: list[pd.DataFrame] = []
    table_sizes: dict[str, dict[str, int]] = {}
    for current_season, block in frame.groupby(SEASON, sort=False, observed=True):
        source_mask = source[SEASON].eq(int(current_season) - 1)
        source_rows = source.loc[source_mask]
        source_targets = targets.loc[source_mask]
        destination = pd.DataFrame(index=block.index)
        season_sizes: dict[str, int] = {}
        for spec_name, key_columns_tuple in OUTCOME_CONTEXT_SPECS.items():
            key_columns = list(key_columns_tuple)
            for target_name in targets.columns:
                usable = source_targets[target_name].notna()
                feature = f"e72_{spec_name}_{target_name}_k{int(k)}"
                if not bool(usable.any()):
                    destination[feature] = np.nan
                    season_sizes[f"{spec_name}:{target_name}"] = 0
                    continue
                league = float(source_targets.loc[usable, target_name].mean())
                table_source = source_rows.loc[usable, key_columns].copy()
                table_source["_target"] = source_targets.loc[
                    usable, target_name
                ].to_numpy()
                table = (
                    table_source.groupby(key_columns, sort=False, observed=True)["_target"]
                    .agg(["sum", "count"])
                    .reset_index()
                )
                table[feature] = (
                    (table["sum"] + k * league) / (table["count"] + k) - league
                ).astype(np.float32)
                left = block[key_columns].copy()
                left["_order"] = np.arange(len(left), dtype=np.int64)
                aligned = left.merge(
                    table[key_columns + [feature]],
                    on=key_columns,
                    how="left",
                    sort=False,
                ).sort_values("_order")
                destination[feature] = aligned[feature].to_numpy(dtype=np.float32)
                season_sizes[f"{spec_name}:{target_name}"] = int(len(table))
        blocks.append(destination)
        table_sizes[str(int(current_season))] = season_sizes

    result = pd.concat(blocks).reindex(frame.index)
    return result, {
        "enabled": True,
        "k": float(k),
        "feature_columns": list(result.columns),
        "feature_count": int(result.shape[1]),
        "detail_label_coverage": float(valid_detail.mean()),
        "table_sizes": table_sizes,
        "source_season": "exactly S-1 for each target season S",
        "label_source": "next historical same-pitcher as-of counter delta",
        "row_independent": True,
    }


def build_completed_entity_context_profile(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    entity_column: str,
    prefix: str,
    k: float,
    fallback_prior: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a wide, completed-season-only entity tendency vector.

    Unlike the row's single current context rate, every row receives all
    frozen context cells for its entity.  This lets a shallow tree infer a
    persistent command/style profile while preserving row-independent test
    inference and strict season cutoffs.
    """
    if k <= 0:
        raise ValueError("entity context profile k must be positive")
    if entity_column == PITCHER:
        specs: list[tuple[str, tuple[str, ...], list[tuple[Any, ...]]]] = [
            (
                "count", ("balls_before", "strikes_before"),
                [(balls, strikes) for balls in range(4) for strikes in range(3)],
            ),
            ("opphand", ("batter_hand",), [(1,), (2,)]),
            ("type", ("game_type",), [("R",), ("F",)]),
            ("outs", ("outs_before",), [(0,), (1,), (2,)]),
        ]
    elif entity_column == "batter_id":
        specs = [
            (
                "count", ("balls_before", "strikes_before"),
                [(balls, strikes) for balls in range(4) for strikes in range(3)],
            ),
            ("opphand", ("pitcher_hand",), [(1,), (2,)]),
            ("type", ("game_type",), [("R",), ("F",)]),
            ("outs", ("outs_before",), [(0,), (1,), (2,)]),
        ]
    else:
        raise ValueError(f"Unsupported entity profile column: {entity_column}")

    season_parts: list[pd.DataFrame] = []
    state_sizes: dict[str, dict[str, int]] = {}
    for current_season, block in frame.groupby(SEASON, sort=True, observed=True):
        current_season = int(current_season)
        completed = source.loc[source[SEASON] < current_season]
        overall_prior = (
            float(completed[TARGET].mean())
            if not completed.empty else float(fallback_prior)
        )
        values: dict[str, np.ndarray] = {}
        state_sizes[str(current_season)] = {}
        for spec_name, columns, cells in specs:
            if completed.empty:
                table = None
                cell_priors: dict[Any, float] = {}
                state_sizes[str(current_season)][spec_name] = 0
            else:
                group_columns = [entity_column, *columns]
                table = completed.groupby(
                    group_columns, sort=False, observed=True
                )[TARGET].agg(["sum", "count"])
                prior_table = completed.groupby(
                    list(columns), sort=False, observed=True
                )[TARGET].mean()
                cell_priors = prior_table.to_dict()
                state_sizes[str(current_season)][spec_name] = int(len(table))
            for cell in cells:
                cell_key: Any = cell[0] if len(cell) == 1 else cell
                label = "_".join(str(value).lower() for value in cell)
                prior = float(cell_priors.get(cell_key, overall_prior))
                if table is None:
                    count = np.zeros(len(block), dtype=np.float64)
                    success = np.zeros(len(block), dtype=np.float64)
                else:
                    try:
                        subset = table.xs(
                            cell_key,
                            level=list(columns) if len(columns) > 1 else columns[0],
                            drop_level=True,
                        )
                    except KeyError:
                        subset = table.iloc[0:0]
                    count = block[entity_column].map(subset["count"]).fillna(0.0).to_numpy(
                        dtype=np.float64
                    )
                    success = block[entity_column].map(subset["sum"]).fillna(0.0).to_numpy(
                        dtype=np.float64
                    )
                rate = (success + k * prior) / (count + k)
                values[f"{prefix}_{spec_name}_{label}_rate"] = rate.astype(np.float32)
                values[f"{prefix}_{spec_name}_{label}_n_log"] = np.log1p(count).astype(
                    np.float32
                )
        season_parts.append(pd.DataFrame(values, index=block.index))
    return pd.concat(season_parts).reindex(frame.index), {
        "entity": entity_column,
        "k": float(k),
        "state_sizes": state_sizes,
        "cutoff": "completed seasons only; season-wise OOF for training rows",
        "row_independent": True,
    }


def build_e22_catboost_probabilities(
    history: pd.DataFrame, valid: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Predict pitch-group probabilities from historical Trackman labels only."""
    from catboost import CatBoostClassifier
    from experiments.run_e22r_probs_rolling import E22_PROB_FEATURES, GROUPS

    labeled = history.dropna(subset=["e22_pitch_type_group"])
    if labeled.empty:
        prior = np.full(len(GROUPS), 1.0 / len(GROUPS), dtype=np.float32)
        return (
            pd.DataFrame(np.tile(prior, (len(history), 1)), columns=E22_PROB_FEATURES, index=history.index),
            pd.DataFrame(np.tile(prior, (len(valid), 1)), columns=E22_PROB_FEATURES, index=valid.index),
            {"backend": "catboost", "labeled_rows": 0, "fallback": True},
        )
    features = list(BASE_FEATURES)
    categorical = [column for column in BOOSTER_CATEGORICAL if column in features]

    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame[features].copy()
        for column in categorical:
            result[column] = result[column].astype("string").fillna("__missing__").astype(str)
        return result

    train_x = prepare(labeled)
    history_x = prepare(history)
    valid_x = prepare(valid)
    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=300,
        depth=6,
        learning_rate=0.08,
        l2_leaf_reg=8.0,
        random_seed=RANDOM_SEED,
        allow_writing_files=False,
        thread_count=6,
        task_type=(
            "GPU" if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu" else "CPU"
        ),
    )
    started = time.perf_counter()
    model.fit(
        train_x,
        labeled["e22_pitch_type_group"].astype(str),
        cat_features=categorical,
        verbose=False,
    )

    def aligned(frame_x: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(model.predict_proba(frame_x), dtype=np.float64)
        classes = [str(value) for value in model.classes_]
        result = np.zeros((len(frame_x), len(GROUPS)), dtype=np.float64)
        for source_index, label in enumerate(classes):
            if label in GROUPS:
                result[:, GROUPS.index(label)] = raw[:, source_index]
        denominator = result.sum(axis=1)
        missing = denominator <= 0
        result[missing] = 1.0 / len(GROUPS)
        denominator[missing] = 1.0
        return (result / denominator[:, None]).astype(np.float32)

    train_probs = aligned(history_x)
    valid_probs = aligned(valid_x)
    metadata = {
        "backend": "catboost",
        "labeled_rows": int(len(labeled)),
        "classes": [str(value) for value in model.classes_],
        "features": features,
        "categorical": categorical,
        "fit_seconds": time.perf_counter() - started,
        "uses_current_trackman": False,
        "valid_labels_not_used": int(valid["e22_pitch_type_group"].notna().sum()),
        "valid_probability_mean": valid_probs.mean(axis=0).tolist(),
    }
    del model, train_x, history_x, valid_x
    gc.collect()
    return (
        pd.DataFrame(train_probs, columns=E22_PROB_FEATURES, index=history.index),
        pd.DataFrame(valid_probs, columns=E22_PROB_FEATURES, index=valid.index),
        metadata,
    )


def build_fine_pitch_latent_probabilities(
    history: pd.DataFrame,
    valid: pd.DataFrame,
    label_column: str = "fine_pitch_type",
    use_profile_features: bool = False,
    fit_full_validation_model: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build three-fold cross-fitted fine-pitch probabilities.

    The matched historical fine type is an auxiliary training label only.
    Validation probabilities and unmatched-history probabilities are averages
    of the same three outer-history models.  A labeled history row receives
    only the prediction from the fold model that excluded that row.
    """
    from catboost import CatBoostClassifier

    if label_column not in history.columns or "row_id" not in history.columns:
        raise ValueError(
            f"pitch latent probabilities require row_id and {label_column}"
        )
    labeled_mask = history[label_column].notna().to_numpy(dtype=bool)
    labeled_positions = np.flatnonzero(labeled_mask)
    if len(labeled_positions) < 1000:
        raise ValueError(
            f"too few fine-pitch auxiliary labels: {len(labeled_positions)}"
        )
    features = list(BASE_FEATURES)
    categorical = [column for column in BOOSTER_CATEGORICAL if column in features]

    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame[features].copy()
        for column in categorical:
            result[column] = (
                result[column].astype("string").fillna("__missing__").astype(str)
            )
        return result

    history_x = prepare(history)
    valid_x = prepare(valid)
    labeled = history.iloc[labeled_positions]
    hashes = pd.util.hash_pandas_object(
        labeled["row_id"].astype(str), index=False
    ).to_numpy(dtype=np.uint64)
    fold_ids = np.asarray(hashes % np.uint64(3), dtype=np.int8)
    train_probabilities = np.zeros(
        (len(history), len(FINE_PITCH_TYPES)), dtype=np.float64
    )
    valid_probabilities = np.zeros(
        (len(valid), len(FINE_PITCH_TYPES)), dtype=np.float64
    )
    fit_rows: list[int] = []
    fit_seconds: list[float] = []
    profile_feature_columns = [
        f"e91_{name}_p_{pitch_type.lower()}"
        for name, _, _ in FINE_PITCH_PROFILE_SPECS
        for pitch_type in FINE_PITCH_TYPES
    ] if use_profile_features else []
    train_profile_values = np.zeros(
        (len(history), len(profile_feature_columns)), dtype=np.float64
    )
    valid_profile_values = np.zeros(
        (len(valid), len(profile_feature_columns)), dtype=np.float64
    )

    def profile_features(
        fit_positions: np.ndarray,
        query: pd.DataFrame,
        leave_one_out: bool = False,
    ) -> pd.DataFrame:
        """Cross-fitted fine-pitch repertoire probabilities.

        Every table is built only from the selector model's fitting rows.  The
        model-fitting matrix receives leave-one-out values, while held-out
        history and outer validation rows receive frozen lookup values.
        """
        fit_frame = history.iloc[fit_positions]
        fit_labels = fit_frame[label_column].astype(str)
        codes = np.array(
            [FINE_PITCH_TYPES.index(value) for value in fit_labels],
            dtype=np.int16,
        )
        one_hot = np.eye(len(FINE_PITCH_TYPES), dtype=np.float64)[codes]
        global_counts = one_hot.sum(axis=0)
        global_prior = global_counts / global_counts.sum()
        result_parts: list[np.ndarray] = []
        query_is_fit = leave_one_out
        if query_is_fit and not query.index.equals(fit_frame.index):
            raise ValueError("leave-one-out pitch profiles require the fit rows")

        for _, keys, shrinkage in FINE_PITCH_PROFILE_SPECS:
            key_columns = list(keys)
            counts = (
                pd.DataFrame(one_hot, index=fit_frame.index)
                .groupby(
                    [fit_frame[column] for column in key_columns],
                    sort=False,
                    observed=True,
                    dropna=False,
                )
                .sum()
            )
            counts.columns = list(FINE_PITCH_TYPES)
            if len(key_columns) == 1:
                query_index = pd.Index(
                    query[key_columns[0]].to_numpy(),
                    name=counts.index.name,
                )
            else:
                query_index = pd.MultiIndex.from_frame(query[key_columns])
                query_index.names = counts.index.names
            mapped = counts.reindex(query_index).to_numpy(dtype=np.float64)
            missing = np.isnan(mapped).all(axis=1)
            mapped[missing] = 0.0
            totals = mapped.sum(axis=1)
            prior = np.broadcast_to(global_prior, mapped.shape).copy()
            if leave_one_out:
                mapped -= one_hot
                totals -= 1.0
                loo_global = (
                    global_counts[None, :] - one_hot
                ) / max(1.0, float(len(fit_positions) - 1))
                prior = loo_global
            probabilities = (
                mapped + float(shrinkage) * prior
            ) / (totals[:, None] + float(shrinkage))
            result_parts.append(probabilities)
        matrix = np.concatenate(result_parts, axis=1)
        return pd.DataFrame(
            matrix.astype(np.float32),
            columns=profile_feature_columns,
            index=query.index,
        )

    def aligned(model: Any, frame_x: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(model.predict_proba(frame_x), dtype=np.float64)
        result = np.zeros((len(frame_x), len(FINE_PITCH_TYPES)), dtype=np.float64)
        for source_index, label in enumerate(str(value) for value in model.classes_):
            if label in FINE_PITCH_TYPES:
                result[:, FINE_PITCH_TYPES.index(label)] = raw[:, source_index]
        denominator = result.sum(axis=1)
        missing = denominator <= 0.0
        result[missing] = 1.0 / len(FINE_PITCH_TYPES)
        denominator[missing] = 1.0
        return result / denominator[:, None]

    for fold in range(3):
        fold_train = fold_ids != fold
        fit_positions = labeled_positions[fold_train]
        model = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=400,
            depth=6,
            learning_rate=0.06,
            l2_leaf_reg=20.0,
            random_seed=RANDOM_SEED + 900 + fold,
            allow_writing_files=False,
            thread_count=6,
            task_type=(
                "GPU"
                if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu"
                else "CPU"
            ),
        )
        started = time.perf_counter()
        fit_x = history_x.iloc[fit_positions].copy()
        history_predict_x = history_x
        valid_predict_x = valid_x
        history_profile = valid_profile = None
        if use_profile_features:
            fit_profile = profile_features(
                fit_positions, history.iloc[fit_positions], True
            )
            history_profile = profile_features(fit_positions, history)
            valid_profile = profile_features(fit_positions, valid)
            fit_x = pd.concat(
                [fit_x, fit_profile], axis=1,
            )
            history_predict_x = pd.concat(
                [history_x, history_profile], axis=1
            )
            valid_predict_x = pd.concat(
                [valid_x, valid_profile], axis=1
            )
        model.fit(
            fit_x,
            history.iloc[fit_positions][label_column].astype(str),
            cat_features=categorical,
            verbose=False,
        )
        history_raw = aligned(model, history_predict_x)
        valid_probabilities += aligned(model, valid_predict_x) / 3.0
        held_out_positions = labeled_positions[fold_ids == fold]
        train_probabilities[held_out_positions] = history_raw[held_out_positions]
        train_probabilities[~labeled_mask] += history_raw[~labeled_mask] / 3.0
        if use_profile_features:
            history_profile_values = history_profile.to_numpy(dtype=np.float64)
            train_profile_values[held_out_positions] = history_profile_values[
                held_out_positions
            ]
            train_profile_values[~labeled_mask] += (
                history_profile_values[~labeled_mask] / 3.0
            )
            valid_profile_values += (
                valid_profile.to_numpy(dtype=np.float64) / 3.0
            )
        fit_rows.append(int(fold_train.sum()))
        fit_seconds.append(float(time.perf_counter() - started))
        del model, history_raw, fit_x, history_predict_x, valid_predict_x
        if use_profile_features:
            del fit_profile, history_profile, valid_profile, history_profile_values
        gc.collect()

    for probabilities in (train_probabilities, valid_probabilities):
        denominator = probabilities.sum(axis=1, keepdims=True)
        if np.any(denominator <= 0.0):
            raise AssertionError("fine-pitch probability row was not assigned")
        probabilities /= denominator

    full_validation_fit_seconds = None
    if fit_full_validation_model:
        full_model = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=400,
            depth=6,
            learning_rate=0.06,
            l2_leaf_reg=20.0,
            random_seed=(
                RANDOM_SEED + int(valid[SEASON].iloc[0])
                + (100 if label_column == "auto_fine_pitch_type" else 0)
            ),
            allow_writing_files=False,
            thread_count=6,
            task_type=(
                "GPU"
                if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu"
                else "CPU"
            ),
        )
        full_started = time.perf_counter()
        full_model.fit(
            history_x.iloc[labeled_positions],
            history.iloc[labeled_positions][label_column].astype(str),
            cat_features=categorical,
            verbose=False,
        )
        valid_probabilities = aligned(full_model, valid_x)
        full_validation_fit_seconds = float(time.perf_counter() - full_started)
        del full_model
        gc.collect()

    def frame_from(probabilities: np.ndarray, index: pd.Index) -> pd.DataFrame:
        entropy = -np.sum(
            np.where(
                probabilities > 0.0,
                probabilities * np.log(np.maximum(probabilities, 1e-12)),
                0.0,
            ),
            axis=1,
        )
        result = pd.DataFrame(
            probabilities.astype(np.float32),
            columns=FINE_PITCH_PROBABILITY_COLUMNS,
            index=index,
        )
        result["e90_entropy"] = entropy.astype(np.float32)
        result["e90_max_probability"] = probabilities.max(axis=1).astype(np.float32)
        return result

    valid_matched = valid[label_column].notna().to_numpy(dtype=bool)
    truth = valid.loc[valid_matched, label_column].astype(str).to_numpy()
    truth_index = np.array(
        [FINE_PITCH_TYPES.index(value) for value in truth], dtype=np.int16
    )
    matched_probabilities = valid_probabilities[valid_matched]
    chosen = matched_probabilities[np.arange(len(truth_index)), truth_index]
    metadata = {
        "enabled": True,
        "label_column": label_column,
        "backend": "three_fold_cross_fitted_catboost_multiclass",
        "history_labeled_rows": int(labeled_mask.sum()),
        "history_unmatched_rows": int((~labeled_mask).sum()),
        "fold_fit_rows": fit_rows,
        "fold_fit_seconds": fit_seconds,
        "valid_matched_rows_not_used": int(valid_matched.sum()),
        "valid_top1_accuracy": float(
            np.mean(matched_probabilities.argmax(axis=1) == truth_index)
        ),
        "valid_log_loss": float(
            -np.mean(np.log(np.clip(chosen, 1e-12, 1.0)))
        ),
        "feature_columns": [
            *FINE_PITCH_PROBABILITY_COLUMNS,
            "e90_entropy",
            "e90_max_probability",
            *profile_feature_columns,
        ],
        "profile_features_enabled": bool(use_profile_features),
        "profile_specs": [
            {"name": name, "keys": list(keys), "shrinkage": float(shrinkage)}
            for name, keys, shrinkage in FINE_PITCH_PROFILE_SPECS
        ] if use_profile_features else [],
        "profile_feature_columns": profile_feature_columns,
        "profile_training_self_excluded": bool(use_profile_features),
        "full_history_validation_model": bool(fit_full_validation_model),
        "full_history_validation_fit_seconds": full_validation_fit_seconds,
        "current_pitch_type_at_inference": False,
        "row_independent": True,
    }
    del history_x, valid_x, labeled
    gc.collect()
    return (
        pd.concat(
            [
                frame_from(train_probabilities, history.index),
                pd.DataFrame(
                    train_profile_values.astype(np.float32),
                    columns=profile_feature_columns,
                    index=history.index,
                ),
            ],
            axis=1,
        ) if use_profile_features else frame_from(train_probabilities, history.index),
        pd.concat(
            [
                frame_from(valid_probabilities, valid.index),
                pd.DataFrame(
                    valid_profile_values.astype(np.float32),
                    columns=profile_feature_columns,
                    index=valid.index,
                ),
            ],
            axis=1,
        ) if use_profile_features else frame_from(valid_probabilities, valid.index),
        metadata,
    )


def build_expanded_auto_pitch_probabilities(
    history: pd.DataFrame,
    valid: pd.DataFrame,
    baseline_train: pd.DataFrame,
    baseline_valid: pd.DataFrame,
    joined_trackman: pd.DataFrame,
    raw_trackman: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply the source-locked full-TrackMan repertoire selector.

    Exact game linkage is used only to recover a high-purity identity map.
    Unmatched official TrackMan games contribute pitcher/count repertoire
    counts, but are never asserted to be individual main-table rows.  For a
    matched training row, its own auto-pitch label is subtracted from both
    profile cells before the probability is exposed to the outcome model.
    """
    pitcher_k = 100.0
    count_k = 20.0
    catboost_weight = 0.5
    allowed_seasons = sorted(int(value) for value in history[SEASON].unique())
    joined_history = joined_trackman.loc[
        joined_trackman[SEASON].isin(allowed_seasons),
        [PITCHER, "pitcher_trackman_id", "auto_pitch_type"],
    ].dropna(subset=[PITCHER, "pitcher_trackman_id"])
    pair_counts = joined_history.groupby(
        [PITCHER, "pitcher_trackman_id"], sort=False, observed=True
    ).size().rename("n").reset_index()
    totals = pair_counts.groupby(PITCHER, observed=True)["n"].transform("sum")
    pair_counts["purity"] = pair_counts["n"] / totals
    best = pair_counts.sort_values(
        [PITCHER, "n"], ascending=[True, False], kind="stable"
    ).drop_duplicates(PITCHER)
    best = best.loc[best["purity"].ge(0.99)].copy()
    best = best.sort_values(
        ["pitcher_trackman_id", "n"], ascending=[True, False], kind="stable"
    ).drop_duplicates("pitcher_trackman_id")
    inverse_map = {
        trackman_id: int(pitcher_id)
        for pitcher_id, trackman_id in zip(
            best[PITCHER], best["pitcher_trackman_id"]
        )
    }

    expanded = raw_trackman.loc[
        raw_trackman[SEASON].isin(allowed_seasons),
        [
            "pitcher_trackman_id", "trackman_game_id", "balls_before",
            "strikes_before", "auto_pitch_type",
        ],
    ].copy()
    expanded[PITCHER] = expanded["pitcher_trackman_id"].map(inverse_map)
    expanded = expanded.loc[expanded[PITCHER].notna()].copy()
    expanded[PITCHER] = expanded[PITCHER].astype(np.int64)
    normalized = expanded["auto_pitch_type"].astype("string").replace(
        {"Changeup": "ChangeUp", "Four-Seam": "Fastball", "SInker": "Sinker"}
    )
    expanded["_fine_auto"] = normalized.where(
        normalized.isin(FINE_PITCH_TYPES[:-1]), "Other"
    )
    global_counts = (
        expanded["_fine_auto"].value_counts().reindex(FINE_PITCH_TYPES)
        .fillna(0.0).to_numpy(dtype=np.float64)
    )
    global_total = float(global_counts.sum())
    global_prior = global_counts / global_total

    def table(keys: list[str]) -> pd.DataFrame:
        result = expanded.groupby(
            [*keys, "_fine_auto"], sort=False, observed=True, dropna=False
        ).size().unstack("_fine_auto", fill_value=0)
        return result.reindex(columns=FINE_PITCH_TYPES, fill_value=0).astype(
            np.float64
        )

    pitcher_counts = table([PITCHER])
    cell_counts = table([PITCHER, "balls_before", "strikes_before"])

    def mapped_counts(
        query: pd.DataFrame, source: pd.DataFrame, keys: list[str]
    ) -> np.ndarray:
        if len(keys) == 1:
            index = pd.Index(query[keys[0]].to_numpy(), name=source.index.name)
        else:
            index = pd.MultiIndex.from_frame(query[keys])
            index.names = source.index.names
        values = source.reindex(index).to_numpy(dtype=np.float64)
        missing = np.isnan(values).all(axis=1)
        values[missing] = 0.0
        return values

    def profile(query: pd.DataFrame, subtract_self: bool) -> np.ndarray:
        pitcher_raw = mapped_counts(query, pitcher_counts, [PITCHER])
        cell_raw = mapped_counts(
            query, cell_counts, [PITCHER, "balls_before", "strikes_before"]
        )
        prior = np.broadcast_to(global_prior, pitcher_raw.shape).copy()
        if subtract_self:
            labels = query["auto_fine_pitch_type"].astype("string")
            matched = labels.isin(FINE_PITCH_TYPES).to_numpy(dtype=bool)
            codes = np.full(len(query), -1, dtype=np.int16)
            codes[matched] = np.array(
                [FINE_PITCH_TYPES.index(value) for value in labels.loc[matched]],
                dtype=np.int16,
            )
            rows = np.flatnonzero(matched)
            present = (
                (pitcher_raw[rows, codes[rows]] >= 1.0)
                & (cell_raw[rows, codes[rows]] >= 1.0)
            )
            rows = rows[present]
            pitcher_raw[rows, codes[rows]] -= 1.0
            cell_raw[rows, codes[rows]] -= 1.0
            if global_total > 1.0:
                prior[rows] = (
                    global_counts[None, :]
                    - np.eye(len(FINE_PITCH_TYPES))[codes[rows]]
                ) / (global_total - 1.0)
        pitcher_total = pitcher_raw.sum(axis=1)
        pitcher_probability = (
            pitcher_raw + pitcher_k * prior
        ) / (pitcher_total[:, None] + pitcher_k)
        cell_total = cell_raw.sum(axis=1)
        return (
            cell_raw + count_k * pitcher_probability
        ) / (cell_total[:, None] + count_k)

    train_profile = profile(history, True)
    valid_profile = profile(valid, False)
    baseline_columns = FINE_PITCH_PROBABILITY_COLUMNS
    baseline_train_probability = baseline_train[baseline_columns].to_numpy(
        dtype=np.float64
    )
    baseline_valid_probability = baseline_valid[baseline_columns].to_numpy(
        dtype=np.float64
    )

    def blend(baseline: np.ndarray, repertoire: np.ndarray) -> np.ndarray:
        result = np.exp(
            catboost_weight * np.log(np.clip(baseline, 1e-12, 1.0))
            + (1.0 - catboost_weight)
            * np.log(np.clip(repertoire, 1e-12, 1.0))
        )
        return result / result.sum(axis=1, keepdims=True)

    train_probability = blend(baseline_train_probability, train_profile)
    valid_probability = blend(baseline_valid_probability, valid_profile)
    columns = [
        f"e92_p_{pitch_type.lower()}" for pitch_type in FINE_PITCH_TYPES
    ]

    def output_frame(probability: np.ndarray, index: pd.Index) -> pd.DataFrame:
        result = pd.DataFrame(
            probability.astype(np.float32), columns=columns, index=index
        )
        result["e92_entropy"] = -np.sum(
            probability * np.log(np.clip(probability, 1e-12, 1.0)), axis=1
        ).astype(np.float32)
        result["e92_max_probability"] = probability.max(axis=1).astype(np.float32)
        return result

    matched_valid = valid["auto_fine_pitch_type"].isin(FINE_PITCH_TYPES).to_numpy(
        dtype=bool
    )
    truth = valid.loc[matched_valid, "auto_fine_pitch_type"].astype(str)
    truth_index = np.array(
        [FINE_PITCH_TYPES.index(value) for value in truth], dtype=np.int16
    )

    def selector_diagnostic(probability: np.ndarray) -> dict[str, float]:
        selected = probability[matched_valid]
        chosen = selected[np.arange(len(truth_index)), truth_index]
        return {
            "top1_accuracy": float(
                np.mean(selected.argmax(axis=1) == truth_index)
            ),
            "log_loss": float(-np.mean(np.log(np.clip(chosen, 1e-12, 1.0)))),
        }

    baseline_diagnostic = selector_diagnostic(baseline_valid_probability)
    expanded_diagnostic = selector_diagnostic(valid_probability)
    joined_labeled_rows = int(
        joined_history["auto_pitch_type"].notna().sum()
    )
    metadata = {
        "enabled": True,
        "architecture": "history_mapped_full_trackman_repertoire_geometric_selector",
        "allowed_history_seasons": allowed_seasons,
        "identity_minimum_purity": float(best["purity"].min()),
        "mapped_pitchers": int(len(best)),
        "expanded_trackman_rows": int(len(expanded)),
        "joined_labeled_history_rows": joined_labeled_rows,
        "expansion_factor": float(len(expanded) / max(1, joined_labeled_rows)),
        "pitcher_k": pitcher_k,
        "pitcher_count_k": count_k,
        "count_weight": 1.0,
        "catboost_geometric_weight": catboost_weight,
        "baseline_selector": baseline_diagnostic,
        "expanded_selector": expanded_diagnostic,
        "selector_log_loss_improvement": float(
            baseline_diagnostic["log_loss"] - expanded_diagnostic["log_loss"]
        ),
        "selector_top1_improvement": float(
            expanded_diagnostic["top1_accuracy"]
            - baseline_diagnostic["top1_accuracy"]
        ),
        "training_self_pitch_subtracted": True,
        "current_pitch_type_at_inference": False,
        "current_pitch_trackman_at_inference": False,
        "row_independent": True,
        "feature_columns": [*columns, "e92_entropy", "e92_max_probability"],
    }
    del expanded, pitcher_counts, cell_counts, normalized
    gc.collect()
    return (
        output_frame(train_probability, history.index),
        output_frame(valid_probability, valid.index),
        metadata,
    )


def build_e14_multi_features(
    frame: pd.DataFrame,
    e14: pd.DataFrame,
    prior_by_season: dict[int, float],
    validation_prior: float,
) -> pd.DataFrame:
    """Expose several empirical-Bayes reliability scales to tree models.

    ``e14_s_season`` and ``e14_n_season`` are reconstructed only from the
    current row's official as-of counters and frozen pre-season state.  The
    transformations below therefore remain row independent and future safe.
    """
    n = e14["e14_n_season"].to_numpy(dtype=np.float64, copy=False)
    s = e14["e14_s_season"].to_numpy(dtype=np.float64, copy=False)
    career = frame["asof_pitcher_success_rate"].fillna(validation_prior).to_numpy(
        dtype=np.float64, copy=False
    )
    row_priors = np.asarray(
        [
            prior_by_season.get(int(season), validation_prior)
            for season in frame[SEASON]
        ],
        dtype=np.float64,
    )
    values: dict[str, np.ndarray] = {
        "e14_raw_rate_season": np.divide(
            s, n, out=row_priors.copy(), where=n > 0
        ).astype(np.float32),
    }
    for k in E14_MULTI_KS:
        label = str(int(k))
        rate = (s + k * row_priors) / (n + k)
        values[f"e14_rate_k{label}"] = rate.astype(np.float32)
        values[f"e14_delta_k{label}"] = (rate - career).astype(np.float32)
        values[f"e14_reliability_k{label}"] = (n / (n + k)).astype(np.float32)
    return pd.DataFrame(values, index=frame.index)


def component_states_before_each_season(
    frame: pd.DataFrame,
) -> tuple[
    dict[int, dict[int, tuple[int, ...]]],
    dict[int, dict[str, float]],
    dict[int, tuple[int, ...]],
    dict[str, float],
]:
    """Freeze cumulative component counters before each historical season.

    Component labels for the final historical pitch are not exposed.  We
    therefore freeze the official as-of counters immediately before that
    pitch (rather than fabricating its outcome).  The one-row lag is identical
    in rolling validation and final inference and is negligible after EB
    smoothing.
    """
    states_before: dict[int, dict[int, tuple[int, ...]]] = {}
    priors_before: dict[int, dict[str, float]] = {}
    state: dict[int, tuple[int, ...]] = {}

    def state_priors(current: dict[int, tuple[int, ...]]) -> dict[str, float]:
        denominator = float(sum(values[0] for values in current.values()))
        if denominator <= 0:
            return {name: 0.5 for name in COMPONENT_RATE_COLUMNS}
        return {
            name: float(sum(values[index + 1] for values in current.values()) / denominator)
            for index, name in enumerate(COMPONENT_RATE_COLUMNS)
        }

    for season in sorted(int(value) for value in frame[SEASON].unique()):
        states_before[season] = dict(state)
        priors_before[season] = state_priors(state)
        last_rows = frame.loc[frame[SEASON] == season].groupby(
            PITCHER, sort=False, observed=True
        ).tail(1)
        for row in last_rows.itertuples(index=False):
            n = int(getattr(row, "asof_pitcher_n") or 0)
            counts = []
            for column in COMPONENT_RATE_COLUMNS.values():
                value = getattr(row, column)
                counts.append(int(np.rint((0.0 if pd.isna(value) else float(value)) * n)))
            state[int(getattr(row, PITCHER))] = (n, *counts)
    return states_before, priors_before, state, state_priors(state)


def build_component_features(
    frame: pd.DataFrame,
    states_before: dict[int, dict[int, tuple[int, ...]]],
    priors_before: dict[int, dict[str, float]],
    validation_priors: dict[str, float],
    k: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recover season-to-date rates for four official cumulative components."""
    if k <= 0:
        raise ValueError("--component-k must be positive")
    n_end = np.zeros(len(frame), dtype=np.int64)
    component_end = np.zeros((len(frame), len(COMPONENT_RATE_COLUMNS)), dtype=np.int64)
    seasons = frame[SEASON].to_numpy(dtype=np.int16, copy=False)
    pitchers = frame[PITCHER].to_numpy(dtype=np.int32, copy=False)
    for index, (season, pitcher) in enumerate(zip(seasons, pitchers)):
        state = states_before.get(int(season), {}).get(int(pitcher))
        if state is not None:
            n_end[index] = state[0]
            component_end[index] = state[1:]

    n_asof = frame["asof_pitcher_n"].fillna(0).to_numpy(dtype=np.int64, copy=False)
    n_delta = n_asof - n_end
    values: dict[str, np.ndarray] = {}
    invalid_total = np.zeros(len(frame), dtype=bool)
    for component_index, (name, column) in enumerate(COMPONENT_RATE_COLUMNS.items()):
        career = frame[column].fillna(validation_priors[name]).to_numpy(
            dtype=np.float64, copy=False
        )
        count_asof = np.rint(career * n_asof).astype(np.int64)
        count_delta = count_asof - component_end[:, component_index]
        invalid = (n_delta < 0) | (count_delta < 0) | (count_delta > n_delta)
        invalid_total |= invalid
        safe_n = np.where(invalid, 0, n_delta)
        safe_count = np.where(invalid, 0, count_delta)
        row_prior = np.asarray(
            [
                priors_before.get(int(season), validation_priors).get(
                    name, validation_priors[name]
                )
                for season in seasons
            ],
            dtype=np.float64,
        )
        raw = np.divide(
            safe_count, safe_n, out=row_prior.copy(), where=safe_n > 0
        )
        smooth = (safe_count + k * row_prior) / (safe_n + k)
        values[f"e31_{name}_rate_season"] = smooth.astype(np.float32)
        values[f"e31_{name}_raw_season"] = raw.astype(np.float32)
        values[f"e31_{name}_delta_career"] = (smooth - career).astype(np.float32)
    values["e31_component_invalid"] = invalid_total.astype(np.int8)
    result = pd.DataFrame(values, index=frame.index)
    return result, {
        "k": float(k),
        "invalid_rows": int(invalid_total.sum()),
        "feature_columns": list(result.columns),
        "state_cutoff": "official cumulative counters immediately before final prior-season pitch",
    }


def generic_component_states_before_each_season(
    frame: pd.DataFrame,
    entity_column: str,
    n_column: str,
    component_columns: dict[str, str],
) -> tuple[
    dict[int, dict[int, tuple[int, ...]]],
    dict[int, dict[str, float]],
    dict[int, tuple[int, ...]],
    dict[str, float],
]:
    """Freeze one or more official cumulative component counters by entity."""
    states_before: dict[int, dict[int, tuple[int, ...]]] = {}
    priors_before: dict[int, dict[str, float]] = {}
    state: dict[int, tuple[int, ...]] = {}

    def priors(current: dict[int, tuple[int, ...]]) -> dict[str, float]:
        denominator = float(sum(value[0] for value in current.values()))
        if denominator <= 0:
            return {name: 0.0 for name in component_columns}
        return {
            name: float(sum(value[index + 1] for value in current.values()) / denominator)
            for index, name in enumerate(component_columns)
        }

    for current_season in sorted(int(value) for value in frame[SEASON].unique()):
        states_before[current_season] = dict(state)
        priors_before[current_season] = priors(state)
        last_rows = frame.loc[frame[SEASON] == current_season].groupby(
            entity_column, sort=False, observed=True
        ).tail(1)
        for row in last_rows.itertuples(index=False):
            n = int(getattr(row, n_column) or 0)
            counts = []
            for column in component_columns.values():
                value = getattr(row, column)
                counts.append(
                    int(np.rint((0.0 if pd.isna(value) else float(value)) * n))
                )
            state[int(getattr(row, entity_column))] = (n, *counts)
    return states_before, priors_before, state, priors(state)


def build_generic_component_features(
    frame: pd.DataFrame,
    states_before: dict[int, dict[int, tuple[int, ...]]],
    priors_before: dict[int, dict[str, float]],
    validation_priors: dict[str, float],
    entity_column: str,
    n_column: str,
    component_columns: dict[str, str],
    prefix: str,
    k: float,
    include_raw: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if k <= 0:
        raise ValueError("generic component smoothing k must be positive")
    seasons = frame[SEASON].to_numpy(dtype=np.int16, copy=False)
    entities = pd.to_numeric(frame[entity_column], errors="coerce").fillna(-1).to_numpy(
        dtype=np.int64
    )
    n_end = np.zeros(len(frame), dtype=np.int64)
    component_end = np.zeros((len(frame), len(component_columns)), dtype=np.int64)
    unseen = np.zeros(len(frame), dtype=np.int8)
    for index, (current_season, entity) in enumerate(zip(seasons, entities)):
        state = states_before.get(int(current_season), {}).get(int(entity))
        if state is None:
            unseen[index] = 1
        else:
            n_end[index] = state[0]
            component_end[index] = state[1:]
    n_asof = pd.to_numeric(frame[n_column], errors="coerce").fillna(0).to_numpy(
        dtype=np.int64
    )
    n_delta = n_asof - n_end
    values: dict[str, np.ndarray] = {
        f"{prefix}_n_season": np.maximum(n_delta, 0).astype(np.int32),
        f"{prefix}_log_n_season": np.log1p(np.maximum(n_delta, 0)).astype(np.float32),
        f"{prefix}_unseen": unseen,
    }
    invalid_total = n_delta < 0
    for component_index, (name, column) in enumerate(component_columns.items()):
        career = pd.to_numeric(frame[column], errors="coerce").fillna(
            validation_priors[name]
        ).to_numpy(dtype=np.float64)
        count_asof = np.rint(career * n_asof).astype(np.int64)
        count_delta = count_asof - component_end[:, component_index]
        invalid = (n_delta < 0) | (count_delta < 0) | (count_delta > n_delta)
        invalid_total |= invalid
        safe_n = np.where(invalid, 0, n_delta)
        safe_count = np.where(invalid, 0, count_delta)
        row_prior = np.asarray(
            [
                priors_before.get(int(current_season), validation_priors).get(
                    name, validation_priors[name]
                )
                for current_season in seasons
            ],
            dtype=np.float64,
        )
        rate = (safe_count + k * row_prior) / (safe_n + k)
        values[f"{prefix}_{name}_rate_season"] = rate.astype(np.float32)
        values[f"{prefix}_{name}_rate_delta"] = (rate - career).astype(np.float32)
        if include_raw:
            raw = np.divide(
                safe_count,
                safe_n,
                out=np.full(len(frame), np.nan, dtype=np.float64),
                where=safe_n > 0,
            )
            values[f"{prefix}_{name}_raw_season"] = raw.astype(np.float32)
            values[f"{prefix}_{name}_count_season"] = safe_count.astype(np.int32)
    values[f"{prefix}_counter_invalid"] = invalid_total.astype(np.int8)
    return pd.DataFrame(values, index=frame.index), {
        "k": float(k),
        "invalid_rows": int(invalid_total.sum()),
        "unseen_rows": int(unseen.sum()),
        "n_positive_rate": float(np.mean(n_delta > 0)),
        "n_median": float(np.median(np.maximum(n_delta, 0))),
        "state_cutoff": "official as-of counters before the final prior-season row",
        "raw_current_state_included": bool(include_raw),
    }


def build_current_state_full_features(
    e14: pd.DataFrame,
    components: pd.DataFrame,
    batter_e14: pd.DataFrame,
    auxiliary: pd.DataFrame,
) -> pd.DataFrame:
    """Expose the complete 13-value current-season state decomposition.

    The official as-of counters are lifetime cumulative.  Every value below is
    the current row's cumulative count minus a player constant frozen before
    that row's season.  It therefore remains a row-independent lookup at test
    time.  Raw rates intentionally remain NaN at zero current-season samples;
    CatBoost can distinguish that state from an observed league-average rate.
    """

    def raw_rate(n_column: str, count_column: str) -> np.ndarray:
        n = e14[n_column].to_numpy(dtype=np.float64, copy=False)
        count = e14[count_column].to_numpy(dtype=np.float64, copy=False)
        return np.divide(
            count,
            n,
            out=np.full(len(e14), np.nan, dtype=np.float64),
            where=n > 0,
        ).astype(np.float32)

    pitcher_n = e14["e14_n_season"].to_numpy(dtype=np.float64, copy=False)
    batter_n = batter_e14["e49_batter_n_season"].to_numpy(
        dtype=np.float64, copy=False
    )
    batter_success = np.divide(
        batter_e14["e49_batter_s_season"].to_numpy(dtype=np.float64, copy=False),
        batter_n,
        out=np.full(len(e14), np.nan, dtype=np.float64),
        where=batter_n > 0,
    )
    required_components = [
        f"e31_{name}_raw_season"
        for name in ("middle", "ball", "reverse", "strike")
    ]
    required_auxiliary = [
        "e52_batter_middle_raw_season",
        "e53_pitchmix_fastball_raw_season",
        "e53_pitchmix_breaking_raw_season",
        "e53_pitchmix_offspeed_raw_season",
        "e53_pitchmix_log_n_season",
    ]
    missing = [
        column
        for column in [*required_components, *required_auxiliary]
        if column not in components.columns and column not in auxiliary.columns
    ]
    if missing:
        raise ValueError(f"current_state_full missing dependency columns: {missing}")
    return pd.DataFrame(
        {
            "e70_cur_pitcher_success": raw_rate(
                "e14_n_season", "e14_s_season"
            ),
            "e70_cur_pitcher_middle": components["e31_middle_raw_season"].to_numpy(),
            "e70_cur_pitcher_ball": components["e31_ball_raw_season"].to_numpy(),
            "e70_cur_pitcher_reverse": components["e31_reverse_raw_season"].to_numpy(),
            "e70_cur_pitcher_strike": components["e31_strike_raw_season"].to_numpy(),
            "e70_cur_pitcher_fastball": auxiliary[
                "e53_pitchmix_fastball_raw_season"
            ].to_numpy(),
            "e70_cur_pitcher_breaking": auxiliary[
                "e53_pitchmix_breaking_raw_season"
            ].to_numpy(),
            "e70_cur_pitcher_offspeed": auxiliary[
                "e53_pitchmix_offspeed_raw_season"
            ].to_numpy(),
            "e70_cur_batter_success": batter_success.astype(np.float32),
            "e70_cur_batter_middle": auxiliary[
                "e52_batter_middle_raw_season"
            ].to_numpy(),
            "e70_cur_pitcher_log_n": np.log1p(pitcher_n).astype(np.float32),
            "e70_cur_batter_log_n": np.log1p(batter_n).astype(np.float32),
            "e70_cur_pitchmix_log_n": auxiliary[
                "e53_pitchmix_log_n_season"
            ].to_numpy(),
        },
        index=e14.index,
    )


def build_current_state_interaction_features(
    frame: pd.DataFrame,
    current_state: pd.DataFrame,
    include_context: bool,
    include_level: bool,
) -> pd.DataFrame:
    """Add row-local interactions to the reconstructed current-season state.

    Every multiplier is known on the pitch row itself.  The rate inputs are
    produced by :func:`build_current_state_full_features` from official as-of
    counters and constants frozen before the evaluated season, so these
    interactions remain independent of test batch composition and row order.
    """

    balls = pd.to_numeric(frame["balls_before"], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    pitcher_hand = pd.to_numeric(
        frame["pitcher_hand"], errors="coerce"
    ).to_numpy(dtype=np.float64, copy=False)
    batter_hand = pd.to_numeric(
        frame["batter_hand"], errors="coerce"
    ).to_numpy(dtype=np.float64, copy=False)
    runners = pd.to_numeric(
        frame["num_runners_on"], errors="coerce"
    ).to_numpy(dtype=np.float64, copy=False)
    multipliers = {
        "count_advantage": (strikes > balls).astype(np.float32),
        "runner_present": (runners > 0).astype(np.float32),
        "same_hand": (pitcher_hand == batter_hand).astype(np.float32),
        "ball_strike_gap": (balls - strikes).astype(np.float32),
    }
    values: dict[str, np.ndarray] = {}
    if include_context:
        for rate_name in ("success", "middle"):
            rate = current_state[f"e70_cur_pitcher_{rate_name}"].to_numpy(
                dtype=np.float32, copy=False
            )
            for context_name, multiplier in multipliers.items():
                values[f"e72_{rate_name}_x_{context_name}"] = rate * multiplier
    if include_level:
        for rate_name in ("ball", "reverse", "strike"):
            rate = current_state[f"e70_cur_pitcher_{rate_name}"].to_numpy(
                dtype=np.float32, copy=False
            )
            for context_name in ("same_hand", "ball_strike_gap"):
                values[f"e73_{rate_name}_x_{context_name}"] = (
                    rate * multipliers[context_name]
                )
    return pd.DataFrame(values, index=frame.index)


def build_pitcher_te_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose pitcher identity to sklearn TargetEncoder as a dedicated branch.

    The encoder itself lives inside the fitted sklearn pipeline.  During fit,
    ``TargetEncoder.fit_transform`` uses cross-fitting; during inference it uses
    the full training mapping and therefore remains row independent.
    """
    values = pd.to_numeric(frame[PITCHER], errors="coerce").astype("Int64").astype("string")
    return pd.DataFrame(
        {"b2_pitcher_id": values.fillna("__missing__")}, index=frame.index
    )


# --------------------------------------------------------------------------- #
# Platoon encoder (B1'): pitcher main effect plus a shrunk platoon residual.
# --------------------------------------------------------------------------- #
def build_platoon_state(
    history: pd.DataFrame,
    prior: float,
    k_pitcher: float = PLATOON_K_PITCHER,
    k_platoon: float = PLATOON_K_PLATOON,
) -> dict[str, Any]:
    """Fit the pitcher and platoon-residual encoders on one slice of history."""
    if history.empty:
        return {
            "prior": float(prior),
            "k_pitcher": float(k_pitcher),
            "k_platoon": float(k_platoon),
            "pitcher_rate": {},
            "platoon_delta": {},
            "platoon_n": {},
        }
    grouped = history.groupby(PITCHER, observed=True)[TARGET].agg(["sum", "size"])
    pitcher_rate = (grouped["sum"] + k_pitcher * prior) / (grouped["size"] + k_pitcher)

    base = history[PITCHER].map(pitcher_rate).astype(np.float64).fillna(prior)
    residual = history[TARGET].to_numpy(dtype=np.float64) - base.to_numpy(dtype=np.float64)
    cells = pd.DataFrame(
        {
            "cell": history[PITCHER].astype(str) + "|" + history[BATTER_HAND].astype(str),
            "residual": residual,
        }
    )
    cell_stats = cells.groupby("cell", observed=True)["residual"].agg(["sum", "size"])
    platoon_delta = cell_stats["sum"] / (cell_stats["size"] + k_platoon)
    return {
        "prior": float(prior),
        "k_pitcher": float(k_pitcher),
        "k_platoon": float(k_platoon),
        "pitcher_rate": {str(key): float(value) for key, value in pitcher_rate.items()},
        "platoon_delta": {str(key): float(value) for key, value in platoon_delta.items()},
        "platoon_n": {str(key): int(value) for key, value in cell_stats["size"].items()},
    }


def apply_platoon_features(frame: pd.DataFrame, state: dict[str, Any]) -> pd.DataFrame:
    """Per-row lookup against a frozen encoder. Never reads another row."""
    prior = float(state["prior"])
    pitcher_rate = state["pitcher_rate"]
    platoon_delta = state["platoon_delta"]
    platoon_n = state["platoon_n"]

    pitcher_key = frame[PITCHER].astype(str)
    cell_key = pitcher_key + "|" + frame[BATTER_HAND].astype(str)
    rate = pitcher_key.map(pitcher_rate)
    delta = cell_key.map(platoon_delta)
    count = cell_key.map(platoon_n)
    unseen = delta.isna().to_numpy()
    return pd.DataFrame(
        {
            "e30_pitcher_rate": rate.fillna(prior).to_numpy(dtype=np.float32),
            "e30_platoon_delta": delta.fillna(0.0).to_numpy(dtype=np.float32),
            "e30_platoon_n_log": np.log1p(
                count.fillna(0.0).to_numpy(dtype=np.float64)
            ).astype(np.float32),
            "e30_platoon_unseen": unseen.astype(np.int8),
        },
        index=frame.index,
    )


def platoon_states_before_each_season(
    history: pd.DataFrame, priors: dict[int, float], k_pitcher: float, k_platoon: float
) -> tuple[dict[int, dict], dict]:
    """Season-wise out-of-fold encoders plus the frozen full-history encoder.

    Training season s is encoded from seasons < s, mirroring the E14 pattern, so
    the training matrix never contains an encoding fitted on its own rows.
    """
    before: dict[int, dict] = {}
    seasons = sorted(int(value) for value in history[SEASON].unique())
    for season in seasons:
        slice_ = history.loc[history[SEASON] < season]
        before[season] = build_platoon_state(
            slice_, priors.get(season, 0.5), k_pitcher, k_platoon
        )
    final = build_platoon_state(
        history, float(history[TARGET].mean()), k_pitcher, k_platoon
    )
    return before, final


def build_platoon_frame(
    frame: pd.DataFrame, states_before: dict[int, dict], fallback: dict
) -> pd.DataFrame:
    """Apply the per-season encoder to each season block of `frame`."""
    parts = []
    for season, block in frame.groupby(SEASON, sort=True, observed=True):
        state = states_before.get(int(season), fallback)
        parts.append(apply_platoon_features(block, state))
    result = pd.concat(parts).reindex(frame.index)
    return result


def build_centered_platoon_state(
    history: pd.DataFrame,
    k: float,
    season_window: int | None,
) -> dict[str, Any]:
    """Estimate stable pitcher-by-batter-hand contrasts after regime centering."""
    if k <= 0:
        raise ValueError("--centered-platoon-k must be positive")
    source = history
    if season_window is not None and not history.empty:
        if season_window < 1:
            raise ValueError("--centered-platoon-window must be >= 1")
        cutoff = int(history[SEASON].max()) - season_window + 1
        source = history.loc[history[SEASON] >= cutoff]
    if source.empty:
        return {"delta": {}, "n": {}, "k": float(k), "seasons": []}
    centered = source[[SEASON, "game_type", PITCHER, BATTER_HAND, TARGET]].copy()
    local_mean = centered.groupby(
        [SEASON, "game_type", PITCHER], observed=True
    )[TARGET].transform("mean")
    centered["residual"] = centered[TARGET].to_numpy(dtype=np.float64) - local_mean
    cells = centered.groupby([PITCHER, BATTER_HAND], observed=True)["residual"].agg(
        ["sum", "size"]
    )
    delta = cells["sum"] / (cells["size"] + k)
    return {
        "delta": {
            f"{pitcher}|{hand}": float(value)
            for (pitcher, hand), value in delta.items()
        },
        "n": {
            f"{pitcher}|{hand}": int(value)
            for (pitcher, hand), value in cells["size"].items()
        },
        "k": float(k),
        "seasons": sorted(int(value) for value in source[SEASON].unique()),
    }


def centered_platoon_states_before_each_season(
    history: pd.DataFrame,
    k: float,
    season_window: int | None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    before: dict[int, dict[str, Any]] = {}
    for season in sorted(int(value) for value in history[SEASON].unique()):
        before[season] = build_centered_platoon_state(
            history.loc[history[SEASON] < season], k, season_window
        )
    final = build_centered_platoon_state(history, k, season_window)
    return before, final


def apply_centered_platoon_features(
    frame: pd.DataFrame, state: dict[str, Any]
) -> pd.DataFrame:
    key = frame[PITCHER].astype(str) + "|" + frame[BATTER_HAND].astype(str)
    delta = key.map(state["delta"])
    count = key.map(state["n"])
    return pd.DataFrame(
        {
            "e32_platoon_centered_delta": delta.fillna(0.0).to_numpy(dtype=np.float32),
            "e32_platoon_centered_n_log": np.log1p(
                count.fillna(0.0).to_numpy(dtype=np.float64)
            ).astype(np.float32),
            "e32_platoon_centered_unseen": delta.isna().to_numpy(dtype=np.int8),
        },
        index=frame.index,
    )


def build_centered_platoon_frame(
    frame: pd.DataFrame,
    states_before: dict[int, dict[str, Any]],
    fallback: dict[str, Any],
) -> pd.DataFrame:
    parts = []
    for season, block in frame.groupby(SEASON, sort=True, observed=True):
        parts.append(
            apply_centered_platoon_features(
                block, states_before.get(int(season), fallback)
            )
        )
    return pd.concat(parts).reindex(frame.index)


# --------------------------------------------------------------------------- #
# Booster wrappers shared by rolling evaluation and final packaging.
# --------------------------------------------------------------------------- #
class CategoricalFrameModel:
    """Adapter so LightGBM/CatBoost consume a DataFrame like an sklearn Pipeline."""

    def __init__(self, estimator, categorical: list[str], backend: str):
        self.estimator = estimator
        self.categorical = categorical
        self.backend = backend
        self.categories_: dict[str, pd.Index] = {}
        self.best_iteration_: int | None = None
        self.early_stopping_validation_rows_: int = 0

    def _prepare(self, frame: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        prepared = frame.copy()
        for column in self.categorical:
            if column not in prepared.columns:
                continue
            if self.backend in {"lgbm", "xgboost"}:
                if fitting:
                    self.categories_[column] = pd.Index(
                        sorted(pd.unique(prepared[column].dropna().astype(str)))
                    )
                prepared[column] = pd.Categorical(
                    prepared[column].astype(str), categories=self.categories_[column]
                )
            else:  # catboost rejects NaN inside categorical columns
                prepared[column] = (
                    prepared[column].astype("string")
                    .fillna("__missing__").astype(str)
                )
        return prepared

    def fit(
        self, X: pd.DataFrame, y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ):
        prepared = self._prepare(X, fitting=True)
        if self.backend == "catboost":
            present = [c for c in self.categorical if c in prepared.columns]
            self.estimator.fit(
                prepared, y, cat_features=present,
                sample_weight=sample_weight, verbose=False,
            )
        else:
            self.estimator.fit(prepared, y, sample_weight=sample_weight)
        return self

    @staticmethod
    def _brier_eval(y_true: np.ndarray, prediction: np.ndarray):
        return "brier", float(np.mean(np.square(prediction - y_true))), False

    def fit_time_ordered(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        eval_X: pd.DataFrame,
        eval_y: np.ndarray,
        refit_full: bool = True,
        refit_X: pd.DataFrame | None = None,
        refit_y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        eval_sample_weight: np.ndarray | None = None,
        refit_sample_weight: np.ndarray | None = None,
    ):
        """Choose iteration count on the last history season, then refit all history."""
        prepared = self._prepare(X, fitting=True)
        prepared_eval = self._prepare(eval_X, fitting=False)
        self.early_stopping_validation_rows_ = int(len(eval_X))
        if self.backend == "lgbm":
            from lightgbm import early_stopping, log_evaluation

            self.estimator.fit(
                prepared,
                y,
                sample_weight=sample_weight,
                eval_X=prepared_eval,
                eval_y=eval_y,
                eval_sample_weight=([eval_sample_weight] if eval_sample_weight is not None else None),
                eval_metric=self._brier_eval,
                callbacks=[
                    early_stopping(stopping_rounds=100, first_metric_only=True, verbose=False),
                    log_evaluation(0),
                ],
            )
            best = int(self.estimator.best_iteration_ or self.estimator.get_params()["n_estimators"])
            iteration_key = "n_estimators"
        else:
            from catboost import Pool

            present = [c for c in self.categorical if c in prepared.columns]
            eval_set = (
                Pool(
                    prepared_eval, label=eval_y, weight=eval_sample_weight,
                    cat_features=present,
                )
                if eval_sample_weight is not None
                else (prepared_eval, eval_y)
            )
            self.estimator.fit(
                prepared,
                y,
                cat_features=present,
                sample_weight=sample_weight,
                eval_set=eval_set,
                early_stopping_rounds=100,
                use_best_model=True,
                verbose=False,
            )
            best = max(1, int(self.estimator.get_best_iteration()) + 1)
            iteration_key = "iterations"
        self.best_iteration_ = best

        if refit_full:
            if (refit_X is None) != (refit_y is None):
                raise ValueError("refit_X and refit_y must be supplied together")
            full_x = refit_X if refit_X is not None else pd.concat([X, eval_X], axis=0)
            full_y = refit_y if refit_y is not None else np.concatenate([y, eval_y])
            full_weight = (
                refit_sample_weight
                if refit_sample_weight is not None
                else (
                    np.concatenate([sample_weight, eval_sample_weight])
                    if sample_weight is not None and eval_sample_weight is not None
                    else None
                )
            )
            estimator = clone(self.estimator)
            estimator.set_params(**{iteration_key: best})
            self.estimator = estimator
            self.categories_.clear()
            prepared_full = self._prepare(full_x, fitting=True)
            if self.backend == "catboost":
                present = [c for c in self.categorical if c in prepared_full.columns]
                self.estimator.fit(
                    prepared_full, full_y, cat_features=present,
                    sample_weight=full_weight, verbose=False,
                )
            else:
                self.estimator.fit(prepared_full, full_y, sample_weight=full_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self._prepare(X, fitting=False))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(self._prepare(X, fitting=False))

    @property
    def named_steps(self) -> dict[str, Any]:
        return {"clf": self.estimator}


class OrdinalNumericCatBoostModel(CategoricalFrameModel):
    """Encode categories as stable numeric codes before fitting CatBoost.

    This disables ordered target statistics while retaining the shared
    leakage-safe feature assembly and temporal fold implementation.
    """

    def __init__(self, estimator, encoding_columns: list[str]):
        super().__init__(estimator, [], "catboost")
        self.encoding_columns = encoding_columns

    def _prepare(self, frame: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        prepared = frame.copy()
        for column in self.encoding_columns:
            if column not in prepared.columns:
                continue
            values = (
                prepared[column]
                .astype("string")
                .fillna("__missing__")
                .astype(str)
            )
            if fitting:
                self.categories_[column] = pd.Index(sorted(pd.unique(values)))
            categories = self.categories_.get(column, pd.Index([]))
            prepared[column] = pd.Categorical(
                values, categories=categories
            ).codes.astype(np.int32)
        return prepared


def export_catboost_model(
    label: str,
    model: CategoricalFrameModel,
    feature_columns: list[str],
    kind: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Optionally persist a fitted CatBoost and its row-wise preprocessing spec."""
    destination = os.environ.get("V2_EXPORT_MODEL_DIR")
    if not destination or model.backend != "catboost":
        return
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    stem = label.replace("\\", "__").replace("/", "__").replace(":", "_")
    model_path = output / f"{stem}.cbm"
    spec_path = output / f"{stem}.json"
    model.estimator.save_model(model_path)
    spec: dict[str, Any] = {
        "label": label,
        "kind": kind,
        "model_file": model_path.name,
        "feature_columns": list(feature_columns),
        "categorical": list(model.categorical),
        "ordinal_encoding_columns": list(
            getattr(model, "encoding_columns", [])
        ),
        "ordinal_categories": {
            column: [str(value) for value in categories]
            for column, categories in model.categories_.items()
        },
    }
    spec.update(extra or {})
    spec_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[{label}] exported {model_path}", flush=True)


class TorchTabularModel:
    """Leakage-safe PyTorch adapter for compact tabular neural networks.

    The fitted artifact contains only train-derived numeric scalers and
    categorical vocabularies.  ``predict_proba`` transforms every row
    independently, so prediction does not depend on evaluation batch order,
    duplicates, or aggregate statistics.
    """

    def __init__(self, features: list[str], architecture: str, params: dict | None = None):
        self.features = list(features)
        self.architecture = architecture
        self.params = dict(params or {})
        self.categorical_: list[str] = []
        self.numeric_: list[str] = []
        self.categories_: dict[str, pd.Index] = {}
        self.numeric_median_: np.ndarray | None = None
        self.numeric_mean_: np.ndarray | None = None
        self.numeric_scale_: np.ndarray | None = None
        self.numeric_bins_: list[Any] | None = None
        self.model_: Any = None
        self.device_: str | None = None
        self.best_iteration_: int | None = None
        self.n_iter_: int | None = None
        self.early_stopping_validation_rows_: int = 0
        self.training_history_: list[dict[str, float]] = []

    @staticmethod
    def _torch():
        try:
            import torch
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "PyTorch is not installed. Install the pinned CUDA build from "
                "LOCAL_ENVIRONMENT.md before running deep models."
            ) from error
        return torch

    def _seed(self) -> None:
        torch = self._torch()
        seed = int(self.params.get("random_seed", RANDOM_SEED))
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _select_columns(self, frame: pd.DataFrame) -> None:
        from pandas.api.types import is_bool_dtype, is_numeric_dtype

        explicit = set(BOOSTER_CATEGORICAL)
        self.categorical_ = [
            column for column in self.features
            if column in explicit
            or is_bool_dtype(frame[column].dtype)
            or not is_numeric_dtype(frame[column].dtype)
        ]
        categorical = set(self.categorical_)
        self.numeric_ = [column for column in self.features if column not in categorical]

    @staticmethod
    def _categorical_values(series: pd.Series) -> pd.Series:
        return series.astype("string").fillna("__missing__").astype(str)

    def _fit_preprocessor(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        self._select_columns(frame)
        categorical_parts: list[np.ndarray] = []
        max_categories = int(self.params.get("max_categories", 100_000))
        min_category_count = int(self.params.get("min_category_count", 1))
        for column in self.categorical_:
            values = self._categorical_values(frame[column])
            counts = values.value_counts(sort=False)
            kept = counts[counts >= min_category_count]
            if len(kept) > max_categories:
                kept = kept.nlargest(max_categories)
            categories = pd.Index(sorted(kept.index.astype(str)))
            self.categories_[column] = categories
            codes = pd.Categorical(values, categories=categories).codes.astype(np.int64) + 1
            categorical_parts.append(codes)
        categorical = (
            np.column_stack(categorical_parts).astype(np.int64, copy=False)
            if categorical_parts else np.empty((len(frame), 0), dtype=np.int64)
        )

        if self.numeric_:
            numeric = frame[self.numeric_].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=np.float32, copy=True)
            numeric[~np.isfinite(numeric)] = np.nan
            median = np.nanmedian(numeric, axis=0).astype(np.float32)
            median[~np.isfinite(median)] = 0.0
            missing = np.where(np.isnan(numeric))
            numeric[missing] = median[missing[1]]
            mean = numeric.mean(axis=0, dtype=np.float64).astype(np.float32)
            scale = numeric.std(axis=0, dtype=np.float64).astype(np.float32)
            keep = np.isfinite(scale) & (scale >= 1e-6)
            if not np.all(keep):
                self.numeric_ = [
                    column for column, retained in zip(self.numeric_, keep)
                    if retained
                ]
                numeric = numeric[:, keep]
                median = median[keep]
                mean = mean[keep]
                scale = scale[keep]
            numeric = ((numeric - mean) / scale).astype(np.float32, copy=False)
            clip = float(self.params.get("numeric_clip", 10.0))
            np.clip(numeric, -clip, clip, out=numeric)
            self.numeric_median_, self.numeric_mean_, self.numeric_scale_ = (
                median, mean, scale
            )
            if self.architecture == "tabm_piecewise":
                torch = self._torch()
                from rtdl_num_embeddings import compute_bins

                maximum = int(self.params.get("bin_sample_rows", 200_000))
                if len(numeric) > maximum:
                    indices = np.linspace(
                        0, len(numeric) - 1, maximum, dtype=np.int64
                    )
                    bin_values = numeric[indices]
                else:
                    bin_values = numeric
                self.numeric_bins_ = compute_bins(
                    torch.from_numpy(np.ascontiguousarray(bin_values)),
                    n_bins=int(self.params.get("num_bins", 64)),
                )
        else:
            numeric = np.empty((len(frame), 0), dtype=np.float32)
            self.numeric_median_ = self.numeric_mean_ = self.numeric_scale_ = (
                np.empty(0, dtype=np.float32)
            )
            self.numeric_bins_ = None
        return numeric, categorical

    def _transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        categorical_parts: list[np.ndarray] = []
        for column in self.categorical_:
            values = self._categorical_values(frame[column])
            codes = pd.Categorical(
                values, categories=self.categories_[column]
            ).codes.astype(np.int64) + 1
            categorical_parts.append(codes)
        categorical = (
            np.column_stack(categorical_parts).astype(np.int64, copy=False)
            if categorical_parts else np.empty((len(frame), 0), dtype=np.int64)
        )
        if self.numeric_:
            numeric = frame[self.numeric_].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=np.float32, copy=True)
            numeric[~np.isfinite(numeric)] = np.nan
            missing = np.where(np.isnan(numeric))
            numeric[missing] = self.numeric_median_[missing[1]]
            numeric = (
                (numeric - self.numeric_mean_) / self.numeric_scale_
            ).astype(np.float32, copy=False)
            clip = float(self.params.get("numeric_clip", 10.0))
            np.clip(numeric, -clip, clip, out=numeric)
        else:
            numeric = np.empty((len(frame), 0), dtype=np.float32)
        return numeric, categorical

    def _build_network(self):
        torch = self._torch()
        nn = torch.nn
        cardinalities = [len(self.categories_[column]) + 1 for column in self.categorical_]
        n_numeric = len(self.numeric_)
        architecture = self.architecture
        params = self.params
        pitch_gated = bool(params.get("_pitch_gated", False))
        output_dim = (
            6
            if pitch_gated
            else int(params.get("_multilabel_outputs", params.get("_num_classes", 1)))
        )

        class EmbeddingMLP(nn.Module):
            def __init__(self):
                super().__init__()
                max_dim = int(params.get("embedding_dim", 24))
                dims = [
                    min(max_dim, max(4, int(round(1.6 * (cardinality ** 0.56)))))
                    for cardinality in cardinalities
                ]
                self.embeddings = nn.ModuleList([
                    nn.Embedding(cardinality, dim, padding_idx=0)
                    for cardinality, dim in zip(cardinalities, dims)
                ])
                for embedding in self.embeddings:
                    nn.init.normal_(embedding.weight, std=0.02)
                    with torch.no_grad():
                        embedding.weight[0].zero_()
                hidden = [int(value) for value in params.get("hidden_dims", [256, 128])]
                dropout = float(params.get("dropout", 0.15))
                layers: list[Any] = []
                width = n_numeric + sum(dims)
                for next_width in hidden:
                    layers.extend([
                        nn.Linear(width, next_width),
                        nn.BatchNorm1d(next_width),
                        nn.SiLU(),
                        nn.Dropout(dropout),
                    ])
                    width = next_width
                layers.append(nn.Linear(width, output_dim))
                self.network = nn.Sequential(*layers)

            def forward(self, numeric, categorical):
                pieces = [numeric]
                pieces.extend(
                    embedding(categorical[:, index])
                    for index, embedding in enumerate(self.embeddings)
                )
                output = self.network(torch.cat(pieces, dim=1))
                return output.squeeze(1) if output_dim == 1 else output

        class DeepFM(nn.Module):
            def __init__(self):
                super().__init__()
                dim = int(params.get("embedding_dim", 16))
                self.embeddings = nn.ModuleList([
                    nn.Embedding(cardinality, dim, padding_idx=0)
                    for cardinality in cardinalities
                ])
                self.first_order = nn.ModuleList([
                    nn.Embedding(cardinality, output_dim, padding_idx=0)
                    for cardinality in cardinalities
                ])
                for embedding in self.embeddings:
                    nn.init.normal_(embedding.weight, std=0.01)
                    with torch.no_grad():
                        embedding.weight[0].zero_()
                for embedding in self.first_order:
                    nn.init.zeros_(embedding.weight)
                self.numeric_linear = (
                    nn.Linear(n_numeric, output_dim) if n_numeric else None
                )
                hidden = [int(value) for value in params.get("hidden_dims", [256, 128])]
                dropout = float(params.get("dropout", 0.15))
                layers: list[Any] = []
                width = n_numeric + dim * len(cardinalities)
                for next_width in hidden:
                    layers.extend([
                        nn.Linear(width, next_width), nn.SiLU(), nn.Dropout(dropout)
                    ])
                    width = next_width
                layers.append(nn.Linear(width, output_dim))
                self.deep = nn.Sequential(*layers)
                self.bias = nn.Parameter(torch.zeros(output_dim))
                self.fm_projection = nn.Parameter(torch.ones(output_dim))

            def forward(self, numeric, categorical):
                embedded = [
                    embedding(categorical[:, index])
                    for index, embedding in enumerate(self.embeddings)
                ]
                deep_input = torch.cat([numeric, *embedded], dim=1)
                result = self.deep(deep_input) + self.bias
                if self.numeric_linear is not None:
                    result = result + self.numeric_linear(numeric)
                if embedded:
                    stack = torch.stack(embedded, dim=1)
                    fm = 0.5 * (
                        torch.square(stack.sum(dim=1))
                        - torch.square(stack).sum(dim=1)
                    ).sum(dim=1)
                    first = torch.stack([
                        embedding(categorical[:, index])
                        for index, embedding in enumerate(self.first_order)
                    ], dim=1).sum(dim=1)
                    result = result + (
                        float(params.get("fm_scale", 0.1))
                        * fm[:, None]
                        * self.fm_projection[None, :]
                    ) + first
                return result.squeeze(1) if output_dim == 1 else result

        class TabTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                dim = int(params.get("transformer_dim", 32))
                heads = int(params.get("transformer_heads", 4))
                layers_count = int(params.get("transformer_layers", 2))
                dropout = float(params.get("dropout", 0.1))
                self.embeddings = nn.ModuleList([
                    nn.Embedding(cardinality, dim, padding_idx=0)
                    for cardinality in cardinalities
                ])
                for embedding in self.embeddings:
                    nn.init.normal_(embedding.weight, std=0.02)
                    with torch.no_grad():
                        embedding.weight[0].zero_()
                self.field_embedding = nn.Parameter(
                    torch.zeros(1, max(1, len(cardinalities)), dim)
                )
                nn.init.normal_(self.field_embedding, std=0.02)
                if cardinalities:
                    layer = nn.TransformerEncoderLayer(
                        d_model=dim, nhead=heads, dim_feedforward=dim * 4,
                        dropout=dropout, activation="gelu", batch_first=True,
                        norm_first=True,
                    )
                    self.transformer = nn.TransformerEncoder(layer, layers_count)
                    transformed_width = dim * len(cardinalities)
                else:
                    self.transformer = None
                    transformed_width = 0
                hidden = [int(value) for value in params.get("hidden_dims", [256, 128])]
                blocks: list[Any] = []
                width = n_numeric + transformed_width
                for next_width in hidden:
                    blocks.extend([
                        nn.Linear(width, next_width), nn.LayerNorm(next_width),
                        nn.GELU(), nn.Dropout(dropout),
                    ])
                    width = next_width
                blocks.append(nn.Linear(width, output_dim))
                self.head = nn.Sequential(*blocks)

            def forward(self, numeric, categorical):
                pieces = [numeric]
                if self.transformer is not None:
                    tokens = torch.stack([
                        embedding(categorical[:, index])
                        for index, embedding in enumerate(self.embeddings)
                    ], dim=1)
                    tokens = self.transformer(
                        tokens + self.field_embedding[:, : tokens.shape[1]]
                    )
                    pieces.append(tokens.flatten(1))
                output = self.head(torch.cat(pieces, dim=1))
                return output.squeeze(1) if output_dim == 1 else output

        def build_tabm(embedding_kind: str):
            try:
                from tabm import TabM
            except ImportError as error:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "tabm is not installed. Install the pinned package from "
                    "LOCAL_ENVIRONMENT.md before running TabM."
                ) from error
            num_embeddings = None
            if embedding_kind == "periodic" and n_numeric:
                from rtdl_num_embeddings import PeriodicEmbeddings

                num_embeddings = PeriodicEmbeddings(
                    n_numeric,
                    d_embedding=int(params.get("num_embedding_dim", 16)),
                    n_frequencies=int(params.get("num_frequencies", 32)),
                    frequency_init_scale=float(
                        params.get("frequency_init_scale", 0.01)
                    ),
                    activation=True,
                    lite=bool(params.get("num_embedding_lite", True)),
                )
            elif embedding_kind == "piecewise" and n_numeric:
                from rtdl_num_embeddings import PiecewiseLinearEmbeddings

                if self.numeric_bins_ is None:
                    raise RuntimeError("Piecewise bins were not fitted")
                num_embeddings = PiecewiseLinearEmbeddings(
                    self.numeric_bins_,
                    d_embedding=int(params.get("num_embedding_dim", 16)),
                    activation=True,
                    version="B",
                )
            return TabM.make(
                n_num_features=n_numeric,
                cat_cardinalities=cardinalities,
                d_out=output_dim,
                num_embeddings=num_embeddings,
                arch_type=str(params.get("tabm_arch_type", "tabm")),
                k=int(params.get("tabm_k", 16)),
                n_blocks=int(params.get("tabm_blocks", 3)),
                d_block=int(params.get("tabm_width", 256)),
                dropout=float(params.get("dropout", 0.1)),
            )

        builders = {
            "deep_mlp": EmbeddingMLP,
            "deepfm": DeepFM,
            "tabtransformer": TabTransformer,
        }
        if architecture == "tabm":
            return build_tabm("none")
        if architecture == "tabm_pitch_gated":
            return build_tabm("none")
        if architecture == "tabm_periodic":
            return build_tabm("periodic")
        if architecture == "tabm_piecewise":
            return build_tabm("piecewise")
        if architecture not in builders:
            raise ValueError(f"Unknown torch architecture: {architecture}")
        return builders[architecture]()

    def _loss(self, logits, target, weight):
        torch = self._torch()
        if bool(self.params.get("_pitch_gated", False)):
            if target.ndim != 2 or target.shape[1] != 2:
                raise ValueError("pitch-gated target must be [control, group_code]")
            gate_logits = logits[..., :3]
            expert_logits = logits[..., 3:]
            gate_probability = torch.softmax(gate_logits, dim=-1)
            expert_probability = torch.sigmoid(expert_logits)
            final_probability = (gate_probability * expert_probability).sum(dim=-1)
            control_target = target[:, 0]
            group_target = target[:, 1].long()
            if logits.ndim == 3:
                ensemble_size = logits.shape[1]
                expanded_control = control_target[:, None].expand_as(final_probability)
                flat_gate = gate_logits.reshape(-1, 3)
                expanded_group = group_target[:, None].expand(
                    -1, ensemble_size
                ).reshape(-1)
                gate_loss = torch.nn.functional.cross_entropy(
                    flat_gate, expanded_group, reduction="none"
                ).reshape(-1, ensemble_size)
                selected_expert = expert_logits.gather(
                    -1,
                    group_target[:, None, None].expand(-1, ensemble_size, 1),
                ).squeeze(-1)
                row_weight = weight[:, None]
            else:
                ensemble_size = 1
                expanded_control = control_target
                gate_loss = torch.nn.functional.cross_entropy(
                    gate_logits, group_target, reduction="none"
                )
                selected_expert = expert_logits.gather(
                    -1, group_target[:, None]
                ).squeeze(-1)
                row_weight = weight
            stable_probability = final_probability.float().clamp(1e-6, 1.0 - 1e-6)
            final_logit = torch.logit(stable_probability)
            primary_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                final_logit, expanded_control.float(), reduction="none"
            )
            primary_brier = torch.square(
                stable_probability - expanded_control.float()
            )
            oracle_expert = torch.nn.functional.binary_cross_entropy_with_logits(
                selected_expert, expanded_control, reduction="none"
            )
            loss = (
                primary_bce
                + float(self.params.get("brier_weight", 1.0)) * primary_brier
                + float(self.params.get("pitch_gate_weight", 0.25)) * gate_loss
                + float(self.params.get("oracle_expert_weight", 0.5)) * oracle_expert
            )
            return (loss * row_weight).sum() / (
                weight.sum().clamp_min(1e-8) * float(ensemble_size)
            )
        multilabel_outputs = int(self.params.get("_multilabel_outputs", 0))
        multiclass = (
            multilabel_outputs == 0
            and int(self.params.get("_num_classes", 1)) > 1
        )
        if multiclass:
            mode = str(self.params.get("loss", "ce"))
            if logits.ndim == 3:
                batch_size, ensemble_size, class_count = logits.shape
                flat_logits = logits.reshape(-1, class_count)
                flat_target = target[:, None].expand(-1, ensemble_size).reshape(-1)
                cross_entropy = torch.nn.functional.cross_entropy(
                    flat_logits, flat_target, reduction="none"
                ).reshape(batch_size, ensemble_size)
            else:
                ensemble_size = 1
                cross_entropy = torch.nn.functional.cross_entropy(
                    logits, target, reduction="none"
                )
            if mode == "ce":
                loss = cross_entropy
            elif mode == "ce_brier":
                probabilities = torch.softmax(logits, dim=-1)
                one_hot = torch.nn.functional.one_hot(
                    target, num_classes=logits.shape[-1]
                ).to(probabilities.dtype)
                if logits.ndim == 3:
                    one_hot = one_hot[:, None, :]
                multiclass_brier = torch.square(
                    probabilities - one_hot
                ).sum(dim=-1)
                loss = cross_entropy + float(
                    self.params.get("brier_weight", 1.0)
                ) * multiclass_brier
            else:
                raise ValueError(f"Unknown multiclass deep loss: {mode}")
            if loss.ndim == 2:
                expanded_weight = weight[:, None]
                return (loss * expanded_weight).sum() / (
                    expanded_weight.sum() * ensemble_size
                ).clamp_min(1e-8)
            return (loss * weight).sum() / weight.sum().clamp_min(1e-8)
        if multilabel_outputs:
            if target.ndim != 2 or target.shape[1] != multilabel_outputs:
                raise ValueError(
                    "multilabel target shape does not match _multilabel_outputs"
                )
            if logits.ndim == 3:
                target = target[:, None, :].expand_as(logits)
                row_weight = weight[:, None, None]
                ensemble_size = logits.shape[1]
            elif logits.ndim == 2:
                row_weight = weight[:, None]
                ensemble_size = 1
            else:
                raise ValueError(f"unexpected multilabel logit shape: {logits.shape}")
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            brier = torch.square(torch.sigmoid(logits) - target)
            mode = str(self.params.get("loss", "bce"))
            if mode == "bce":
                loss = bce
            elif mode == "brier":
                loss = brier
            elif mode == "bce_brier":
                loss = bce + float(
                    self.params.get("brier_weight", 1.0)
                ) * brier
            else:
                raise ValueError(f"Unknown multilabel deep loss: {mode}")
            configured = self.params.get("multilabel_head_weights")
            if configured is None:
                configured = [1.0] * multilabel_outputs
            if len(configured) != multilabel_outputs:
                raise ValueError("multilabel_head_weights length mismatch")
            head_weight = torch.as_tensor(
                configured, dtype=logits.dtype, device=logits.device
            )
            weighted = loss * row_weight * head_weight
            denominator = (
                weight.sum().clamp_min(1e-8)
                * float(ensemble_size)
                * head_weight.sum().clamp_min(1e-8)
            )
            return weighted.sum() / denominator
        if logits.ndim == 3:
            logits = logits.squeeze(-1)
            target = target[:, None].expand_as(logits)
            weight = weight[:, None].expand_as(logits)
        mode = str(self.params.get("loss", "bce"))
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        brier = torch.square(torch.sigmoid(logits) - target)
        if mode == "bce":
            loss = bce
        elif mode == "brier":
            loss = brier
        elif mode == "bce_brier":
            loss = bce + float(self.params.get("brier_weight", 1.0)) * brier
        else:
            raise ValueError(f"Unknown deep loss: {mode}")
        return (loss * weight).sum() / weight.sum().clamp_min(1e-8)

    def _predict_arrays(
        self,
        numeric: np.ndarray,
        categorical: np.ndarray,
        preserve_ensemble: bool = False,
    ) -> np.ndarray:
        torch = self._torch()
        if self.model_ is None or self.device_ is None:
            raise RuntimeError("Model is not fitted")
        batch_size = int(self.params.get("predict_batch_size", 16_384))
        output: list[np.ndarray] = []
        self.model_.eval()
        use_amp = self.device_ == "cuda" and bool(self.params.get("amp", True))
        with torch.inference_mode():
            for start in range(0, len(numeric), batch_size):
                stop = min(len(numeric), start + batch_size)
                numeric_batch = torch.from_numpy(numeric[start:stop]).to(
                    self.device_, non_blocking=True
                )
                categorical_batch = torch.from_numpy(categorical[start:stop]).to(
                    self.device_, non_blocking=True
                )
                with torch.autocast(
                    device_type=self.device_, dtype=torch.float16, enabled=use_amp
                ):
                    logits = self.model_(numeric_batch, categorical_batch)
                pitch_gated = bool(self.params.get("_pitch_gated", False))
                if pitch_gated:
                    probabilities = (
                        torch.softmax(logits[..., :3], dim=-1)
                        * torch.sigmoid(logits[..., 3:])
                    ).sum(dim=-1)
                    if probabilities.ndim == 2 and not preserve_ensemble:
                        probabilities = probabilities.mean(dim=1)
                    output.append(probabilities.float().cpu().numpy())
                    continue
                multilabel = int(self.params.get("_multilabel_outputs", 0)) > 0
                multiclass = (
                    not multilabel
                    and int(self.params.get("_num_classes", 1)) > 1
                )
                if logits.ndim == 3:
                    probabilities = (
                        torch.softmax(logits, dim=-1)
                        if multiclass
                        else torch.sigmoid(logits.squeeze(-1))
                    )
                    if not preserve_ensemble:
                        probabilities = probabilities.mean(dim=1)
                else:
                    probabilities = (
                        torch.softmax(logits, dim=1)
                        if multiclass and logits.ndim == 2
                        else torch.sigmoid(logits)
                    )
                output.append(probabilities.float().cpu().numpy())
        return np.concatenate(output).astype(np.float64, copy=False)

    def _fit_arrays(
        self,
        train_numeric: np.ndarray,
        train_categorical: np.ndarray,
        train_y: np.ndarray,
        sample_weight: np.ndarray | None,
        epochs: int,
        eval_arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> int:
        torch = self._torch()
        self._seed()
        requested = str(self.params.get("device", "cuda"))
        self.device_ = "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"
        self.model_ = self._build_network().to(self.device_)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=float(self.params.get("learning_rate", 1e-3)),
            weight_decay=float(self.params.get("weight_decay", 1e-5)),
        )
        batch_size = int(self.params.get("batch_size", 4096))
        patience = int(self.params.get("early_stopping_patience", 3))
        clip_grad = float(self.params.get("clip_grad_norm", 5.0))
        use_amp = self.device_ == "cuda" and bool(self.params.get("amp", True))
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        multiclass = int(self.params.get("_num_classes", 1)) > 1
        y_values = np.asarray(
            train_y, dtype=(np.int64 if multiclass else np.float32)
        )
        weights = (
            np.asarray(sample_weight, dtype=np.float32)
            if sample_weight is not None else np.ones(len(train_y), dtype=np.float32)
        )
        rng = np.random.default_rng(int(self.params.get("random_seed", RANDOM_SEED)))
        best_epoch = epochs
        best_brier = np.inf
        best_state = None
        stale = 0
        self.training_history_ = []
        for epoch in range(1, epochs + 1):
            self.model_.train()
            order = rng.permutation(len(train_y))
            running_loss = 0.0
            seen = 0
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                numeric_batch = torch.from_numpy(train_numeric[indices]).to(
                    self.device_, non_blocking=True
                )
                categorical_batch = torch.from_numpy(train_categorical[indices]).to(
                    self.device_, non_blocking=True
                )
                target_batch = torch.from_numpy(y_values[indices]).to(
                    self.device_, non_blocking=True
                )
                weight_batch = torch.from_numpy(weights[indices]).to(
                    self.device_, non_blocking=True
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device_, dtype=torch.float16, enabled=use_amp
                ):
                    logits = self.model_(numeric_batch, categorical_batch)
                    loss = self._loss(logits, target_batch, weight_batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                running_loss += float(loss.detach().cpu()) * len(indices)
                seen += len(indices)
            record = {"epoch": float(epoch), "train_loss": running_loss / max(1, seen)}
            if eval_arrays is not None:
                eval_numeric, eval_categorical, eval_y = eval_arrays
                prediction = self._predict_arrays(eval_numeric, eval_categorical)
                if prediction.ndim == 2:
                    success_indices = [
                        int(value) for value in self.params.get("_success_indices", [])
                    ]
                    if not success_indices:
                        raise ValueError(
                            "Multiclass early stopping requires _success_indices"
                        )
                    prediction = prediction[:, success_indices].sum(axis=1)
                brier = float(np.mean(np.square(prediction - eval_y)))
                record["validation_brier"] = brier
                if brier < best_brier - float(self.params.get("early_stopping_min_delta", 1e-7)):
                    best_brier = brier
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model_.state_dict().items()
                    }
                    stale = 0
                else:
                    stale += 1
            self.training_history_.append(record)
            print(
                f"[torch/{self.architecture}] epoch={epoch}/{epochs} "
                f"loss={record['train_loss']:.6f}"
                + (
                    f" val_brier={record['validation_brier']:.8f}"
                    if "validation_brier" in record else ""
                ),
                flush=True,
            )
            if eval_arrays is not None and stale >= patience:
                break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.n_iter_ = int(best_epoch if eval_arrays is not None else epochs)
        return self.n_iter_

    def fit(
        self, X: pd.DataFrame, y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ):
        train_numeric, train_categorical = self._fit_preprocessor(X)
        epochs = int(self.params.get("epochs", 8))
        self._fit_arrays(
            train_numeric, train_categorical, y, sample_weight, epochs
        )
        self.best_iteration_ = self.n_iter_
        return self

    def fit_with_binary_eval(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        eval_X: pd.DataFrame,
        eval_binary_y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ):
        """Select an epoch on a declared outer development fold without refit."""
        train_numeric, train_categorical = self._fit_preprocessor(X)
        eval_numeric, eval_categorical = self._transform(eval_X)
        self.early_stopping_validation_rows_ = len(eval_X)
        epochs = int(self.params.get("epochs", 30))
        chosen = self._fit_arrays(
            train_numeric, train_categorical, y, sample_weight, epochs,
            (
                eval_numeric,
                eval_categorical,
                np.asarray(eval_binary_y, dtype=np.float32),
            ),
        )
        self.best_iteration_ = chosen
        self.n_iter_ = chosen
        return self

    def fit_time_ordered(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        eval_X: pd.DataFrame,
        eval_y: np.ndarray,
        refit_full: bool = True,
        refit_X: pd.DataFrame | None = None,
        refit_y: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        eval_sample_weight: np.ndarray | None = None,
        refit_sample_weight: np.ndarray | None = None,
    ):
        train_numeric, train_categorical = self._fit_preprocessor(X)
        eval_numeric, eval_categorical = self._transform(eval_X)
        self.early_stopping_validation_rows_ = len(eval_X)
        epochs = int(self.params.get("epochs", 20))
        chosen = self._fit_arrays(
            train_numeric, train_categorical, y, sample_weight, epochs,
            (eval_numeric, eval_categorical, np.asarray(eval_y, dtype=np.float32)),
        )
        selection_history = list(self.training_history_)
        if refit_full:
            if refit_X is None or refit_y is None:
                refit_X = pd.concat([X, eval_X], axis=0)
                refit_y = np.concatenate([y, eval_y])
                if sample_weight is not None and eval_sample_weight is not None:
                    refit_sample_weight = np.concatenate([
                        sample_weight, eval_sample_weight
                    ])
            self.categories_.clear()
            full_numeric, full_categorical = self._fit_preprocessor(refit_X)
            self._fit_arrays(
                full_numeric, full_categorical, refit_y,
                refit_sample_weight, chosen,
            )
            self.training_history_ = selection_history
        self.best_iteration_ = chosen
        self.n_iter_ = chosen
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        numeric, categorical = self._transform(X)
        positive = self._predict_arrays(numeric, categorical)
        if positive.ndim == 2:
            return positive
        return np.column_stack([1.0 - positive, positive])

    def predict_member_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return TabM probabilities before averaging its internal members."""
        if not self.architecture.startswith("tabm"):
            raise ValueError("Member probabilities are available only for TabM")
        numeric, categorical = self._transform(X)
        result = self._predict_arrays(
            numeric, categorical, preserve_ensemble=True
        )
        if result.ndim < 2:
            raise RuntimeError("TabM did not return an ensemble dimension")
        return result

    @property
    def named_steps(self) -> dict[str, Any]:
        return {"clf": self}

    def __del__(self):
        try:
            torch = self._torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def make_linear_v2(features: list[str], params: dict | None = None) -> Pipeline:
    params = dict(params or {})
    drop = set(params.pop("drop_collinear", []))
    missing_indicator = bool(params.pop("missing_indicator", False))
    min_frequency = int(params.pop("min_frequency", 10))
    model_features = [column for column in features if column not in drop]
    te_columns = [column for column in PITCHER_TE_FEATURES if column in model_features]
    categorical = [column for column in LINEAR_CATEGORICAL if column in model_features]
    numeric = [
        column for column in model_features
        if column not in categorical and column not in te_columns
    ]
    transformers: list[tuple[str, Any, list[str]]] = [
        (
            "numeric",
            Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=missing_indicator)),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ),
        (
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore", min_frequency=min_frequency, dtype=np.float32
                )),
            ]),
            categorical,
        ),
    ]
    if te_columns:
        transformers.append((
            "pitcher_te",
            TargetEncoder(
                target_type="binary", smooth="auto", cv=5,
                shuffle=True, random_state=RANDOM_SEED,
            ),
            te_columns,
        ))
    preprocessor = ColumnTransformer(transformers, sparse_threshold=1.0)
    settings = {
        "loss": "log_loss", "penalty": "l2", "alpha": 0.3,
        "learning_rate": "constant", "eta0": 0.001, "max_iter": 100,
        "tol": 1e-4, "average": True, "random_state": RANDOM_SEED, "n_jobs": -1,
    }
    settings.update(params)
    return Pipeline([("pre", preprocessor), ("clf", SGDClassifier(**settings))])


def make_hgb_v2(features: list[str], params: dict | None = None) -> Pipeline:
    params = dict(params or {})
    drop = set(HGB_DROPPED) | set(params.pop("drop_collinear", []))
    params.pop("missing_indicator", None)  # HGB handles NaN natively.
    model_features = [column for column in features if column not in drop]
    te_columns = [column for column in PITCHER_TE_FEATURES if column in model_features]
    categorical = [column for column in HGB_CATEGORICAL if column in model_features]
    numeric = [
        column for column in model_features
        if column not in categorical and column not in te_columns
    ]
    transformers: list[tuple[str, Any, list[str]]] = [
        (
            "cat",
            OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1,
                encoded_missing_value=-1,
            ),
            categorical,
        ),
        ("num", "passthrough", numeric),
    ]
    if te_columns:
        transformers.append((
            "pitcher_te",
            TargetEncoder(
                target_type="binary", smooth="auto", cv=5,
                shuffle=True, random_state=RANDOM_SEED,
            ),
            te_columns,
        ))
    categorical_mask = [True] * len(categorical) + [False] * (len(numeric) + len(te_columns))
    settings = {
        "loss": "log_loss", "learning_rate": 0.05, "max_iter": 250,
        "max_leaf_nodes": 31, "min_samples_leaf": 100, "l2_regularization": 5.0,
        "categorical_features": categorical_mask, "early_stopping": True,
        "validation_fraction": 0.1, "n_iter_no_change": 20,
        "random_state": RANDOM_SEED,
    }
    settings.update(params)
    return Pipeline([
        ("pre", ColumnTransformer(transformers)),
        ("clf", HistGradientBoostingClassifier(**settings)),
    ])


def make_lgbm(features: list[str], params: dict | None = None) -> CategoricalFrameModel:
    if _LGBMClassifier is None:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "lightgbm is not installed. See EXPERIMENT_PLAN_V2.md section 5.2."
        )
    settings = {
        "objective": "binary",
        "metric": "None",
        "learning_rate": 0.03,
        "n_estimators": 1200,
        "num_leaves": 127,
        "min_child_samples": 500,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 10.0,
        "n_jobs": 6,
        "random_state": RANDOM_SEED,
        "verbosity": -1,
    }
    settings.update(params or {})
    categorical = [c for c in BOOSTER_CATEGORICAL if c in features]
    return CategoricalFrameModel(_LGBMClassifier(**settings), categorical, "lgbm")


def make_catboost(
    features: list[str], params: dict | None = None
) -> CategoricalFrameModel:
    try:
        from catboost import CatBoostClassifier
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "catboost is not installed. Offload this stage with kaggle/offload.py."
        ) from error
    settings = {
        "loss_function": "Logloss",
        "eval_metric": "BrierScore",
        "learning_rate": 0.05,
        "iterations": 1500,
        "depth": 8,
        "l2_leaf_reg": 6.0,
        # Ordered target statistics plus greedy categorical combinations: this is
        # what can reconstruct pitcher x batter_hand without an explicit encoder.
        "max_ctr_complexity": 4,
        "random_seed": RANDOM_SEED,
        "allow_writing_files": False,
        "thread_count": 6,
        "task_type": "GPU" if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu" else "CPU",
    }
    settings.update(params or {})
    categorical = [c for c in BOOSTER_CATEGORICAL if c in features]
    return CategoricalFrameModel(CatBoostClassifier(**settings), categorical, "catboost")


def make_catboost_numeric(
    features: list[str], params: dict | None = None
) -> OrdinalNumericCatBoostModel:
    try:
        from catboost import CatBoostClassifier
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("catboost is required for catboost_numeric") from error
    settings = {
        "loss_function": "Logloss",
        "eval_metric": "BrierScore",
        "iterations": 1200,
        "depth": 6,
        "learning_rate": 0.02,
        "l2_leaf_reg": 100.0,
        "border_count": 32,
        "random_seed": RANDOM_SEED,
        "allow_writing_files": False,
        "thread_count": 11,
        "task_type": (
            "GPU"
            if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu"
            else "CPU"
        ),
    }
    settings.update(params or {})
    encoding_columns = [column for column in BOOSTER_CATEGORICAL if column in features]
    return OrdinalNumericCatBoostModel(
        CatBoostClassifier(**settings), encoding_columns
    )


def make_xgboost(
    features: list[str], params: dict | None = None
) -> CategoricalFrameModel:
    try:
        from xgboost import XGBClassifier
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "xgboost is not installed. Install the pinned version in requirements."
        ) from error
    settings: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": (
            "cuda" if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu" else "cpu"
        ),
        "enable_categorical": True,
        "max_cat_to_onehot": 4,
        "n_estimators": 1000,
        "max_depth": 8,
        "learning_rate": 0.03,
        "min_child_weight": 100.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 10.0,
        "reg_alpha": 0.0,
        "max_bin": 256,
        "random_state": RANDOM_SEED,
        "n_jobs": 6,
    }
    settings.update(params or {})
    categorical = [column for column in BOOSTER_CATEGORICAL if column in features]
    return CategoricalFrameModel(XGBClassifier(**settings), categorical, "xgboost")


def make_ebm(
    features: list[str], params: dict | None = None
) -> CategoricalFrameModel:
    """Build a regularized GA2M/EBM arm for A12.

    InterpretML accepts mixed pandas frames directly.  The shared adapter
    freezes categorical string handling between fit and predict, while all
    target-derived features have already been built with the outer-season
    cutoff before this factory is called.
    """
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "interpret-core is not installed. Install the pinned version from "
            "LOCAL_ENVIRONMENT.md before running the A12 EBM arm."
        ) from error
    settings: dict[str, Any] = {
        "max_bins": 128,
        "max_interaction_bins": 32,
        "interactions": 20,
        "validation_size": 0.1,
        "outer_bags": 4,
        "inner_bags": 0,
        "learning_rate": 0.03,
        "smoothing_rounds": 100,
        "interaction_smoothing_rounds": 100,
        "max_rounds": 5000,
        "early_stopping_rounds": 100,
        "early_stopping_tolerance": 1e-5,
        "min_samples_leaf": 100,
        "reg_lambda": 10.0,
        "max_leaves": 3,
        "n_jobs": 6,
        "random_state": RANDOM_SEED,
    }
    settings.update(params or {})
    categorical = [column for column in BOOSTER_CATEGORICAL if column in features]
    return CategoricalFrameModel(
        ExplainableBoostingClassifier(**settings), categorical, "ebm"
    )


def make_torch_tabular(
    features: list[str], architecture: str, params: dict | None = None
) -> TorchTabularModel:
    """Build one of the compact neural tabular arms used by the V4 plan."""
    return TorchTabularModel(features, architecture, params)


def model_factory(
    name: str, params: dict | None
) -> Callable[[list[str]], Any]:
    if name == "linear":
        return lambda features: make_linear_v2(features, params)
    if name == "hgb":
        return lambda features: make_hgb_v2(features, params)
    if name == "lgbm":
        return lambda features: make_lgbm(features, params)
    if name == "xgboost":
        return lambda features: make_xgboost(features, params)
    if name == "catboost":
        return lambda features: make_catboost(features, params)
    if name == "catboost_numeric":
        return lambda features: make_catboost_numeric(features, params)
    if name == "ebm":
        return lambda features: make_ebm(features, params)
    if name in {
        "deep_mlp", "deepfm", "tabtransformer", "tabm", "tabm_periodic",
        "tabm_piecewise",
    }:
        return lambda features: make_torch_tabular(features, name, params)
    raise ValueError(f"Unknown model: {name}")


# --------------------------------------------------------------------------- #
# Fold execution
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, help="Name used for output files.")
    parser.add_argument("--data", type=Path, default=ROOT / "open/data/train.csv")
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=["linear", "hgb"])
    parser.add_argument("--features", nargs="+", choices=FEATURE_CHOICES, default=["base", "e14"])
    parser.add_argument(
        "--feature-view",
        choices=FEATURE_VIEWS,
        default="all",
        help=(
            "A50 expert split applied after cutoff-correct feature assembly: "
            "application excludes player histories; behavioral keeps histories "
            "plus only essential count/regime context."
        ),
    )
    parser.add_argument("--validation-seasons", nargs="+", type=int, default=[2022, 2023, 2024])
    parser.add_argument(
        "--fit-game-types",
        nargs="+",
        choices=("R", "F"),
        default=None,
        help="Fit on selected historical game types while still predicting every validation row.",
    )
    parser.add_argument(
        "--fit-count-states",
        nargs="+",
        default=None,
        help="Fit on selected B-S states such as 0-0; validation rows remain complete.",
    )
    parser.add_argument("--prior-mode", default="r_recent3")
    parser.add_argument(
        "--inner-validation", choices=("all", "regular", "none"), default="all",
        help=(
            "Iteration selection for LightGBM/CatBoost. 'all' preserves V2; "
            "'regular' evaluates only R rows in the last history season and then "
            "refits every history row; 'none' uses the configured fixed iteration count."
        ),
    )
    parser.add_argument("--params", type=Path, default=None, help="JSON of booster params.")
    parser.add_argument("--blend", nargs="+", type=float, default=None,
                        help="Weights matching --models; omit to score each model alone.")
    parser.add_argument("--baseline-models", nargs="+", default=None,
                        help="Models forming the comparison baseline (default: first model).")
    parser.add_argument("--save-predictions", type=Path, default=ROOT / "experiments/results/predictions",
                        help="Directory for per-fold validation predictions (npz).")
    parser.add_argument("--baseline-stage", default=None,
                        help="Earlier stage whose saved prediction becomes the comparison baseline.")
    parser.add_argument("--baseline-key", default="blend",
                        help="Which entry of --baseline-stage to compare against.")
    parser.add_argument("--k-pitcher", type=float, default=PLATOON_K_PITCHER)
    parser.add_argument("--k-platoon", type=float, default=PLATOON_K_PLATOON)
    parser.add_argument("--component-k", type=float, default=120.0)
    parser.add_argument("--pitcher-profile-k", type=float, default=300.0)
    parser.add_argument("--batter-e14-k", type=float, default=200.0)
    parser.add_argument("--batter-middle-k", type=float, default=100.0)
    parser.add_argument("--pitchmix-k", type=float, default=100.0)
    parser.add_argument(
        "--trackman-window",
        type=int,
        default=None,
        help="Optional number of completed seasons retained in Trackman profiles.",
    )
    parser.add_argument(
        "--trackman-trend-window",
        type=int,
        default=2,
        help="Completed recent seasons contrasted with the full TrackMan profile.",
    )
    parser.add_argument("--trackman-platoon-k", type=float, default=200.0)
    parser.add_argument("--trackman-count-k", type=float, default=200.0)
    parser.add_argument("--history-group-k", type=float, default=500.0)
    parser.add_argument("--outcome-context-k", type=float, default=200.0)
    parser.add_argument(
        "--history-group-window",
        type=int,
        default=None,
        help="Optional number of completed seasons used by historical group rates.",
    )
    parser.add_argument(
        "--outcome-scheme",
        choices=OUTCOME_SCHEMES,
        default="five",
        help="Failure taxonomy used only by catboost_outcome.",
    )
    parser.add_argument(
        "--save-outcome-components",
        action="store_true",
        help="Persist each outcome-class probability beside the success sum.",
    )
    parser.add_argument(
        "--teacher-stage",
        default=None,
        help=(
            "Prediction artifact stem for catboost_teacher.  Only explicitly "
            "listed prior-season outer-OOF rows are used as soft targets."
        ),
    )
    parser.add_argument("--teacher-key", default="final_prediction")
    parser.add_argument("--teacher-years", nargs="+", type=int, default=None)
    parser.add_argument(
        "--teacher-anchor-stage",
        default=None,
        help=(
            "Optional prediction-artifact stem used as a temporal anchor. "
            "The model learns teacher minus anchor and adds the validation anchor."
        ),
    )
    parser.add_argument("--teacher-anchor-key", default="final_prediction")
    parser.add_argument(
        "--teacher-center",
        choices=("none", "year", "year_game_type"),
        default="none",
        help=(
            "Center anchor residuals within each historical teacher fold; "
            "no validation/test aggregate is read."
        ),
    )
    parser.add_argument(
        "--teacher-residual-output",
        action="store_true",
        help=(
            "When a teacher anchor is supplied, fit the centered residual but "
            "return it around 0.5 instead of loading and adding a validation "
            "anchor.  This supports leakage-safe residual directions whose "
            "deployment parent is supplied by a separate frozen artifact."
        ),
    )
    parser.add_argument(
        "--teacher-alpha",
        type=float,
        default=0.0,
        help="Supervised target fraction mixed into the outer-OOF teacher target.",
    )
    parser.add_argument(
        "--teacher-fill-hard-labels",
        action="store_true",
        help=(
            "Train catboost_teacher on the full historical slice: rows with an "
            "outer-OOF teacher use the mixed soft target and all remaining rows "
            "retain their official hard label.  Incompatible with teacher anchors."
        ),
    )
    parser.add_argument("--e14-k", type=float, default=E14_K)
    parser.add_argument("--centered-platoon-k", type=float, default=100.0)
    parser.add_argument("--centered-platoon-window", type=int, default=None)
    parser.add_argument(
        "--drop-features",
        nargs="+",
        default=None,
        help="Ablation-only list of assembled model columns to remove.",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--history-window", type=int, default=None,
        help="Fit the model on only the latest N seasons; frozen state still uses all history.",
    )
    parser.add_argument(
        "--f-regime-start", type=int, default=None,
        help=(
            "For targets after this year, fit F rows only from this season onward. "
            "R rows and frozen state are unaffected."
        ),
    )
    parser.add_argument(
        "--season-decay", type=float, default=1.0,
        help="Per-season multiplicative sample weight in (0, 1]; latest history season has weight 1.",
    )
    parser.add_argument(
        "--f-pre-regime-weight", type=float, default=0.0,
        help="Weight for F rows before --f-regime-start; 0 keeps the hard-filter behavior.",
    )
    parser.add_argument("--max-history-rows", type=int, default=None, help="Smoke test only.")
    parser.add_argument("--max-valid-rows", type=int, default=None, help="Smoke test only.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/results")
    return parser.parse_args()


def subsample(frame: pd.DataFrame, maximum: int | None) -> pd.DataFrame:
    if maximum is None or len(frame) <= maximum:
        return frame
    return frame.sample(n=maximum, random_state=RANDOM_SEED).sort_index()


def assemble(
    frame: pd.DataFrame,
    e14: pd.DataFrame | None,
    e14_multi: pd.DataFrame | None,
    platoon: pd.DataFrame | None,
    pitcher_te: pd.DataFrame | None,
    trackman: pd.DataFrame | None,
    e22_probs: pd.DataFrame | None,
    components: pd.DataFrame | None,
    centered_platoon: pd.DataFrame | None,
    pitcher_hand_category: pd.DataFrame | None,
    f_regime: pd.DataFrame | None,
    hand_matchup: pd.DataFrame | None,
    count_state: pd.DataFrame | None,
    type_count: pd.DataFrame | None,
    type_month: pd.DataFrame | None,
    e14_hand_cells: pd.DataFrame | None,
    rate_hand_cells: pd.DataFrame | None,
    e14_hand_bins: pd.DataFrame | None,
    team_matchup: pd.DataFrame | None,
) -> pd.DataFrame:
    parts = [frame[BASE_FEATURES]]
    if e14 is not None:
        parts.append(e14)
    if e14_multi is not None:
        parts.append(e14_multi)
    if platoon is not None:
        parts.append(platoon)
    if pitcher_te is not None:
        parts.append(pitcher_te)
    if trackman is not None:
        parts.append(trackman)
    if e22_probs is not None:
        parts.append(e22_probs)
    if components is not None:
        parts.append(components)
    if centered_platoon is not None:
        parts.append(centered_platoon)
    if pitcher_hand_category is not None:
        parts.append(pitcher_hand_category)
    if f_regime is not None:
        parts.append(f_regime)
    if hand_matchup is not None:
        parts.append(hand_matchup)
    if count_state is not None:
        parts.append(count_state)
    if type_count is not None:
        parts.append(type_count)
    if type_month is not None:
        parts.append(type_month)
    if e14_hand_cells is not None:
        parts.append(e14_hand_cells)
    if rate_hand_cells is not None:
        parts.append(rate_hand_cells)
    if e14_hand_bins is not None:
        parts.append(e14_hand_bins)
    if team_matchup is not None:
        parts.append(team_matchup)
    return pd.concat(parts, axis=1)


def _prepare_tabicl_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve categorical semantics while bounding numeric memory for TabICL."""
    prepared = frame.copy()
    for column in prepared.columns:
        series = prepared[column]
        if column in BOOSTER_CATEGORICAL or not pd.api.types.is_numeric_dtype(series):
            prepared[column] = (
                series.astype("string").fillna("__MISSING__").astype(object)
            )
        else:
            prepared[column] = pd.to_numeric(series, errors="coerce").astype(np.float32)
    return prepared


def _fixed_tabicl_predict(
    model: Any,
    frame: pd.DataFrame,
    frozen_pad_row: pd.DataFrame,
    query_rows: int,
    label: str,
) -> np.ndarray:
    """Predict with a fixed query shape and a frozen history-only pad row.

    TabICLv2's row attention uses training rows as keys/values, but different
    query lengths select different CUDA kernels and produced non-negligible
    rounding changes in the mandatory invariance smoke.  Fixing every query
    call to the same length makes single/batch/shuffle/duplicate predictions
    bit-identical.  Padding is copied from a frozen training feature row.
    """
    if query_rows < 1:
        raise ValueError("tabicl fixed_query_rows must be >= 1")
    outputs: list[np.ndarray] = []
    total_blocks = (len(frame) + query_rows - 1) // query_rows
    started = time.perf_counter()
    for block, start in enumerate(range(0, len(frame), query_rows), start=1):
        chunk = frame.iloc[start : start + query_rows].reset_index(drop=True)
        real_rows = len(chunk)
        if real_rows < query_rows:
            padding = pd.concat(
                [frozen_pad_row] * (query_rows - real_rows), ignore_index=True
            )
            chunk = pd.concat([chunk, padding], ignore_index=True)
        outputs.append(
            model.predict_proba(chunk)[:real_rows, 1].astype(np.float64)
        )
        if block == 1 or block == total_blocks or block % 100 == 0:
            print(
                f"[{label}] query block {block:,}/{total_blocks:,}, "
                f"rows={min(start + real_rows, len(frame)):,}/{len(frame):,}, "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float64)


def fit_tabicl_model(
    label: str,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: pd.DataFrame,
    history_seasons: pd.Series,
    params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a synthetic-pretrained TabICLv2 context and predict row-independently."""
    if os.environ.get("V2_PRELOAD_TORCH", "0") != "1":
        raise RuntimeError(
            "tabicl requires V2_PRELOAD_TORCH=1 so CUDA DLLs load before NumPy/pandas"
        )
    isolated_site = ROOT / "experiments" / "_tabicl_site"
    if not isolated_site.is_dir():
        raise FileNotFoundError(f"Isolated TabICL runtime not found: {isolated_site}")
    if str(isolated_site) not in sys.path:
        sys.path.insert(0, str(isolated_site))

    import hashlib
    import importlib.metadata
    import torch
    from tabicl import TabICLClassifier

    config = dict(params or {})
    max_context_rows = int(config.pop("max_context_rows", 16_384))
    context_strategy = str(config.pop("context_strategy", "latest_season_random"))
    fixed_query_rows = int(config.pop("fixed_query_rows", 256))
    n_estimators = int(config.pop("n_estimators", 2))
    batch_size = int(config.pop("batch_size", 1))
    random_state = int(config.pop("random_state", RANDOM_SEED))
    softmax_temperature = float(config.pop("softmax_temperature", 0.9))
    average_logits = bool(config.pop("average_logits", True))
    checkpoint_version = str(
        config.pop("checkpoint_version", "tabicl-classifier-v2-20260212.ckpt")
    )
    checkpoint_path = ROOT / str(
        config.pop(
            "checkpoint_path",
            "experiments/_cache/tabicl/tabiclv2_classifier.ckpt",
        )
    )
    offload_mode = str(config.pop("offload_mode", "auto"))
    offload_dir = ROOT / str(
        config.pop("disk_offload_dir", "experiments/_cache/tabicl/offload")
    )
    use_amp = config.pop("use_amp", "auto")
    use_fa3 = bool(config.pop("use_fa3", False))
    n_jobs = int(config.pop("n_jobs", 6))
    if config:
        raise ValueError(f"Unknown tabicl params: {sorted(config)}")
    if context_strategy != "latest_season_random":
        raise ValueError(
            "Only preregistered tabicl context_strategy=latest_season_random is allowed"
        )
    if max_context_rows < 300:
        raise ValueError("tabicl max_context_rows must be >= 300")

    season_values = history_seasons.to_numpy(dtype=np.int16, copy=False)
    rng = np.random.default_rng(random_state)
    selected_parts: list[np.ndarray] = []
    remaining = min(max_context_rows, len(train_x))
    for context_season in sorted(np.unique(season_values), reverse=True):
        positions = np.flatnonzero(season_values == context_season)
        if len(positions) > remaining:
            positions = np.sort(rng.choice(positions, size=remaining, replace=False))
        selected_parts.append(positions)
        remaining -= len(positions)
        if remaining == 0:
            break
    selected = np.sort(np.concatenate(selected_parts))
    context_x = _prepare_tabicl_frame(train_x.iloc[selected].reset_index(drop=True))
    context_y = np.asarray(train_y[selected], dtype=np.int8)
    prepared_valid = _prepare_tabicl_frame(valid_x.reset_index(drop=True))
    frozen_pad_row = context_x.iloc[[0]].copy()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    offload_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[{label}] context={len(context_x):,}/{len(train_x):,}, "
        f"valid={len(prepared_valid):,}, features={context_x.shape[1]}, "
        f"estimators={n_estimators}, fixed_query_rows={fixed_query_rows}",
        flush=True,
    )
    torch.manual_seed(random_state)
    torch.cuda.reset_peak_memory_stats()
    model = TabICLClassifier(
        n_estimators=n_estimators,
        batch_size=batch_size,
        kv_cache=True,
        model_path=checkpoint_path,
        allow_auto_download=False,
        checkpoint_version=checkpoint_version,
        device="cuda",
        use_amp=use_amp,
        use_fa3=use_fa3,
        offload_mode=offload_mode,
        disk_offload_dir=str(offload_dir),
        softmax_temperature=softmax_temperature,
        average_logits=average_logits,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=False,
    )
    fit_started = time.perf_counter()
    model.fit(context_x, context_y)
    fit_seconds = time.perf_counter() - fit_started
    predict_started = time.perf_counter()
    prediction = _fixed_tabicl_predict(
        model,
        prepared_valid,
        frozen_pad_row,
        fixed_query_rows,
        label,
    )
    predict_seconds = time.perf_counter() - predict_started

    sentinel = _fixed_tabicl_predict(
        model,
        prepared_valid.iloc[[0]],
        frozen_pad_row,
        fixed_query_rows,
        f"{label}/invariance",
    )[0]
    invariance_delta = float(abs(sentinel - prediction[0]))
    if invariance_delta > 2e-6:
        raise RuntimeError(
            f"TabICL fixed-query row invariance failed: {invariance_delta:.9g}"
        )

    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    unique_context_seasons, season_counts = np.unique(
        season_values[selected], return_counts=True
    )
    details = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "prediction_std": float(np.std(prediction)),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_min": float(np.min(prediction)),
        "prediction_max": float(np.max(prediction)),
        "context_rows": int(len(context_x)),
        "context_target_mean": float(np.mean(context_y)),
        "context_strategy": context_strategy,
        "context_season_rows": {
            str(int(season)): int(count)
            for season, count in zip(unique_context_seasons, season_counts)
        },
        "feature_count": int(context_x.shape[1]),
        "n_estimators": n_estimators,
        "batch_size": batch_size,
        "fixed_query_rows": fixed_query_rows,
        "fixed_pad_source": "first selected history feature row",
        "row_independent_attention": "queries attend only to frozen training keys/values",
        "fixed_query_invariance_max_abs": invariance_delta,
        "direct_variable_length_api_prohibited": True,
        "checkpoint_version": checkpoint_version,
        "checkpoint_sha256": digest.hexdigest(),
        "checkpoint_bytes": int(checkpoint_path.stat().st_size),
        "tabicl_version": importlib.metadata.version("tabicl"),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "synthetic_pretrained_checkpoint": True,
        "validation_target_used_for_fit_or_selection": False,
    }
    del model, context_x, prepared_valid
    gc.collect()
    torch.cuda.empty_cache()
    return prediction, details


def fit_model(
    label: str,
    factory: Callable[[list[str]], Any],
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: pd.DataFrame,
    history_seasons: pd.Series,
    history_game_type: pd.Series,
    inner_validation: str,
    train_weight: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one model, using a chronological inner validation for boosters."""
    print(f"[{label}] fit={len(train_x):,} rows, features={train_x.shape[1]}", flush=True)
    model = factory(list(train_x.columns))
    started = time.perf_counter()
    inner_season: int | None = None
    if (
        isinstance(model, (CategoricalFrameModel, TorchTabularModel))
        and inner_validation != "none"
        and history_seasons.nunique() >= 2
    ):
        inner_season = int(history_seasons.max())
        earlier = history_seasons.to_numpy() < inner_season
        inner_valid = history_seasons.to_numpy() == inner_season
        if inner_validation == "regular":
            inner_valid &= history_game_type.astype(str).to_numpy() == "R"
        if not np.any(earlier) or not np.any(inner_valid):
            raise ValueError(
                f"Empty chronological inner split: season={inner_season}, "
                f"mode={inner_validation}"
            )
        model.fit_time_ordered(
            train_x.loc[earlier], train_y[earlier],
            train_x.loc[inner_valid], train_y[inner_valid], refit_full=True,
            refit_X=train_x, refit_y=train_y,
            sample_weight=(train_weight[earlier] if train_weight is not None else None),
            eval_sample_weight=(train_weight[inner_valid] if train_weight is not None else None),
            refit_sample_weight=train_weight,
        )
    else:
        if isinstance(model, (CategoricalFrameModel, TorchTabularModel)):
            model.fit(train_x, train_y, sample_weight=train_weight)
        elif train_weight is not None:
            model.fit(train_x, train_y, clf__sample_weight=train_weight)
        else:
            model.fit(train_x, train_y)
    fit_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    prediction = model.predict_proba(valid_x)[:, 1].astype(np.float64)
    predict_seconds = time.perf_counter() - prediction_started
    classifier = model.named_steps["clf"]
    iteration = getattr(model, "best_iteration_", None) or getattr(classifier, "n_iter_", None)
    if isinstance(iteration, np.ndarray):
        iteration = int(np.max(iteration))
    elif iteration is not None:
        iteration = int(iteration)
    details = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_iter": iteration,
        "prediction_std": float(np.std(prediction)),
        "early_stopping_validation_season": inner_season,
        "inner_validation_mode": inner_validation,
        "early_stopping_validation_rows": int(
            getattr(model, "early_stopping_validation_rows_", 0)
        ),
        "refit_full_history": bool(inner_season is not None),
        "sample_weighted": bool(train_weight is not None),
    }
    if isinstance(model, TorchTabularModel):
        details.update({
            "architecture": model.architecture,
            "device": model.device_,
            "categorical_features": list(model.categorical_),
            "numeric_feature_count": len(model.numeric_),
            "categorical_cardinalities": {
                column: int(len(categories) + 1)
                for column, categories in model.categories_.items()
            },
            "training_history": model.training_history_,
            "row_independent_preprocessing": True,
        })
    if hasattr(classifier, "get_feature_importance"):
        importance = np.asarray(classifier.get_feature_importance(), dtype=np.float64)
        if len(importance) == len(train_x.columns):
            details["feature_importance"] = [
                {"feature": feature, "importance": float(value)}
                for feature, value in sorted(
                    zip(train_x.columns, importance),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
    if isinstance(model, CategoricalFrameModel):
        classes = [str(value) for value in getattr(model.estimator, "classes_", [0, 1])]
        positive_indices = [
            index for index, value in enumerate(classes)
            if value in {"1", "1.0", "True", "true"}
        ]
        if not positive_indices and len(classes) == 2:
            positive_indices = [1]
        export_catboost_model(
            label,
            model,
            list(train_x.columns),
            "binary_classifier",
            {
                "classes": classes,
                "positive_indices": positive_indices,
            },
        )
    del model
    gc.collect()
    return prediction, details


def derive_control_outcome_labels(
    frame: pd.DataFrame, scheme: str = "five"
) -> pd.Series:
    """Recover auxiliary training outcomes from official as-of counters.

    The next row is read only inside the supplied historical training slice and
    only to reconstruct the current historical pitch's label.  Boundary rows
    and mechanically inconsistent cases are excluded.  A failed pitch can
    increment both provided failure counters, so that overlap is retained as
    its own ``reverse_middle`` class instead of discarding valid history.
    """
    n = frame["asof_pitcher_n"].fillna(0).to_numpy(dtype=np.int64, copy=False)
    reverse_count = np.rint(
        frame["asof_pitcher_reverse_rate"].fillna(0.0).to_numpy(dtype=np.float64) * n
    ).astype(np.int64)
    middle_count = np.rint(
        frame["asof_pitcher_middle_rate"].fillna(0.0).to_numpy(dtype=np.float64) * n
    ).astype(np.int64)
    ball_count = np.rint(
        frame["asof_pitcher_ball_rate"].fillna(0.0).to_numpy(dtype=np.float64) * n
    ).astype(np.int64)
    strike_count = np.rint(
        frame["asof_pitcher_strike_rate"].fillna(0.0).to_numpy(dtype=np.float64) * n
    ).astype(np.int64)
    work = pd.DataFrame(
        {
            "pitcher": frame[PITCHER].to_numpy(),
            "n": n,
            "reverse": reverse_count,
            "middle": middle_count,
            "ball": ball_count,
            "strike": strike_count,
        },
        index=frame.index,
    )
    grouped = work.groupby("pitcher", sort=False, observed=True)
    next_n = grouped["n"].shift(-1)
    reverse_event = grouped["reverse"].shift(-1) - work["reverse"]
    middle_event = grouped["middle"].shift(-1) - work["middle"]
    ball_event = grouped["ball"].shift(-1) - work["ball"]
    strike_event = grouped["strike"].shift(-1) - work["strike"]
    consecutive = next_n.eq(work["n"] + 1)
    reverse_zero_one = reverse_event.isin([0.0, 1.0])
    middle_zero_one = middle_event.isin([0.0, 1.0])
    ball_zero_one = ball_event.isin([0.0, 1.0])
    strike_zero_one = strike_event.isin([0.0, 1.0])
    success = frame[TARGET].eq(1)
    failure = ~success
    labels = pd.Series(pd.NA, index=frame.index, dtype="string")
    valid = consecutive & reverse_zero_one & middle_zero_one
    labels.loc[valid & success & reverse_event.eq(0) & middle_event.eq(0)] = "success"
    reverse_only = valid & failure & reverse_event.eq(1) & middle_event.eq(0)
    middle_only = valid & failure & reverse_event.eq(0) & middle_event.eq(1)
    overlap = valid & failure & reverse_event.eq(1) & middle_event.eq(1)
    wide = valid & failure & reverse_event.eq(0) & middle_event.eq(0)
    contextual_scheme = scheme in {
        "success_count", "all_count", "success_type", "all_type", "binary_count",
        "reverse_count", "middle_count", "wide_count", "reverse_type",
        "reverse_hand", "all_hand", "success_call", "all_call",
        "component15", "component15_type", "component15_count",
    }
    base_scheme = "reverse_any" if contextual_scheme else scheme
    if base_scheme in {"five", "drop_overlap", "reverse_any", "middle_any"}:
        labels.loc[reverse_only] = "reverse"
        labels.loc[middle_only] = "middle"
        labels.loc[wide] = "wide"
        if base_scheme == "five":
            labels.loc[overlap] = "reverse_middle"
        elif base_scheme == "reverse_any":
            labels.loc[overlap] = "reverse"
        elif base_scheme == "middle_any":
            labels.loc[overlap] = "middle"
    elif base_scheme == "coarse3":
        labels.loc[reverse_only | overlap] = "reverse"
        labels.loc[middle_only | wide] = "other_failure"
    else:  # Guard direct callers as well as argparse users.
        raise ValueError(f"Unknown outcome scheme: {scheme}")
    if contextual_scheme:
        count_suffix = (
            frame["balls_before"].astype(str)
            + "-"
            + frame["strikes_before"].astype(str)
        )
        type_suffix = frame["game_type"].astype(str)
        if scheme == "binary_count":
            labels.loc[valid & failure] = "failure"
            labels.loc[labels.notna()] = (
                labels.loc[labels.notna()] + "|" + count_suffix.loc[labels.notna()]
            )
        elif scheme == "success_count":
            mask = labels.eq("success").fillna(False)
            labels.loc[mask] = labels.loc[mask] + "|" + count_suffix.loc[mask]
        elif scheme == "all_count":
            mask = labels.notna()
            labels.loc[mask] = labels.loc[mask] + "|" + count_suffix.loc[mask]
        elif scheme == "success_type":
            mask = labels.eq("success").fillna(False)
            labels.loc[mask] = labels.loc[mask] + "|" + type_suffix.loc[mask]
        elif scheme == "all_type":
            mask = labels.notna()
            labels.loc[mask] = labels.loc[mask] + "|" + type_suffix.loc[mask]
        elif scheme in {"reverse_count", "middle_count", "wide_count"}:
            component = scheme.removesuffix("_count")
            mask = labels.eq(component).fillna(False)
            labels.loc[mask] = labels.loc[mask] + "|" + count_suffix.loc[mask]
        elif scheme == "reverse_type":
            mask = labels.eq("reverse").fillna(False)
            labels.loc[mask] = labels.loc[mask] + "|" + type_suffix.loc[mask]
        elif scheme in {"reverse_hand", "all_hand"}:
            hand_suffix = (
                frame["pitcher_hand"].astype(str)
                + "-"
                + frame["batter_hand"].astype(str)
            )
            mask = (
                labels.eq("reverse").fillna(False)
                if scheme == "reverse_hand"
                else labels.notna()
            )
            labels.loc[mask] = labels.loc[mask] + "|" + hand_suffix.loc[mask]
        elif scheme in {"success_call", "all_call"}:
            # The official cumulative ball/strike counters expose a stable
            # three-way pitch-result subtype.  Split the auxiliary outcome,
            # while the final probability remains the sum of all success
            # subclasses.  Counter deltas are reconstructed only for historic
            # training rows; validation/test labels are never inspected.
            component_valid = valid & ball_zero_one & strike_zero_one
            labels.loc[~component_valid] = pd.NA
            call_suffix = pd.Series("other", index=frame.index, dtype="string")
            call_suffix.loc[ball_event.eq(1.0)] = "ball"
            call_suffix.loc[strike_event.eq(1.0)] = "strike"
            mask = (
                labels.eq("success").fillna(False)
                if scheme == "success_call"
                else labels.notna()
            )
            labels.loc[mask] = labels.loc[mask] + "|" + call_suffix.loc[mask]
        elif scheme in {"component15", "component15_type", "component15_count"}:
            # Across the official history there are 15 observed binary
            # reverse/middle/ball/strike patterns.  Modelling the exact pattern
            # supplies substantially richer supervision than the previous
            # reverse/middle-only taxonomy without changing inference inputs.
            component_valid = valid & ball_zero_one & strike_zero_one
            labels.loc[~component_valid] = pd.NA
            prefix = pd.Series("failure", index=frame.index, dtype="string")
            prefix.loc[success & component_valid] = "success"
            suffix = (
                "r" + reverse_event.fillna(-1).astype(int).astype(str)
                + "m" + middle_event.fillna(-1).astype(int).astype(str)
                + "b" + ball_event.fillna(-1).astype(int).astype(str)
                + "s" + strike_event.fillna(-1).astype(int).astype(str)
            )
            labels.loc[component_valid] = (
                prefix.loc[component_valid] + "|" + suffix.loc[component_valid]
            )
            if scheme == "component15_type":
                labels.loc[component_valid] = (
                    labels.loc[component_valid]
                    + "|type="
                    + frame.loc[component_valid, "game_type"].astype(str)
                )
            elif scheme == "component15_count":
                labels.loc[component_valid] = (
                    labels.loc[component_valid]
                    + "|count="
                    + count_suffix.loc[component_valid]
                )
    return labels


def fit_outcome_model(
    label: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    valid_x: pd.DataFrame,
    params: dict[str, Any] | None = None,
    outcome_scheme: str = "five",
    train_weight: np.ndarray | None = None,
    outcome_labels: pd.Series | None = None,
    save_components: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    from catboost import CatBoostClassifier

    outcome = (
        derive_control_outcome_labels(history, outcome_scheme)
        if outcome_labels is None
        else outcome_labels.reindex(history.index)
    )
    usable = outcome.notna().to_numpy(dtype=bool)
    custom_params = dict(params or {})
    season_class_balance = custom_params.pop("season_class_balance", None)
    season_class_balance_clip = float(
        custom_params.pop("season_class_balance_clip", 4.0)
    )
    settings = {
        "loss_function": "MultiClass",
        "iterations": 400,
        "depth": 7,
        "learning_rate": 0.05,
        "l2_leaf_reg": 12.0,
        "random_seed": RANDOM_SEED,
        "allow_writing_files": False,
        "thread_count": 6,
        "task_type": (
            "GPU" if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu" else "CPU"
        ),
    }
    settings.update(custom_params)
    categorical = [column for column in BOOSTER_CATEGORICAL if column in train_x.columns]
    model = CategoricalFrameModel(
        CatBoostClassifier(**settings), categorical, "catboost"
    )
    print(
        f"[{label}] outcome fit={int(usable.sum()):,}/{len(train_x):,} rows, "
        f"features={train_x.shape[1]}",
        flush=True,
    )
    started = time.perf_counter()
    fit_weight = train_weight[usable].astype(np.float64, copy=True) if train_weight is not None else None
    balance_details: dict[str, Any] = {"mode": season_class_balance}
    if season_class_balance is not None:
        if season_class_balance not in {"latest", "latest_game_type"}:
            raise ValueError(
                "season_class_balance must be one of: latest, latest_game_type"
            )
        label_frame = pd.DataFrame({
            "season": history.loc[usable, SEASON].to_numpy(dtype=np.int16),
            "game_type": history.loc[usable, "game_type"].astype(str).to_numpy(),
            "outcome": outcome.loc[usable].astype(str).to_numpy(),
        })
        latest_season = int(label_frame["season"].max())
        group_columns = (
            ["game_type"] if season_class_balance == "latest_game_type" else []
        )
        target_source = label_frame.loc[label_frame["season"] == latest_season]
        target_frequency = target_source.groupby(
            [*group_columns, "outcome"], observed=True
        ).size()
        if group_columns:
            target_frequency = target_frequency / target_frequency.groupby(
                level=list(range(len(group_columns)))
            ).transform("sum")
        else:
            target_frequency = target_frequency / target_frequency.sum()
        current_frequency = label_frame.groupby(
            ["season", *group_columns, "outcome"], observed=True
        ).size()
        current_frequency = current_frequency / current_frequency.groupby(
            level=list(range(1 + len(group_columns)))
        ).transform("sum")
        if fit_weight is None:
            fit_weight = np.ones(len(label_frame), dtype=np.float64)
        ratios = np.ones(len(label_frame), dtype=np.float64)
        for index, row in enumerate(label_frame.itertuples(index=False)):
            group_key = (
                (row.game_type, row.outcome)
                if group_columns else row.outcome
            )
            current_key = (
                (row.season, row.game_type, row.outcome)
                if group_columns else (row.season, row.outcome)
            )
            target_value = float(target_frequency.get(group_key, 0.0))
            current_value = float(current_frequency.get(current_key, 0.0))
            if target_value > 0 and current_value > 0:
                ratios[index] = target_value / current_value
        ratios = np.clip(
            ratios, 1.0 / season_class_balance_clip, season_class_balance_clip
        )
        ratios /= ratios.mean()
        fit_weight *= ratios
        balance_details.update({
            "latest_season": latest_season,
            "weight_min": float(ratios.min()),
            "weight_max": float(ratios.max()),
            "weight_mean": float(ratios.mean()),
        })
    model.fit(
        train_x.loc[usable],
        outcome.loc[usable].astype(str).to_numpy(),
        sample_weight=fit_weight,
    )
    fit_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    probabilities = model.predict_proba(valid_x)
    predict_seconds = time.perf_counter() - prediction_started
    classes = [str(value) for value in model.estimator.classes_]
    probability_unweighting = False
    configured_weights = settings.get("class_weights")
    configured_names = settings.get("class_names")
    if configured_weights is not None:
        if configured_names is not None:
            weight_map = {
                str(name): float(weight)
                for name, weight in zip(configured_names, configured_weights)
            }
            class_weight_vector = np.asarray(
                [weight_map[value] for value in classes], dtype=np.float64
            )
        else:
            class_weight_vector = np.asarray(configured_weights, dtype=np.float64)
            if len(class_weight_vector) != len(classes):
                raise ValueError(
                    "class_weights length does not match learned outcome classes"
                )
        probabilities = probabilities / class_weight_vector[None, :]
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        probability_unweighting = True
    success_indices = [
        index for index, value in enumerate(classes)
        if value == "success" or value.startswith("success|")
    ]
    if not success_indices:
        raise ValueError(f"Outcome model has no success class: {classes}")
    prediction = probabilities[:, success_indices].sum(axis=1).astype(np.float64)
    counts = outcome.loc[usable].value_counts().to_dict()
    details = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_iter": settings["iterations"],
        "model_params": settings,
        "prediction_std": float(np.std(prediction)),
        "outcome_classes": classes,
        "success_class_count": len(success_indices),
        "probability_unweighted_after_class_weighting": probability_unweighting,
        "outcome_counts": {str(key): int(value) for key, value in counts.items()},
        "outcome_usable_rows": int(usable.sum()),
        "outcome_dropped_rows": int((~usable).sum()),
        "outcome_scheme": outcome_scheme,
        "sample_weighted": bool(train_weight is not None),
        "season_class_balance": balance_details,
        "training_label_source": "next historical same-pitcher as-of counter delta",
    }
    if save_components:
        details["_component_predictions"] = {
            f"p_{index}_{value.replace('|', '_').replace(' ', '_')}": (
                probabilities[:, index].astype(np.float64)
            )
            for index, value in enumerate(classes)
        }
    if hasattr(model.estimator, "get_feature_importance"):
        importance = np.asarray(model.estimator.get_feature_importance(), dtype=np.float64)
        if len(importance) == len(train_x.columns):
            details["feature_importance"] = [
                {"feature": feature, "importance": float(value)}
                for feature, value in sorted(
                    zip(train_x.columns, importance),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
    export_catboost_model(
        label,
        model,
        list(train_x.columns),
        "outcome_classifier",
        {
            "classes": classes,
            "success_indices": success_indices,
            "class_weight_vector": (
                class_weight_vector.tolist() if probability_unweighting else None
            ),
        },
    )
    del model, probabilities, outcome
    gc.collect()
    return prediction, details


def _derive_training_game_ids(frame: pd.DataFrame) -> np.ndarray:
    """Reconstruct contiguous games using only official row-order context.

    The identifier is used only while fitting on labelled historical rows.  It
    is never required at inference, so predictions remain strictly row-local.
    """
    if frame.empty:
        return np.empty(0, dtype=np.int64)
    season = frame[SEASON].fillna(-1).to_numpy(dtype=np.int64, copy=False)
    month = frame["game_month"].fillna(-1).to_numpy(dtype=np.int64, copy=False)
    dow = frame["game_dayofweek"].fillna(-1).to_numpy(dtype=np.int64, copy=False)
    pitcher_team = frame["pitcher_team_id"].fillna(-1).to_numpy(
        dtype=np.int64, copy=False
    )
    batter_team = frame["batter_team_id"].fillna(-1).to_numpy(
        dtype=np.int64, copy=False
    )
    key = np.stack(
        [season, month, dow, np.minimum(pitcher_team, batter_team),
         np.maximum(pitcher_team, batter_team)],
        axis=1,
    )
    half = frame["top_bottom"].eq("B").to_numpy(
        dtype=np.int64, na_value=False
    )
    progress = (
        frame["inning"].fillna(-1).to_numpy(dtype=np.int64, copy=False) * 2
        + half
    )
    runs = frame["run_total_before"].fillna(-1).to_numpy(
        dtype=np.int64, copy=False
    )
    boundary = np.concatenate(
        [
            np.asarray([True]),
            np.any(key[1:] != key[:-1], axis=1)
            | (progress[1:] < progress[:-1])
            | (runs[1:] < runs[:-1]),
        ]
    )
    return boundary.cumsum(dtype=np.int64) - 1


def fit_game_centered_brier_model(
    label: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    valid_x: pd.DataFrame,
    params: dict[str, Any] | None = None,
    train_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a Brier regressor after removing labelled historical game means.

    The historical game fixed effect is a nuisance that cannot be observed for
    a future row.  Removing it from the training target asks the model to learn
    only within-game conditional deviations.  Output is centered at 0.5 so a
    downstream locked recipe can add the learned deviation to an honest parent.
    """
    from catboost import CatBoostRegressor

    game_id = _derive_training_game_ids(history)
    target = history[TARGET].to_numpy(dtype=np.float64, copy=False)
    target_frame = pd.DataFrame({"game_id": game_id, "target": target})
    game_mean = target_frame.groupby(
        "game_id", sort=False, observed=True
    )["target"].transform("mean").to_numpy(dtype=np.float64)
    centered_target = target - game_mean
    custom_params = dict(params or {})
    settings = {
        "loss_function": "RMSE",
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 50.0,
        "random_seed": RANDOM_SEED,
        "allow_writing_files": False,
        "thread_count": 6,
        "task_type": (
            "GPU"
            if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu"
            else "CPU"
        ),
    }
    settings.update(custom_params)
    categorical = [
        column for column in BOOSTER_CATEGORICAL if column in train_x.columns
    ]
    model = CategoricalFrameModel(
        CatBoostRegressor(**settings), categorical, "catboost"
    )
    started = time.perf_counter()
    model.fit(train_x, centered_target, sample_weight=train_weight)
    fit_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    deviation = np.asarray(model.predict(valid_x), dtype=np.float64).reshape(-1)
    predict_seconds = time.perf_counter() - prediction_started
    prediction = np.clip(0.5 + deviation, 1e-6, 1.0 - 1e-6)
    game_sizes = target_frame.groupby(
        "game_id", sort=False, observed=True
    ).size().to_numpy(dtype=np.int64)
    details = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_iter": int(settings["iterations"]),
        "model_params": settings,
        "prediction_std": float(np.std(prediction)),
        "deviation_mean": float(np.mean(deviation)),
        "deviation_std": float(np.std(deviation)),
        "training_target_mean": float(np.mean(centered_target)),
        "training_target_std": float(np.std(centered_target)),
        "training_games": int(len(game_sizes)),
        "training_game_rows_min": int(game_sizes.min()),
        "training_game_rows_median": float(np.median(game_sizes)),
        "training_game_rows_max": int(game_sizes.max()),
        "output_center": 0.5,
        "sample_weighted": bool(train_weight is not None),
        "training_only_grouping": True,
        "row_independent_inference": True,
    }
    if hasattr(model.estimator, "get_feature_importance"):
        importance = np.asarray(
            model.estimator.get_feature_importance(
                type="PredictionValuesChange"
            ),
            dtype=np.float64,
        )
        if len(importance) == len(train_x.columns):
            details["feature_importance"] = [
                {"feature": str(feature), "importance": float(value)}
                for feature, value in sorted(
                    zip(train_x.columns, importance),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
    return prediction, details


def fit_game_pairwise_rank_model(
    label: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    valid_x: pd.DataFrame,
    params: dict[str, Any] | None = None,
    train_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Learn success ordering within historical games and emit a row-local score."""
    from catboost import CatBoostRanker

    if train_weight is not None:
        raise ValueError("game pairwise preregistration does not allow sample weights")
    game_id = _derive_training_game_ids(history)
    target = history[TARGET].to_numpy(dtype=np.int8, copy=False)
    group_frame = pd.DataFrame({"game_id": game_id, "target": target})
    group_min = group_frame.groupby(
        "game_id", sort=False, observed=True
    )["target"].transform("min").to_numpy(dtype=np.int8)
    group_max = group_frame.groupby(
        "game_id", sort=False, observed=True
    )["target"].transform("max").to_numpy(dtype=np.int8)
    usable = group_min < group_max
    usable_x = train_x.loc[usable]
    usable_y = target[usable]
    usable_group = game_id[usable]
    custom_params = dict(params or {})
    output_scale = float(custom_params.pop("output_scale", 0.05))
    output_clip_z = float(custom_params.pop("output_clip_z", 4.0))
    settings = {
        "loss_function": "PairLogitPairwise",
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 50.0,
        "random_seed": RANDOM_SEED,
        "allow_writing_files": False,
        "thread_count": 6,
        "task_type": (
            "GPU"
            if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu"
            else "CPU"
        ),
    }
    settings.update(custom_params)
    categorical = [
        column for column in BOOSTER_CATEGORICAL if column in usable_x.columns
    ]
    model = CategoricalFrameModel(
        CatBoostRanker(**settings), categorical, "catboost"
    )
    prepared = model._prepare(usable_x, fitting=True)
    present = [column for column in categorical if column in prepared.columns]
    started = time.perf_counter()
    model.estimator.fit(
        prepared,
        usable_y,
        group_id=usable_group,
        cat_features=present,
        verbose=False,
    )
    fit_seconds = time.perf_counter() - started
    train_raw = np.asarray(model.estimator.predict(prepared), dtype=np.float64)
    raw_center = float(np.mean(train_raw))
    raw_scale = float(np.std(train_raw))
    if not np.isfinite(raw_scale) or raw_scale < 1e-8:
        raise ValueError(f"degenerate pairwise rank score scale: {raw_scale}")
    prediction_started = time.perf_counter()
    valid_raw = np.asarray(model.predict(valid_x), dtype=np.float64).reshape(-1)
    predict_seconds = time.perf_counter() - prediction_started
    z = np.clip((valid_raw - raw_center) / raw_scale, -output_clip_z, output_clip_z)
    deviation = output_scale * z
    prediction = np.clip(0.5 + deviation, 1e-6, 1.0 - 1e-6)
    group_sizes = group_frame.loc[usable].groupby(
        "game_id", sort=False, observed=True
    ).size().to_numpy(dtype=np.int64)
    details = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_iter": int(settings["iterations"]),
        "model_params": settings,
        "fit_rows": int(usable.sum()),
        "dropped_single_class_rows": int((~usable).sum()),
        "training_games": int(len(group_sizes)),
        "training_game_rows_min": int(group_sizes.min()),
        "training_game_rows_median": float(np.median(group_sizes)),
        "training_game_rows_max": int(group_sizes.max()),
        "raw_train_mean": raw_center,
        "raw_train_std": raw_scale,
        "valid_z_mean": float(np.mean(z)),
        "valid_z_std": float(np.std(z)),
        "output_scale": output_scale,
        "output_clip_z": output_clip_z,
        "prediction_std": float(np.std(prediction)),
        "output_center": 0.5,
        "training_only_grouping": True,
        "row_independent_inference": True,
    }
    if hasattr(model.estimator, "get_feature_importance"):
        importance = np.asarray(
            model.estimator.get_feature_importance(
                type="PredictionValuesChange"
            ),
            dtype=np.float64,
        )
        if len(importance) == len(train_x.columns):
            details["feature_importance"] = [
                {"feature": str(feature), "importance": float(value)}
                for feature, value in sorted(
                    zip(train_x.columns, importance),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
    return prediction, details


def fit_lgbm_outcome_model(
    label: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    valid_x: pd.DataFrame,
    params: dict[str, Any] | None = None,
    outcome_scheme: str = "five",
    train_weight: np.ndarray | None = None,
    outcome_labels: pd.Series | None = None,
    save_components: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the legal auxiliary outcome with a LightGBM multiclass model.

    This deliberately mirrors :func:`fit_outcome_model`: labels are recovered
    only inside the completed historical slice and the deployable output is the
    sum of the learned success subclasses.  It supplies a model-family-diverse
    axis without changing the feature or temporal protocol of exact-C.
    """
    if _LGBMClassifier is None:  # pragma: no cover - environment dependent
        raise RuntimeError("lightgbm is not installed")
    outcome = (
        derive_control_outcome_labels(history, outcome_scheme)
        if outcome_labels is None
        else outcome_labels.reindex(history.index)
    )
    usable = outcome.notna().to_numpy(dtype=bool)
    settings = {
        "objective": "multiclass",
        "metric": "None",
        "learning_rate": 0.03,
        "n_estimators": 700,
        "num_leaves": 31,
        "min_child_samples": 500,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 20.0,
        "n_jobs": 6,
        "random_state": RANDOM_SEED,
        "verbosity": -1,
    }
    settings.update(dict(params or {}))
    categorical = [column for column in BOOSTER_CATEGORICAL if column in train_x.columns]
    model = CategoricalFrameModel(
        _LGBMClassifier(**settings), categorical, "lgbm"
    )
    print(
        f"[{label}] LightGBM outcome fit={int(usable.sum()):,}/{len(train_x):,} "
        f"rows, features={train_x.shape[1]}",
        flush=True,
    )
    started = time.perf_counter()
    fit_weight = train_weight[usable] if train_weight is not None else None
    model.fit(
        train_x.loc[usable],
        outcome.loc[usable].astype(str).to_numpy(),
        sample_weight=fit_weight,
    )
    fit_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    probabilities = model.predict_proba(valid_x)
    predict_seconds = time.perf_counter() - prediction_started
    classes = [str(value) for value in model.estimator.classes_]
    success_indices = [
        index for index, value in enumerate(classes)
        if value == "success" or value.startswith("success|")
    ]
    if not success_indices:
        raise ValueError(f"LightGBM outcome model has no success class: {classes}")
    prediction = probabilities[:, success_indices].sum(axis=1).astype(np.float64)
    counts = outcome.loc[usable].value_counts().to_dict()
    details: dict[str, Any] = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_iter": int(settings["n_estimators"]),
        "model_params": settings,
        "prediction_std": float(np.std(prediction)),
        "outcome_classes": classes,
        "success_class_count": len(success_indices),
        "outcome_counts": {str(key): int(value) for key, value in counts.items()},
        "outcome_usable_rows": int(usable.sum()),
        "outcome_dropped_rows": int((~usable).sum()),
        "outcome_scheme": outcome_scheme,
        "sample_weighted": bool(train_weight is not None),
        "training_label_source": "next historical same-pitcher as-of counter delta",
        "row_independent_preprocessing": True,
    }
    if save_components:
        details["_component_predictions"] = {
            f"p_{index}_{value.replace('|', '_').replace(' ', '_')}": (
                probabilities[:, index].astype(np.float64)
            )
            for index, value in enumerate(classes)
        }
    importance = np.asarray(model.estimator.feature_importances_, dtype=np.float64)
    if len(importance) == len(train_x.columns):
        details["feature_importance"] = [
            {"feature": feature, "importance": float(value)}
            for feature, value in sorted(
                zip(train_x.columns, importance),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
    del model, probabilities, outcome
    gc.collect()
    return prediction, details


def fit_count_moe_outcome_model(
    label: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    valid_x: pd.DataFrame,
    valid_context: pd.DataFrame,
    params: dict[str, Any] | None = None,
    outcome_scheme: str = "reverse_any",
    train_weight: np.ndarray | None = None,
    outcome_labels: pd.Series | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one outcome expert per legal pre-pitch count and route by that row.

    Count is an input of the row being predicted, so this mixture is fully
    row-independent.  Auxiliary outcome labels are reconstructed once on the
    complete historical slice before it is partitioned; this preserves the
    same strict temporal label construction used by ``fit_outcome_model``.
    """
    if outcome_labels is None:
        outcome_labels = derive_control_outcome_labels(history, outcome_scheme)
    else:
        outcome_labels = outcome_labels.reindex(history.index)
    if len(valid_x) != len(valid_context):
        raise ValueError("valid_x and valid_context length mismatch")
    if not valid_x.index.equals(valid_context.index):
        raise ValueError("valid_x and valid_context index mismatch")

    train_balls = pd.to_numeric(history["balls_before"], errors="coerce")
    train_strikes = pd.to_numeric(history["strikes_before"], errors="coerce")
    valid_balls = pd.to_numeric(valid_context["balls_before"], errors="coerce")
    valid_strikes = pd.to_numeric(valid_context["strikes_before"], errors="coerce")
    prediction = np.full(len(valid_x), np.nan, dtype=np.float64)
    experts: dict[str, Any] = {}
    total_fit_seconds = 0.0
    total_predict_seconds = 0.0

    for balls in range(4):
        for strikes in range(3):
            name = f"{balls}-{strikes}"
            train_mask = (
                train_balls.eq(balls).to_numpy(dtype=bool)
                & train_strikes.eq(strikes).to_numpy(dtype=bool)
            )
            valid_mask = (
                valid_balls.eq(balls).to_numpy(dtype=bool)
                & valid_strikes.eq(strikes).to_numpy(dtype=bool)
            )
            if not np.any(train_mask):
                raise ValueError(f"Count expert {name} has no historical rows")
            if not np.any(valid_mask):
                experts[name] = {
                    "train_rows": int(train_mask.sum()),
                    "valid_rows": 0,
                    "skipped_empty_validation": True,
                }
                continue
            expert_prediction, expert_details = fit_outcome_model(
                f"{label}/count_{balls}_{strikes}",
                train_x.loc[train_mask],
                history.loc[train_mask],
                valid_x.loc[valid_mask],
                params,
                outcome_scheme,
                (
                    train_weight[train_mask]
                    if train_weight is not None
                    else None
                ),
                outcome_labels,
                False,
            )
            prediction[valid_mask] = expert_prediction
            total_fit_seconds += float(expert_details.get("fit_seconds", 0.0))
            total_predict_seconds += float(
                expert_details.get("predict_seconds", 0.0)
            )
            experts[name] = {
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                **expert_details,
            }

    missing = ~np.isfinite(prediction)
    if np.any(missing):
        bad = valid_context.loc[missing, ["balls_before", "strikes_before"]]
        raise ValueError(
            "Count MoE left validation rows unrouted: "
            f"{len(bad)} rows, examples={bad.head().to_dict(orient='records')}"
        )
    return prediction, {
        "architecture": "hard_count_routed_mixture_of_outcome_experts",
        "routing_columns": ["balls_before", "strikes_before"],
        "expert_count": 12,
        "fit_seconds": total_fit_seconds,
        "predict_seconds": total_predict_seconds,
        "prediction_std": float(np.std(prediction)),
        "outcome_scheme": outcome_scheme,
        "row_independent_routing": True,
        "training_label_source": (
            "full-history next same-pitcher as-of counter delta, then count split"
        ),
        "experts": experts,
    }


def fit_pitchtype_moe_outcome_model(
    label: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    valid_x: pd.DataFrame,
    valid_context: pd.DataFrame,
    params: dict[str, Any] | None = None,
    outcome_scheme: str = "reverse_any",
    train_weight: np.ndarray | None = None,
    outcome_labels: pd.Series | None = None,
    save_components: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Marginalize strong outcome experts over a deployable pitch-group model.

    Expert labels are recovered only for historical rows matched to the
    official 2019--2024 TrackMan table.  At prediction time the mixture weights
    are the four ``e22_cat`` probabilities inferred from pre-pitch row fields;
    the current pitch type is never used by the deployable prediction.  When
    requested, a true-group oracle is saved only as a clearly named diagnostic
    component for historical validation rows and is never returned as the
    candidate prediction.
    """
    from experiments.run_e22r_probs_rolling import (  # noqa: WPS433
        E22_PROB_FEATURES,
        GROUPS,
    )

    if "e22_pitch_type_group" not in history.columns:
        raise ValueError("Pitch-type MoE requires historical E22 group labels")
    missing_probabilities = [
        name for name in E22_PROB_FEATURES if name not in valid_x.columns
    ]
    if missing_probabilities:
        raise ValueError(
            "Pitch-type MoE requires e22_cat probabilities: "
            f"missing={missing_probabilities}"
        )
    if len(valid_x) != len(valid_context):
        raise ValueError("valid_x and valid_context length mismatch")
    if not valid_x.index.equals(valid_context.index):
        raise ValueError("valid_x and valid_context index mismatch")
    if outcome_labels is None:
        outcome_labels = derive_control_outcome_labels(history, outcome_scheme)
    else:
        outcome_labels = outcome_labels.reindex(history.index)

    # Keep the conditional experts identical to the exact C feature view.  The
    # stage-1 probabilities are used only as mixture weights, which prevents a
    # history in-sample group classifier from becoming an accidental shortcut.
    expert_features = [
        name for name in train_x.columns if name not in E22_PROB_FEATURES
    ]
    group_labels = history["e22_pitch_type_group"].astype("string")
    regular = history["game_type"].eq("R")
    matched = regular & group_labels.isin(GROUPS)
    probabilities = valid_x.loc[:, E22_PROB_FEATURES].to_numpy(dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    denominator = probabilities.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        probabilities,
        denominator,
        out=np.full_like(probabilities, 1.0 / len(GROUPS)),
        where=denominator > 0.0,
    )

    expert_matrix = np.empty((len(valid_x), len(GROUPS)), dtype=np.float64)
    experts: dict[str, Any] = {}
    total_fit_seconds = 0.0
    total_predict_seconds = 0.0
    minimum_usable_rows = 100 if len(history) < 100_000 else 1000
    for group_index, group in enumerate(GROUPS):
        group_mask = (matched & group_labels.eq(group)).to_numpy(dtype=bool)
        usable_rows = int(
            outcome_labels.loc[group_mask].notna().sum()
        )
        if usable_rows < minimum_usable_rows:
            raise ValueError(
                f"Pitch-type expert {group} has too few usable rows: {usable_rows}"
            )
        expert_prediction, expert_details = fit_outcome_model(
            f"{label}/pitch_group_{group}",
            train_x.loc[group_mask, expert_features],
            history.loc[group_mask],
            valid_x.loc[:, expert_features],
            params,
            outcome_scheme,
            train_weight[group_mask] if train_weight is not None else None,
            outcome_labels,
            False,
        )
        expert_matrix[:, group_index] = expert_prediction
        total_fit_seconds += float(expert_details.get("fit_seconds", 0.0))
        total_predict_seconds += float(
            expert_details.get("predict_seconds", 0.0)
        )
        experts[group] = {
            "matched_train_rows": int(group_mask.sum()),
            "usable_outcome_rows": usable_rows,
            **expert_details,
        }

    prediction = np.sum(probabilities * expert_matrix, axis=1)
    prediction = np.clip(prediction, 1e-6, 1.0 - 1e-6)
    details: dict[str, Any] = {
        "architecture": "soft_pitch_group_mixture_of_outcome_experts",
        "groups": list(GROUPS),
        "stage1_probability_columns": list(E22_PROB_FEATURES),
        "expert_feature_columns": expert_features,
        "matched_regular_history_rows": int(matched.sum()),
        "minimum_usable_rows_per_expert": int(minimum_usable_rows),
        "fit_seconds": total_fit_seconds,
        "predict_seconds": total_predict_seconds,
        "prediction_std": float(np.std(prediction)),
        "outcome_scheme": outcome_scheme,
        "current_pitch_type_used_at_inference": False,
        "row_independent_routing": True,
        "training_label_source": (
            "historical official TrackMan pitch group plus full-history next "
            "same-pitcher as-of counter outcome"
        ),
        "experts": experts,
    }
    if save_components:
        components: dict[str, np.ndarray] = {}
        for group_index, group in enumerate(GROUPS):
            components[f"p_{group}"] = probabilities[:, group_index]
            components[f"expert_{group}"] = expert_matrix[:, group_index]

        # This is privileged historical-validation diagnostics only.  It
        # quantifies whether pitch group contains enough signal to justify the
        # deployable marginalization, and is deliberately not the return value.
        oracle = prediction.copy()
        oracle_available = np.zeros(len(valid_context), dtype=np.int8)
        oracle_group_code = np.full(len(valid_context), -1, dtype=np.int8)
        if "e22_pitch_type_group" in valid_context.columns:
            valid_groups = valid_context["e22_pitch_type_group"].astype("string")
            for group_index, group in enumerate(GROUPS):
                mask = valid_groups.eq(group).fillna(False).to_numpy(dtype=bool)
                oracle[mask] = expert_matrix[mask, group_index]
                oracle_available[mask] = 1
                oracle_group_code[mask] = group_index
        components["diagnostic_true_group_oracle"] = oracle
        components["diagnostic_true_group_available"] = oracle_available
        components["diagnostic_true_group_code"] = oracle_group_code
        details["diagnostic_oracle_note"] = (
            "uses current historical validation pitch group; forbidden for "
            "deployment and excluded from every Goal gate"
        )
        details["diagnostic_oracle_available_rows"] = int(
            oracle_available.sum()
        )
        details["_component_predictions"] = components
    return prediction, details


def derive_dense_pitch_group_labels(frame: pd.DataFrame) -> pd.Series:
    """Recover completed-history coarse pitch groups from official counters.

    The label for a historical row is the one component whose cumulative
    pitcher pitchmix count increases at the next same-pitcher row.  This helper
    is training/diagnostic only; no validation or test neighbour is needed by
    the deployable selector or prediction.
    """
    groups = ("fastball", "breaking", "offspeed")
    columns = (
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    )
    n = (
        pd.to_numeric(frame["asof_pitcher_pitchmix_n"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.int64)
    )
    counts = np.column_stack(
        [
            np.rint(
                pd.to_numeric(frame[column], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
                * n
            ).astype(np.int64)
            for column in columns
        ]
    )
    work = pd.DataFrame(
        {
            "pitcher": frame[PITCHER].to_numpy(),
            "n": n,
            **{group: counts[:, index] for index, group in enumerate(groups)},
        },
        index=frame.index,
    )
    grouped = work.groupby("pitcher", sort=False, observed=True)
    next_n = grouped["n"].shift(-1)
    deltas = pd.DataFrame(
        {group: grouped[group].shift(-1) - work[group] for group in groups},
        index=frame.index,
    )
    valid = next_n.eq(work["n"] + 1)
    for group in groups:
        valid &= deltas[group].isin((0.0, 1.0))
    valid &= deltas.sum(axis=1).eq(1.0)
    labels = pd.Series(pd.NA, index=frame.index, dtype="string")
    for group in groups:
        labels.loc[valid & deltas[group].eq(1.0)] = group
    return labels


def fit_dense_pitchtype_moe_outcome_model(
    label: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    valid_x: pd.DataFrame,
    valid_context: pd.DataFrame,
    params: dict[str, Any] | None = None,
    outcome_scheme: str = "reverse_any",
    train_weight: np.ndarray | None = None,
    outcome_labels: pd.Series | None = None,
    save_components: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Marginalize dense counter-labelled exact-C outcome experts."""
    from catboost import CatBoostClassifier

    groups = ("fastball", "breaking", "offspeed")
    supplied = dict(params or {})
    selector_only_prefixes = tuple(
        str(value)
        for value in supplied.pop("dense_selector_only_prefixes", [])
    )
    group_physics_only = bool(
        supplied.pop("dense_expert_group_physics_only", False)
    )
    selector_settings = {
        "loss_function": "MultiClass",
        "iterations": int(supplied.pop("dense_selector_iterations", 400)),
        "depth": int(supplied.pop("dense_selector_depth", 7)),
        "learning_rate": float(
            supplied.pop("dense_selector_learning_rate", 0.05)
        ),
        "l2_leaf_reg": float(
            supplied.pop("dense_selector_l2_leaf_reg", 12.0)
        ),
        "random_seed": int(supplied.get("random_seed", RANDOM_SEED))
        + int(supplied.pop("dense_selector_seed_offset", 7300)),
        "allow_writing_files": False,
        "thread_count": 6,
        "task_type": (
            "GPU"
            if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu"
            else "CPU"
        ),
    }
    if outcome_labels is None:
        outcome_labels = derive_control_outcome_labels(history, outcome_scheme)
    else:
        outcome_labels = outcome_labels.reindex(history.index)
    group_labels = derive_dense_pitch_group_labels(history)
    regular = history["game_type"].eq("R")
    matched = regular & group_labels.isin(groups)
    if float(group_labels.loc[regular].notna().mean()) < 0.995:
        raise ValueError("dense pitch-group history coverage fell below 0.995")

    selector_train = train_x.copy()
    selector_valid = valid_x.copy()
    if PITCHER not in selector_train.columns:
        selector_train[PITCHER] = history[PITCHER].astype(str)
        selector_valid[PITCHER] = valid_context[PITCHER].astype(str)
    selector_train[PITCHER] = selector_train[PITCHER].astype(str)
    selector_valid[PITCHER] = selector_valid[PITCHER].astype(str)
    pitchmix_features = [
        column for column in selector_train.columns
        if column.startswith("e53_pitchmix_")
    ]
    if not pitchmix_features:
        raise ValueError("dense selector requires row-local e53 pitchmix features")
    selector_categorical = [
        column for column in BOOSTER_CATEGORICAL if column in selector_train.columns
    ]
    selector = CatBoostClassifier(**selector_settings)
    print(
        f"[{label}/selector] dense fit={int(matched.sum()):,} rows, "
        f"features={selector_train.shape[1]}",
        flush=True,
    )
    selector_started = time.perf_counter()
    selector.fit(
        selector_train.loc[matched],
        group_labels.loc[matched].astype(str),
        cat_features=selector_categorical,
        sample_weight=(train_weight[matched] if train_weight is not None else None),
        verbose=False,
    )
    selector_fit_seconds = time.perf_counter() - selector_started
    selector_predict_started = time.perf_counter()
    raw_probabilities = np.asarray(
        selector.predict_proba(selector_valid), dtype=np.float64
    )
    selector_predict_seconds = time.perf_counter() - selector_predict_started
    probabilities = np.zeros((len(valid_x), len(groups)), dtype=np.float64)
    selector_classes = [str(value) for value in selector.classes_]
    for source_index, group in enumerate(selector_classes):
        if group in groups:
            probabilities[:, groups.index(group)] = raw_probabilities[:, source_index]
    denominator = probabilities.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        probabilities,
        denominator,
        out=np.full_like(probabilities, 1.0 / len(groups)),
        where=denominator > 0.0,
    )

    # e53 state belongs only to the legal group selector.  Removing it from
    # the experts makes each expert's feature view exactly component C.
    base_expert_features = [
        column for column in train_x.columns
        if not column.startswith("e53_pitchmix_")
        and not column.startswith(selector_only_prefixes)
    ]
    group_physics_prefixes = tuple(
        f"e58_{name}_" for name in ("fastball", "breaking", "offspeed", "other")
    )
    common_expert_features = [
        column for column in base_expert_features
        if not column.startswith(group_physics_prefixes)
    ]
    expert_matrix = np.empty((len(valid_x), len(groups)), dtype=np.float64)
    experts: dict[str, Any] = {}
    expert_feature_columns: dict[str, list[str]] = {}
    total_fit_seconds = selector_fit_seconds
    total_predict_seconds = selector_predict_seconds
    for group_index, group in enumerate(groups):
        expert_features = (
            [
                *common_expert_features,
                *[
                    column for column in base_expert_features
                    if column.startswith(f"e58_{group}_")
                ],
            ]
            if group_physics_only
            else base_expert_features
        )
        expert_feature_columns[group] = expert_features
        group_mask = (matched & group_labels.eq(group)).to_numpy(dtype=bool)
        usable_rows = int(outcome_labels.loc[group_mask].notna().sum())
        if usable_rows < 1000:
            raise ValueError(f"dense pitch expert has too few rows: {group}/{usable_rows}")
        expert_prediction, expert_details = fit_outcome_model(
            f"{label}/dense_pitch_group_{group}",
            train_x.loc[group_mask, expert_features],
            history.loc[group_mask],
            valid_x.loc[:, expert_features],
            supplied,
            outcome_scheme,
            train_weight[group_mask] if train_weight is not None else None,
            outcome_labels,
            False,
        )
        expert_matrix[:, group_index] = expert_prediction
        total_fit_seconds += float(expert_details.get("fit_seconds", 0.0))
        total_predict_seconds += float(
            expert_details.get("predict_seconds", 0.0)
        )
        experts[group] = {
            "matched_train_rows": int(group_mask.sum()),
            "usable_outcome_rows": usable_rows,
            **expert_details,
        }

    prediction = np.clip(
        np.sum(probabilities * expert_matrix, axis=1), 1e-6, 1.0 - 1e-6
    )
    valid_dense = derive_dense_pitch_group_labels(valid_context)
    diagnostic_mask = valid_dense.isin(groups).to_numpy(dtype=bool)
    diagnostic_index = np.full(len(valid_x), -1, dtype=np.int8)
    for group_index, group in enumerate(groups):
        diagnostic_index[
            valid_dense.eq(group).fillna(False).to_numpy(dtype=bool)
        ] = group_index
    selector_accuracy = float(
        np.mean(
            probabilities[diagnostic_mask].argmax(axis=1)
            == diagnostic_index[diagnostic_mask]
        )
    )
    selector_true_probability = probabilities[diagnostic_mask][
        np.arange(int(diagnostic_mask.sum())), diagnostic_index[diagnostic_mask]
    ]
    trackman_agreement = None
    if "e22_pitch_type_group" in history.columns:
        tm = history["e22_pitch_type_group"].astype("string")
        tm_mask = matched & tm.isin(groups)
        if tm_mask.any():
            trackman_agreement = float(
                group_labels.loc[tm_mask].eq(tm.loc[tm_mask]).mean()
            )
    details: dict[str, Any] = {
        "architecture": "dense_counter_labelled_soft_pitch_group_moe",
        "groups": list(groups),
        "fit_seconds": total_fit_seconds,
        "predict_seconds": total_predict_seconds,
        "prediction_std": float(prediction.std()),
        "history_regular_rows": int(regular.sum()),
        "history_dense_label_rows": int(matched.sum()),
        "history_dense_label_coverage": float(
            group_labels.loc[regular].notna().mean()
        ),
        "history_trackman_group_agreement": trackman_agreement,
        "selector_settings": selector_settings,
        "selector_features": list(selector_train.columns),
        "selector_pitchmix_features": pitchmix_features,
        "selector_only_prefixes": list(selector_only_prefixes),
        "selector_only_features": [
            column for column in selector_train.columns
            if column.startswith(selector_only_prefixes)
        ],
        "selector_classes": selector_classes,
        "selector_fit_seconds": selector_fit_seconds,
        "selector_predict_seconds": selector_predict_seconds,
        "diagnostic_validation_dense_rows": int(diagnostic_mask.sum()),
        "diagnostic_selector_top1_accuracy": selector_accuracy,
        "diagnostic_selector_log_loss": float(
            -np.mean(np.log(np.clip(selector_true_probability, 1e-12, 1.0)))
        ),
        "expert_feature_columns": expert_feature_columns,
        "group_physics_only": group_physics_only,
        "experts": experts,
        "outcome_scheme": outcome_scheme,
        "current_pitch_group_used_at_inference": False,
        "row_independent_routing": True,
        "model_params": supplied,
    }
    if save_components:
        components: dict[str, np.ndarray] = {}
        for group_index, group in enumerate(groups):
            components[f"p_{group}"] = probabilities[:, group_index]
            components[f"expert_{group}"] = expert_matrix[:, group_index]
        oracle = prediction.copy()
        oracle[diagnostic_mask] = expert_matrix[diagnostic_mask][
            np.arange(int(diagnostic_mask.sum())), diagnostic_index[diagnostic_mask]
        ]
        components["diagnostic_true_group_oracle"] = oracle
        components["diagnostic_true_group_available"] = diagnostic_mask.astype(np.int8)
        components["diagnostic_true_group_code"] = diagnostic_index
        details["diagnostic_oracle_note"] = (
            "uses next-row reconstructed historical validation group; forbidden "
            "for deployment and excluded from every Goal gate"
        )
        details["_component_predictions"] = components
    del selector, selector_train, selector_valid, raw_probabilities
    gc.collect()
    return prediction, details


def fit_dense_pitch_joint_outcome_model(
    label: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    valid_x: pd.DataFrame,
    params: dict[str, Any] | None = None,
    outcome_scheme: str = "reverse_any",
    train_weight: np.ndarray | None = None,
    outcome_labels: pd.Series | None = None,
    save_components: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Model the joint control-outcome and dense pitch-group distribution.

    Historical pitch groups are recovered from official as-of counter
    increments.  The deployable success probability is the sum of every joint
    class whose outcome prefix is ``success``; no current validation/test group
    is read or predicted by a separate gating model.
    """
    if outcome_labels is None:
        outcome_labels = derive_control_outcome_labels(history, outcome_scheme)
    else:
        outcome_labels = outcome_labels.reindex(history.index)
    group_labels = derive_dense_pitch_group_labels(history)
    usable = outcome_labels.notna() & group_labels.notna()
    joint = pd.Series(pd.NA, index=history.index, dtype="string")
    joint.loc[usable] = (
        outcome_labels.loc[usable].astype(str)
        + "|pitch="
        + group_labels.loc[usable].astype(str)
    )
    prediction, details = fit_outcome_model(
        label,
        train_x,
        history,
        valid_x,
        params,
        outcome_scheme,
        train_weight,
        joint,
        save_components,
    )
    regular = history["game_type"].eq("R")
    details.update(
        {
            "architecture": "joint_dense_pitch_group_control_outcome",
            "joint_classes": sorted(
                str(value) for value in pd.unique(joint.loc[usable])
            ),
            "joint_usable_rows": int(usable.sum()),
            "joint_dropped_rows": int((~usable).sum()),
            "dense_group_coverage_all": float(group_labels.notna().mean()),
            "dense_group_coverage_R": float(
                group_labels.loc[regular].notna().mean()
            ),
            "success_marginalization": "sum classes starting with success|",
            "current_pitch_group_used_at_inference": False,
            "separate_selector_used": False,
            "row_independent_inference": True,
        }
    )
    return prediction, details


def fit_fine_pitch_joint_outcome_model(
    label: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    valid_x: pd.DataFrame,
    params: dict[str, Any] | None = None,
    outcome_scheme: str = "reverse_any",
    train_weight: np.ndarray | None = None,
    outcome_labels: pd.Series | None = None,
    save_components: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Jointly model control outcome and the matched historical fine pitch.

    Fine pitch type is an auxiliary label on completed historical rows only.
    At inference the current pitch type is never read: success probability is
    the marginal sum of the joint classifier's ``success|pitch=...`` classes.
    The optional e90 columns in ``train_x`` are themselves strict three-fold
    OOF/history-only fine-pitch probabilities built by
    ``build_fine_pitch_latent_probabilities``.
    """
    if outcome_labels is None:
        outcome_labels = derive_control_outcome_labels(history, outcome_scheme)
    else:
        outcome_labels = outcome_labels.reindex(history.index)
    if "fine_pitch_type" not in history.columns:
        raise ValueError(
            "catboost_fine_pitch_joint requires the fine_pitch_latent feature"
        )
    fine_labels = history["fine_pitch_type"].astype("string")
    matched = fine_labels.isin(FINE_PITCH_TYPES)
    usable = outcome_labels.notna() & matched
    joint = pd.Series(pd.NA, index=history.index, dtype="string")
    joint.loc[usable] = (
        outcome_labels.loc[usable].astype(str)
        + "|pitch="
        + fine_labels.loc[usable].astype(str)
    )
    prediction, details = fit_outcome_model(
        label,
        train_x,
        history,
        valid_x,
        params,
        outcome_scheme,
        train_weight,
        joint,
        save_components,
    )
    regular = history["game_type"].eq("R")
    details.update(
        {
            "architecture": "joint_matched_fine_pitch_control_outcome",
            "fine_pitch_types": list(FINE_PITCH_TYPES),
            "joint_classes": sorted(
                str(value) for value in pd.unique(joint.loc[usable])
            ),
            "joint_usable_rows": int(usable.sum()),
            "joint_dropped_rows": int((~usable).sum()),
            "fine_label_coverage_all": float(matched.mean()),
            "fine_label_coverage_R": float(matched.loc[regular].mean()),
            "success_marginalization": "sum classes starting with success|",
            "fine_probability_features": [
                column for column in train_x if column.startswith("e90_")
            ],
            "current_pitch_type_used_at_inference": False,
            "current_pitch_trackman_used_at_inference": False,
            "row_independent_inference": True,
        }
    )
    return prediction, details


def fit_physics_joint_outcome_model(
    label: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    valid_x: pd.DataFrame,
    params: dict[str, Any] | None = None,
    outcome_scheme: str = "reverse_any",
    train_weight: np.ndarray | None = None,
    outcome_labels: pd.Series | None = None,
    save_components: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Marginalize a target-free current-pitch physics clustering.

    The robust scaler and MiniBatchKMeans are fit only on matched completed
    history.  Cluster IDs are auxiliary labels for the joint outcome model;
    validation/test physics is never transformed or read.  Deployment only
    needs the ordinary row-local ``train_x`` columns and sums all success
    joint-class probabilities.
    """
    from sklearn.cluster import MiniBatchKMeans

    if outcome_labels is None:
        outcome_labels = derive_control_outcome_labels(history, outcome_scheme)
    else:
        outcome_labels = outcome_labels.reindex(history.index)
    auxiliary = [f"_aux_tm_{column}" for column in PHYSICS_AUX_COLUMNS]
    missing = [column for column in auxiliary if column not in history.columns]
    if missing:
        raise ValueError(
            "catboost_physics_joint requires hidden historical TrackMan labels: "
            f"{missing}"
        )
    physics = history[auxiliary].apply(pd.to_numeric, errors="coerce")
    matched = physics.notna().all(axis=1)
    usable = outcome_labels.notna() & matched
    if int(usable.sum()) < 10000:
        raise ValueError(f"too few usable physics-joint rows: {int(usable.sum())}")
    values = physics.loc[usable].to_numpy(dtype=np.float64)
    center = np.median(values, axis=0)
    lower = np.quantile(values, 0.25, axis=0)
    upper = np.quantile(values, 0.75, axis=0)
    scale = upper - lower
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = np.clip((values - center) / scale, -6.0, 6.0)
    cluster_count = 12
    clusterer = MiniBatchKMeans(
        n_clusters=cluster_count,
        init="k-means++",
        n_init=10,
        max_iter=200,
        batch_size=8192,
        random_state=2026,
        reassignment_ratio=0.0,
    )
    cluster_started = time.perf_counter()
    cluster_codes = clusterer.fit_predict(standardized).astype(np.int16)
    cluster_seconds = time.perf_counter() - cluster_started
    physics_labels = pd.Series(pd.NA, index=history.index, dtype="string")
    physics_labels.loc[usable] = pd.Series(
        [f"p{int(value):02d}" for value in cluster_codes],
        index=history.index[usable],
        dtype="string",
    )
    joint = pd.Series(pd.NA, index=history.index, dtype="string")
    joint.loc[usable] = (
        outcome_labels.loc[usable].astype(str)
        + "|physics="
        + physics_labels.loc[usable].astype(str)
    )
    prediction, details = fit_outcome_model(
        label,
        train_x,
        history,
        valid_x,
        params,
        outcome_scheme,
        train_weight,
        joint,
        save_components,
    )
    regular = history["game_type"].eq("R")
    counts = pd.Series(cluster_codes).value_counts().sort_index()
    details.update(
        {
            "architecture": "joint_history_physics_cluster_control_outcome",
            "physics_columns": list(PHYSICS_AUX_COLUMNS),
            "physics_preprocessing": "history-only median/IQR then clip[-6,6]",
            "physics_center": center.tolist(),
            "physics_scale_iqr": scale.tolist(),
            "cluster_algorithm": "MiniBatchKMeans",
            "cluster_count": cluster_count,
            "cluster_counts": {
                f"p{int(key):02d}": int(value) for key, value in counts.items()
            },
            "cluster_fit_seconds": float(cluster_seconds),
            "cluster_inertia": float(clusterer.inertia_),
            "joint_classes": sorted(
                str(value) for value in pd.unique(joint.loc[usable])
            ),
            "joint_usable_rows": int(usable.sum()),
            "joint_dropped_rows": int((~usable).sum()),
            "physics_label_coverage_all": float(matched.mean()),
            "physics_label_coverage_R": float(matched.loc[regular].mean()),
            "success_marginalization": "sum classes starting with success|",
            "fine_probability_features": [
                column for column in train_x if column.startswith("e90_")
            ],
            "current_pitch_physics_used_at_inference": False,
            "current_pitch_type_used_at_inference": False,
            "row_independent_inference": True,
            "training_label_source": (
                "matched completed-history TrackMan physics clustered without "
                "control target"
            ),
        }
    )
    del clusterer, standardized, values, physics, physics_labels, joint
    gc.collect()
    return prediction, details


def fit_hierarchical_pitch_joint_outcome_model(
    label: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    valid_x: pd.DataFrame,
    params: dict[str, Any] | None = None,
    outcome_scheme: str = "reverse_any",
    train_weight: np.ndarray | None = None,
    outcome_labels: pd.Series | None = None,
    save_components: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use fine pitch labels when matched and dense coarse labels otherwise."""
    if outcome_labels is None:
        outcome_labels = derive_control_outcome_labels(history, outcome_scheme)
    else:
        outcome_labels = outcome_labels.reindex(history.index)
    if "fine_pitch_type" not in history.columns:
        raise ValueError(
            "catboost_hier_pitch_joint requires the fine_pitch_latent feature"
        )
    fine = history["fine_pitch_type"].astype("string")
    fine_matched = fine.isin(FINE_PITCH_TYPES)
    coarse = derive_dense_pitch_group_labels(history).astype("string")
    coarse_matched = coarse.notna()
    latent = pd.Series(pd.NA, index=history.index, dtype="string")
    latent.loc[coarse_matched] = (
        "coarse=" + coarse.loc[coarse_matched].astype(str)
    )
    latent.loc[fine_matched] = "fine=" + fine.loc[fine_matched].astype(str)
    usable = outcome_labels.notna() & latent.notna()
    joint = pd.Series(pd.NA, index=history.index, dtype="string")
    joint.loc[usable] = (
        outcome_labels.loc[usable].astype(str)
        + "|pitch="
        + latent.loc[usable].astype(str)
    )
    prediction, details = fit_outcome_model(
        label,
        train_x,
        history,
        valid_x,
        params,
        outcome_scheme,
        train_weight,
        joint,
        save_components,
    )
    regular = history["game_type"].eq("R")
    details.update(
        {
            "architecture": "joint_hierarchical_fine_else_dense_coarse_pitch",
            "fine_pitch_types": list(FINE_PITCH_TYPES),
            "coarse_pitch_groups": ["fastball", "breaking", "offspeed"],
            "joint_classes": sorted(
                str(value) for value in pd.unique(joint.loc[usable])
            ),
            "joint_usable_rows": int(usable.sum()),
            "joint_dropped_rows": int((~usable).sum()),
            "fine_label_coverage_all": float(fine_matched.mean()),
            "fine_label_coverage_R": float(fine_matched.loc[regular].mean()),
            "coarse_label_coverage_all": float(coarse_matched.mean()),
            "coarse_label_coverage_R": float(coarse_matched.loc[regular].mean()),
            "hierarchical_label_coverage_all": float(latent.notna().mean()),
            "hierarchical_label_coverage_R": float(
                latent.loc[regular].notna().mean()
            ),
            "fine_probability_features": [
                column for column in train_x if column.startswith("e90_")
            ],
            "success_marginalization": "sum classes starting with success|",
            "current_pitch_type_used_at_inference": False,
            "current_pitch_trackman_used_at_inference": False,
            "row_independent_inference": True,
        }
    )
    return prediction, details


def fit_auto_pitch_joint_outcome_model(
    label: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    valid_x: pd.DataFrame,
    params: dict[str, Any] | None = None,
    outcome_scheme: str = "reverse_any",
    train_weight: np.ndarray | None = None,
    outcome_labels: pd.Series | None = None,
    save_components: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Joint control outcome and normalized historical auto pitch type."""
    if outcome_labels is None:
        outcome_labels = derive_control_outcome_labels(history, outcome_scheme)
    else:
        outcome_labels = outcome_labels.reindex(history.index)
    label_column = "auto_fine_pitch_type"
    if label_column not in history.columns:
        raise ValueError(
            "catboost_auto_pitch_joint requires the auto_pitch_latent feature"
        )
    auto = history[label_column].astype("string")
    matched = auto.isin(FINE_PITCH_TYPES)
    usable = outcome_labels.notna() & matched
    joint = pd.Series(pd.NA, index=history.index, dtype="string")
    joint.loc[usable] = (
        outcome_labels.loc[usable].astype(str)
        + "|pitch="
        + auto.loc[usable].astype(str)
    )
    prediction, details = fit_outcome_model(
        label,
        train_x,
        history,
        valid_x,
        params,
        outcome_scheme,
        train_weight,
        joint,
        save_components,
    )
    regular = history["game_type"].eq("R")
    details.update(
        {
            "architecture": "joint_matched_auto_pitch_control_outcome",
            "fine_pitch_types": list(FINE_PITCH_TYPES),
            "joint_classes": sorted(
                str(value) for value in pd.unique(joint.loc[usable])
            ),
            "joint_usable_rows": int(usable.sum()),
            "joint_dropped_rows": int((~usable).sum()),
            "auto_label_coverage_all": float(matched.mean()),
            "auto_label_coverage_R": float(matched.loc[regular].mean()),
            "success_marginalization": "sum classes starting with success|",
            "auto_probability_features": [
                column
                for column in train_x
                if column.startswith(("e90_", "e92_"))
            ],
            "auto_profile_features": [
                column for column in train_x if column.startswith("e91_")
            ],
            "current_pitch_type_used_at_inference": False,
            "current_pitch_trackman_used_at_inference": False,
            "row_independent_inference": True,
        }
    )
    return prediction, details


def fit_fine_pitch_moe_outcome_model(
    label: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    valid_x: pd.DataFrame,
    params: dict[str, Any] | None = None,
    outcome_scheme: str = "reverse_any",
    train_weight: np.ndarray | None = None,
    outcome_labels: pd.Series | None = None,
    save_components: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Marginalize eight auto-fine-pitch conditional outcome experts."""
    label_column = "auto_fine_pitch_type"
    if label_column not in history.columns:
        raise ValueError(
            "catboost_fine_pitch_moe requires an auto-pitch latent feature"
        )
    probability_columns = [
        f"e92_p_{pitch_type.lower()}" for pitch_type in FINE_PITCH_TYPES
    ]
    missing = [column for column in probability_columns if column not in valid_x]
    if missing:
        raise ValueError(
            "catboost_fine_pitch_moe requires the locked expanded selector: "
            f"{missing}"
        )
    if outcome_labels is None:
        outcome_labels = derive_control_outcome_labels(history, outcome_scheme)
    else:
        outcome_labels = outcome_labels.reindex(history.index)
    expert_features = [
        column for column in train_x if not column.startswith("e92_")
    ]
    selector = valid_x[probability_columns].to_numpy(dtype=np.float64)
    selector /= np.maximum(selector.sum(axis=1, keepdims=True), 1e-12)
    expert_matrix = np.empty(
        (len(valid_x), len(FINE_PITCH_TYPES)), dtype=np.float64
    )
    experts: dict[str, Any] = {}
    components: dict[str, np.ndarray] = {}
    total_fit_seconds = 0.0
    total_predict_seconds = 0.0
    labels = history[label_column].astype("string")
    for pitch_index, pitch_type in enumerate(FINE_PITCH_TYPES):
        mask = labels.eq(pitch_type).fillna(False).to_numpy(dtype=bool)
        usable = mask & outcome_labels.notna().to_numpy(dtype=bool)
        if int(usable.sum()) < 500:
            raise ValueError(
                f"fine pitch expert has too few usable rows: "
                f"{pitch_type}/{int(usable.sum())}"
            )
        prediction, expert_details = fit_outcome_model(
            f"{label}/fine_pitch_{pitch_type.lower()}",
            train_x.loc[usable, expert_features],
            history.loc[usable],
            valid_x.loc[:, expert_features],
            params,
            outcome_scheme,
            train_weight[usable] if train_weight is not None else None,
            outcome_labels,
            False,
        )
        expert_matrix[:, pitch_index] = prediction
        total_fit_seconds += float(expert_details.get("fit_seconds", 0.0))
        total_predict_seconds += float(
            expert_details.get("predict_seconds", 0.0)
        )
        experts[pitch_type] = {
            "matched_rows": int(mask.sum()),
            "usable_rows": int(usable.sum()),
            **expert_details,
        }
        if save_components:
            components[f"expert_{pitch_type.lower()}"] = prediction
            components[f"selector_{pitch_type.lower()}"] = selector[:, pitch_index]
    prediction = np.clip(
        np.sum(selector * expert_matrix, axis=1), 1e-6, 1.0 - 1e-6
    )
    regular = history["game_type"].eq("R")
    details: dict[str, Any] = {
        "architecture": "eight_auto_fine_pitch_conditional_outcome_experts",
        "fine_pitch_types": list(FINE_PITCH_TYPES),
        "expert_count": len(experts),
        "experts": experts,
        "expert_feature_columns": expert_features,
        "selector_probability_features": probability_columns,
        "auto_label_coverage_all": float(labels.isin(FINE_PITCH_TYPES).mean()),
        "auto_label_coverage_R": float(
            labels.loc[regular].isin(FINE_PITCH_TYPES).mean()
        ),
        "fit_seconds": total_fit_seconds,
        "predict_seconds": total_predict_seconds,
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "current_pitch_type_used_at_inference": False,
        "current_pitch_trackman_used_at_inference": False,
        "row_independent_inference": True,
    }
    if save_components:
        details["_component_predictions"] = components
    return prediction, details


def fit_fine_pitch_binary_moe_model(
    label: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    valid_x: pd.DataFrame,
    params: dict[str, Any] | None = None,
    train_weight: np.ndarray | None = None,
    save_components: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Marginalize eight direct binary control-success experts."""
    label_column = "auto_fine_pitch_type"
    if label_column not in history.columns:
        raise ValueError(
            "catboost_fine_pitch_binary_moe requires auto-pitch labels"
        )
    probability_columns = [
        f"e92_p_{pitch_type.lower()}" for pitch_type in FINE_PITCH_TYPES
    ]
    missing = [column for column in probability_columns if column not in valid_x]
    if missing:
        raise ValueError(
            "catboost_fine_pitch_binary_moe requires the locked e92 selector: "
            f"{missing}"
        )
    expert_features = [
        column for column in train_x if not column.startswith("e92_")
    ]
    selector = valid_x[probability_columns].to_numpy(dtype=np.float64)
    selector /= np.maximum(selector.sum(axis=1, keepdims=True), 1e-12)
    expert_matrix = np.empty(
        (len(valid_x), len(FINE_PITCH_TYPES)), dtype=np.float64
    )
    labels = history[label_column].astype("string")
    target = history[TARGET].to_numpy(dtype=np.int8)
    experts: dict[str, Any] = {}
    components: dict[str, np.ndarray] = {}
    total_fit_seconds = 0.0
    total_predict_seconds = 0.0
    for pitch_index, pitch_type in enumerate(FINE_PITCH_TYPES):
        usable = labels.eq(pitch_type).fillna(False).to_numpy(dtype=bool)
        if int(usable.sum()) < 500:
            raise ValueError(
                f"binary fine pitch expert has too few rows: "
                f"{pitch_type}/{int(usable.sum())}"
            )
        if np.unique(target[usable]).size != 2:
            raise ValueError(f"binary fine pitch expert has one class: {pitch_type}")
        model = make_catboost(list(expert_features), dict(params or {}))
        print(
            f"[{label}/binary_fine_pitch_{pitch_type.lower()}] "
            f"fit={int(usable.sum()):,} rows, features={len(expert_features)}",
            flush=True,
        )
        started = time.perf_counter()
        model.fit(
            train_x.loc[usable, expert_features],
            target[usable],
            sample_weight=(
                train_weight[usable] if train_weight is not None else None
            ),
        )
        fit_seconds = time.perf_counter() - started
        predict_started = time.perf_counter()
        expert_prediction = model.predict_proba(
            valid_x.loc[:, expert_features]
        )[:, 1].astype(np.float64)
        predict_seconds = time.perf_counter() - predict_started
        expert_matrix[:, pitch_index] = expert_prediction
        total_fit_seconds += fit_seconds
        total_predict_seconds += predict_seconds
        experts[pitch_type] = {
            "usable_rows": int(usable.sum()),
            "success_rate": float(target[usable].mean()),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "prediction_mean": float(expert_prediction.mean()),
            "prediction_std": float(expert_prediction.std()),
            "target_source": "historical control_success hard label",
            "model_params": dict(params or {}),
        }
        if save_components:
            components[f"expert_{pitch_type.lower()}"] = expert_prediction
            components[f"selector_{pitch_type.lower()}"] = selector[:, pitch_index]
        del model
        gc.collect()
    prediction = np.clip(
        np.sum(selector * expert_matrix, axis=1), 1e-6, 1.0 - 1e-6
    )
    regular = history["game_type"].eq("R")
    details: dict[str, Any] = {
        "architecture": "eight_auto_fine_pitch_direct_binary_experts",
        "fine_pitch_types": list(FINE_PITCH_TYPES),
        "expert_count": len(experts),
        "experts": experts,
        "expert_feature_columns": expert_features,
        "selector_probability_features": probability_columns,
        "auto_label_coverage_all": float(labels.isin(FINE_PITCH_TYPES).mean()),
        "auto_label_coverage_R": float(
            labels.loc[regular].isin(FINE_PITCH_TYPES).mean()
        ),
        "fit_seconds": total_fit_seconds,
        "predict_seconds": total_predict_seconds,
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "target_source": "direct historical control_success hard label",
        "current_pitch_type_used_at_inference": False,
        "current_pitch_trackman_used_at_inference": False,
        "row_independent_inference": True,
    }
    if save_components:
        details["_component_predictions"] = components
    return prediction, details


def fit_failure_decomposition_model(
    label: str,
    train_x: pd.DataFrame,
    valid_x: pd.DataFrame,
    outcome_labels: pd.Series,
    params: dict[str, Any] | None = None,
    train_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit independent reverse/middle/wayoff failure experts.

    Labels come only from official historical as-of counter increments.  The
    current validation/test pitch type or any other target-row aggregate is
    never used.  This mirrors the public failure-decomposition method while
    rebuilding every model locally.
    """
    labels = outcome_labels.reindex(train_x.index).astype("string")
    usable = labels.notna().to_numpy(dtype=bool)
    text_labels = labels.loc[usable].astype(str)
    failure = text_labels.str.startswith("failure|")
    targets = {
        "reverse": (failure & text_labels.str.contains("r1m", regex=False)).to_numpy(
            dtype=np.int8
        ),
        "middle": (failure & text_labels.str.contains("m1b", regex=False)).to_numpy(
            dtype=np.int8
        ),
        "wayoff": (
            failure & text_labels.str.contains("r0m0b", regex=False)
        ).to_numpy(dtype=np.int8),
    }
    predictions: dict[str, np.ndarray] = {}
    fit_rows: dict[str, dict[str, Any]] = {}
    fit_seconds = 0.0
    predict_seconds = 0.0
    for component, target in targets.items():
        component_params = dict(params or {})
        component_params["random_seed"] = int(
            component_params.get("random_seed", RANDOM_SEED)
        ) + {"reverse": 0, "middle": 1, "wayoff": 2}[component]
        model = make_catboost(list(train_x.columns), component_params)
        started = time.perf_counter()
        model.fit(
            train_x.loc[usable],
            target,
            sample_weight=(train_weight[usable] if train_weight is not None else None),
        )
        component_fit_seconds = time.perf_counter() - started
        prediction_started = time.perf_counter()
        probability = model.predict_proba(valid_x)[:, 1].astype(np.float64)
        component_predict_seconds = time.perf_counter() - prediction_started
        predictions[component] = probability
        fit_seconds += component_fit_seconds
        predict_seconds += component_predict_seconds
        fit_rows[component] = {
            "positive_rows": int(target.sum()),
            "negative_rows": int(len(target) - target.sum()),
            "target_rate": float(target.mean()),
            "fit_seconds": component_fit_seconds,
            "predict_seconds": component_predict_seconds,
            "random_seed": component_params["random_seed"],
        }
        del model
        gc.collect()

    all_failure = np.clip(
        1.0
        - predictions["reverse"]
        - predictions["middle"]
        - predictions["wayoff"],
        1e-6,
        1.0 - 1e-6,
    )
    details: dict[str, Any] = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_iter": int((params or {}).get("iterations", 1500)),
        "prediction_std": float(np.std(all_failure)),
        "outcome_usable_rows": int(usable.sum()),
        "outcome_dropped_rows": int((~usable).sum()),
        "training_label_source": "next historical same-pitcher as-of counter delta",
        "failure_components": fit_rows,
        "sample_weighted": bool(train_weight is not None),
        "model_params": dict(params or {}),
        "_component_predictions": {
            "p_reverse_failure": predictions["reverse"],
            "p_middle_failure": predictions["middle"],
            "p_wayoff_failure": predictions["wayoff"],
            "p_no_middle": 1.0 - predictions["middle"],
        },
    }
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return all_failure, details


def fit_failure_chain_model(
    label: str,
    train_x: pd.DataFrame,
    valid_x: pd.DataFrame,
    outcome_labels: pd.Series,
    params: dict[str, Any] | None = None,
    train_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Factor mutually exclusive failures into a conditional probability chain.

    The historical labels are the same leakage-safe ``reverse_any`` labels used
    by exact component C.  Unlike ``fit_failure_decomposition_model``, every
    later binary task is trained only among rows that survived the earlier
    failure branches, so multiplying the three survival probabilities yields a
    coherent row-local success probability without clipping a sum of overlaps.
    """
    labels = outcome_labels.reindex(train_x.index).astype("string")
    usable = labels.isin(("success", "reverse", "middle", "wide")).to_numpy(
        dtype=bool
    )
    if not np.any(usable):
        raise ValueError("failure-chain training labels are empty")
    remaining_train = usable.copy()
    success_probability = np.ones(len(valid_x), dtype=np.float64)
    components: dict[str, np.ndarray] = {}
    stages: dict[str, dict[str, Any]] = {}
    total_fit_seconds = 0.0
    total_predict_seconds = 0.0

    for branch in ("reverse", "middle", "wide"):
        target = labels.eq(branch).fillna(False).to_numpy(dtype=np.int8)
        stage_target = target[remaining_train]
        if np.unique(stage_target).size != 2:
            raise ValueError(f"failure-chain branch has one class: {branch}")
        model = make_catboost(list(train_x.columns), dict(params or {}))
        print(
            f"[{label}/{branch}] conditional fit={int(remaining_train.sum()):,} "
            f"positive={int(stage_target.sum()):,}",
            flush=True,
        )
        started = time.perf_counter()
        model.fit(
            train_x.loc[remaining_train],
            stage_target,
            sample_weight=(
                train_weight[remaining_train]
                if train_weight is not None
                else None
            ),
        )
        fit_seconds = time.perf_counter() - started
        prediction_started = time.perf_counter()
        branch_probability = model.predict_proba(valid_x)[:, 1].astype(np.float64)
        predict_seconds = time.perf_counter() - prediction_started
        success_probability *= 1.0 - branch_probability
        components[f"p_{branch}_conditional"] = branch_probability
        stages[branch] = {
            "fit_rows": int(remaining_train.sum()),
            "positive_rows": int(stage_target.sum()),
            "target_rate": float(stage_target.mean()),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "prediction_mean": float(branch_probability.mean()),
            "prediction_std": float(branch_probability.std()),
        }
        total_fit_seconds += fit_seconds
        total_predict_seconds += predict_seconds
        export_catboost_model(
            f"{label}/{branch}",
            model,
            list(train_x.columns),
            "classifier",
            {"failure_chain_branch": branch},
        )
        remaining_train &= target == 0
        del model
        gc.collect()

    prediction = np.clip(success_probability, 1e-6, 1.0 - 1e-6)
    components["p_success_product"] = prediction
    details: dict[str, Any] = {
        "fit_seconds": total_fit_seconds,
        "predict_seconds": total_predict_seconds,
        "n_iter": int((params or {}).get("iterations", 1500)),
        "prediction_std": float(prediction.std()),
        "outcome_usable_rows": int(usable.sum()),
        "outcome_dropped_rows": int((~usable).sum()),
        "remaining_success_rows": int(remaining_train.sum()),
        "chain_order": ["reverse", "middle", "wide", "success"],
        "probability_formula": (
            "(1-p_reverse)*(1-p_middle_given_not_reverse)*"
            "(1-p_wide_given_not_reverse_middle)"
        ),
        "training_label_source": "reverse_any next historical counter outcome",
        "conditional_stages": stages,
        "sample_weighted": bool(train_weight is not None),
        "model_params": dict(params or {}),
        "_component_predictions": components,
    }
    return prediction, details


def _prepare_realmlp_frames(
    train_x: pd.DataFrame,
    valid_x: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Train-only imputation and explicit categorical typing for RealMLP."""
    categorical = [column for column in BOOSTER_CATEGORICAL if column in train_x]
    numeric = [column for column in train_x if column not in categorical]
    train = train_x.copy()
    valid = valid_x.copy()
    for column in categorical:
        train[column] = train[column].astype("string").fillna("__missing__").astype(str)
        valid[column] = valid[column].astype("string").fillna("__missing__").astype(str)
    retained_numeric: list[str] = []
    for column in numeric:
        values = pd.to_numeric(train[column], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        train_values = values.fillna(median).to_numpy(dtype=np.float32)
        valid_values = (
            pd.to_numeric(valid[column], errors="coerce")
            .fillna(median)
            .to_numpy(dtype=np.float32)
        )
        # Constant columns make the quantile preprocessing needlessly costly
        # and contributed no information in the TabM experiments either.
        if float(np.max(train_values) - np.min(train_values)) <= 1e-12:
            train.drop(columns=[column], inplace=True)
            valid.drop(columns=[column], inplace=True)
            continue
        train[column] = train_values
        valid[column] = valid_values
        retained_numeric.append(column)
    return train, valid, categorical, retained_numeric


def fit_realmlp_model(
    label: str,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: pd.DataFrame,
    params: dict[str, Any] | None = None,
    outcome: pd.Series | None = None,
    train_weight: np.ndarray | None = None,
    save_components: bool = False,
    architecture: str = "realmlp",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit an official PyTabKit neural tabular model on one outer fold."""
    if train_weight is not None:
        raise ValueError("RealMLP experiments currently require unweighted rows")
    from pytabkit import (
        RealMLP_TD_Classifier,
        RealTabR_D_Classifier,
        TabR_S_D_Classifier,
    )

    device = "cuda" if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu" else "cpu"
    if architecture == "realmlp":
        model_class = RealMLP_TD_Classifier
        settings: dict[str, Any] = {
            "device": device,
            "random_state": RANDOM_SEED,
            "n_cv": 1,
            "n_refit": 0,
            "val_fraction": 0.1,
            "n_threads": 6,
            "verbosity": 1,
            "val_metric_name": "brier",
            "n_epochs": 64,
            "batch_size": 4096,
            "predict_batch_size": 8192,
            "hidden_sizes": [256, 256, 256],
            "use_ls": False,
            "n_ens": 1,
        }
    elif architecture in {"tabr", "realtabr"}:
        model_class = (
            TabR_S_D_Classifier if architecture == "tabr" else RealTabR_D_Classifier
        )
        settings = {
            # Windows wheels provide faiss-cpu only; PyTabKit otherwise calls
            # unavailable GpuIndexFlat* classes during nearest-neighbour search.
            "device": "cpu",
            "random_state": RANDOM_SEED,
            "n_cv": 1,
            "n_refit": 0,
            "val_fraction": 0.1,
            "n_threads": 6,
            "verbosity": 1,
            "val_metric_name": "cross_entropy",
            "n_epochs": 16,
            "batch_size": 512,
            "eval_batch_size": 4096,
            "context_size": 96,
            "memory_efficient": True,
            "candidate_encoding_batch_size": 4096,
            # Context-freezing uses worker processes whose locally defined
            # dataset cannot be pickled by Windows' spawn start method.
            "freeze_contexts_after_n_epochs": None,
        }
    else:
        raise ValueError(f"Unknown PyTabKit architecture: {architecture}")
    if params:
        settings.update(params)
    usable = (
        np.ones(len(train_x), dtype=bool)
        if outcome is None
        else outcome.notna().to_numpy(dtype=bool)
    )
    labels = train_y if outcome is None else outcome.loc[usable].astype(str).to_numpy()
    prepared_train, prepared_valid, categorical, numeric = _prepare_realmlp_frames(
        train_x.loc[usable], valid_x
    )
    if architecture in {"tabr", "realtabr"}:
        # PyTabKit 1.7.3's TabR categorical re-encoder is incompatible with
        # scikit-learn 1.8 when its intermediate missing code is zero: the
        # constant imputer drops every categorical column.  Freeze train-only
        # ordinal maps and expose them as numerical coordinates instead.  No
        # validation/test category can affect these maps.
        for column in categorical:
            categories = {
                value: index + 1
                for index, value in enumerate(
                    sorted(prepared_train[column].astype(str).unique())
                )
            }
            prepared_train[column] = prepared_train[column].map(categories).fillna(0).to_numpy(
                dtype=np.float32
            )
            prepared_valid[column] = prepared_valid[column].map(categories).fillna(0).to_numpy(
                dtype=np.float32
            )
            numeric.append(column)
        categorical = []
    print(
        f"[{label}] {architecture} fit={len(prepared_train):,}/{len(train_x):,} rows, "
        f"features={prepared_train.shape[1]}",
        flush=True,
    )
    model = model_class(**settings)
    started = time.perf_counter()
    model.fit(prepared_train, labels, cat_col_names=categorical)
    fit_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    probabilities = np.asarray(model.predict_proba(prepared_valid), dtype=np.float64)
    predict_seconds = time.perf_counter() - prediction_started
    classes = [str(value) for value in model.classes_]
    if outcome is None:
        positive = classes.index("1") if "1" in classes else 1
        prediction = probabilities[:, positive]
        success_indices = [positive]
    else:
        success_indices = [
            index for index, value in enumerate(classes)
            if value == "success" or value.startswith("success|")
        ]
        if not success_indices:
            raise RuntimeError(f"RealMLP outcome classes have no success label: {classes}")
        prediction = probabilities[:, success_indices].sum(axis=1)
    details: dict[str, Any] = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "settings": settings,
        "outcome_classes": classes,
        "success_class_count": len(success_indices),
        "fit_rows": int(len(prepared_train)),
        "categorical_features": categorical,
        "numeric_feature_count": len(numeric),
        "prediction_std": float(np.std(prediction)),
        "row_independent_preprocessing": True,
        "train_only_numeric_imputation": True,
        "backend": f"pytabkit.{model_class.__name__}",
        "architecture": architecture,
    }
    if save_components:
        details["_component_predictions"] = {
            f"p_{index}_{value}": probabilities[:, index]
            for index, value in enumerate(classes)
        }
    del model, prepared_train, prepared_valid, probabilities
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return np.asarray(prediction, dtype=np.float64), details


def fit_deep_outcome_model(
    label: str,
    architecture: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    valid_x: pd.DataFrame,
    params: dict[str, Any] | None = None,
    outcome_scheme: str = "reverse_any",
    train_weight: np.ndarray | None = None,
    outcome_labels: pd.Series | None = None,
    save_components: bool = False,
    valid_binary_y: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a neural multiclass outcome model and sum its success classes."""
    outcome = (
        derive_control_outcome_labels(history, outcome_scheme)
        if outcome_labels is None
        else outcome_labels.reindex(history.index)
    )
    usable = outcome.notna().to_numpy(dtype=bool)
    classes = sorted(str(value) for value in pd.unique(outcome.loc[usable].astype(str)))
    class_index = {value: index for index, value in enumerate(classes)}
    targets = outcome.loc[usable].astype(str).map(class_index).to_numpy(dtype=np.int64)
    success_indices = [
        index for index, value in enumerate(classes)
        if value == "success" or value.startswith("success|")
    ]
    if not success_indices:
        raise ValueError(f"Outcome model has no success class: {classes}")
    settings = dict(params or {})
    settings["_num_classes"] = len(classes)
    settings["_success_indices"] = success_indices
    if settings.get("loss") in {None, "bce", "bce_brier", "brier"}:
        settings["loss"] = "ce_brier"
    model = TorchTabularModel(list(train_x.columns), architecture, settings)
    print(
        f"[{label}] neural outcome fit={int(usable.sum()):,}/{len(train_x):,} rows, "
        f"classes={len(classes)}, features={train_x.shape[1]}",
        flush=True,
    )
    started = time.perf_counter()
    usable_weight = train_weight[usable] if train_weight is not None else None
    outer_selection = bool(settings.get("outer_epoch_selection", False))
    if outer_selection:
        if valid_binary_y is None:
            raise ValueError("outer_epoch_selection requires validation targets")
        model.fit_with_binary_eval(
            train_x.loc[usable], targets, valid_x, valid_binary_y,
            sample_weight=usable_weight,
        )
    else:
        model.fit(
            train_x.loc[usable], targets, sample_weight=usable_weight,
        )
    fit_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    member_selection: dict[str, Any] = {"enabled": False}
    requested_members = settings.get("selected_members")
    select_members = bool(settings.get("outer_member_selection", False))
    if architecture.startswith("tabm") and (
        requested_members is not None or select_members
    ):
        member_probabilities = model.predict_member_proba(valid_x)
        if member_probabilities.ndim != 3:
            raise RuntimeError(
                "Multiclass TabM member output must have shape rows x members x classes"
            )
        member_count = member_probabilities.shape[1]
        if requested_members is not None:
            selected_members = [int(value) for value in requested_members]
            if (
                not selected_members
                or min(selected_members) < 0
                or max(selected_members) >= member_count
            ):
                raise ValueError(
                    f"selected_members must be within [0, {member_count - 1}]"
                )
            selection_path: list[dict[str, Any]] = []
            selection_source = "fixed_from_prior_development_fold"
        else:
            if valid_binary_y is None:
                raise ValueError("outer_member_selection requires validation targets")
            binary_members = member_probabilities[:, :, success_indices].sum(axis=2)
            target = np.asarray(valid_binary_y, dtype=np.float64)
            maximum = min(
                member_count, int(settings.get("max_selected_members", member_count))
            )
            running = np.zeros(len(target), dtype=np.float64)
            available = np.ones(member_count, dtype=bool)
            greedy: list[int] = []
            selection_path = []
            for step in range(maximum):
                candidate_prediction = (
                    running[:, None] + binary_members
                ) / float(step + 1)
                candidate_brier = np.mean(
                    np.square(candidate_prediction - target[:, None]), axis=0
                )
                candidate_brier[~available] = np.inf
                chosen = int(np.argmin(candidate_brier))
                greedy.append(chosen)
                running += binary_members[:, chosen]
                available[chosen] = False
                selection_path.append({
                    "members": list(greedy),
                    "brier": float(candidate_brier[chosen]),
                })
            best_step = int(np.argmin([
                item["brier"] for item in selection_path
            ]))
            selected_members = greedy[: best_step + 1]
            selection_source = "2024_outer_development_greedy_subset"
        probabilities = member_probabilities[:, selected_members, :].mean(axis=1)
        member_selection = {
            "enabled": True,
            "source": selection_source,
            "member_count": int(member_count),
            "selected_members": selected_members,
            "selected_count": len(selected_members),
            "path": selection_path,
        }
        del member_probabilities
    else:
        probabilities = model.predict_proba(valid_x)
    predict_seconds = time.perf_counter() - prediction_started
    prediction = probabilities[:, success_indices].sum(axis=1).astype(np.float64)
    counts = outcome.loc[usable].value_counts().to_dict()
    torch = model._torch()
    details: dict[str, Any] = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_iter": model.n_iter_,
        "architecture": architecture,
        "device": model.device_,
        "torch_version": torch.__version__,
        "model_params": settings,
        "prediction_std": float(np.std(prediction)),
        "outcome_classes": classes,
        "success_class_count": len(success_indices),
        "outcome_counts": {str(key): int(value) for key, value in counts.items()},
        "outcome_usable_rows": int(usable.sum()),
        "outcome_dropped_rows": int((~usable).sum()),
        "outcome_scheme": outcome_scheme,
        "sample_weighted": bool(train_weight is not None),
        "training_label_source": "next historical same-pitcher as-of counter delta",
        "categorical_features": list(model.categorical_),
        "numeric_feature_count": len(model.numeric_),
        "categorical_cardinalities": {
            column: int(len(categories) + 1)
            for column, categories in model.categories_.items()
        },
        "training_history": model.training_history_,
        "row_independent_preprocessing": True,
        "outer_epoch_selection": outer_selection,
        "outer_epoch_selection_note": (
            "2024 development-fold hyperparameter selection; exploratory"
            if outer_selection else None
        ),
        "member_selection": member_selection,
    }
    if save_components:
        details["_component_predictions"] = {
            f"p_{index}_{value.replace('|', '_').replace(' ', '_')}": (
                probabilities[:, index].astype(np.float64)
            )
            for index, value in enumerate(classes)
        }
    del model, probabilities, outcome
    gc.collect()
    return prediction, details


def fit_leaf_refit_model(
    label: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    valid_x: pd.DataFrame,
    valid_context: pd.DataFrame,
    params: dict[str, Any] | None = None,
    outcome_scheme: str = "reverse_any",
    train_weight: np.ndarray | None = None,
    outcome_labels: pd.Series | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refit stable CatBoost leaves on the latest completed season.

    The tree structure and categorical target statistics are learned on seasons
    strictly before the adaptation season.  Only smoothed binary target means
    inside those frozen leaves are estimated on the latest completed season.
    Validation rows perform immutable leaf-table lookups.
    """
    from catboost import CatBoostClassifier

    supplied = dict(params or {})
    refit_k = float(supplied.pop("leaf_refit_k", 200.0))
    refit_segment = str(supplied.pop("leaf_refit_segment", "global"))
    refit_blend = float(supplied.pop("leaf_refit_blend", 0.5))
    adapt_seasons = int(supplied.pop("adapt_seasons", 1))
    if refit_k <= 0 or adapt_seasons < 1:
        raise ValueError("leaf_refit_k and adapt_seasons must be positive")
    if refit_segment not in {"global", "game_type", "count", "type_count"}:
        raise ValueError(f"Unknown leaf_refit_segment: {refit_segment}")
    if not 0.0 <= refit_blend <= 1.0:
        raise ValueError("leaf_refit_blend must be in [0, 1]")

    seasons = history[SEASON].to_numpy(dtype=np.int16, copy=False)
    latest = int(seasons.max())
    adapt_start = latest - adapt_seasons + 1
    older = seasons < adapt_start
    adapt = seasons >= adapt_start
    if not np.any(older) or not np.any(adapt):
        raise ValueError("Leaf refit requires both stable and adaptation seasons")
    outcome = (
        derive_control_outcome_labels(history, outcome_scheme)
        if outcome_labels is None
        else outcome_labels.reindex(history.index)
    )
    usable = outcome.notna().to_numpy(dtype=bool)
    stable = older & usable
    settings: dict[str, Any] = {
        "loss_function": "MultiClass",
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 12.0,
        "random_seed": RANDOM_SEED,
        "allow_writing_files": False,
        "thread_count": 6,
        "task_type": (
            "GPU" if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu"
            else "CPU"
        ),
    }
    settings.update(supplied)
    categorical = [column for column in BOOSTER_CATEGORICAL if column in train_x]
    model = CategoricalFrameModel(
        CatBoostClassifier(**settings), categorical, "catboost"
    )
    print(
        f"[{label}] stable tree={int(stable.sum()):,}, "
        f"recent refit={int(adapt.sum()):,}, features={train_x.shape[1]}",
        flush=True,
    )
    started = time.perf_counter()
    model.fit(
        train_x.loc[stable], outcome.loc[stable].astype(str).to_numpy(),
        sample_weight=(train_weight[stable] if train_weight is not None else None),
    )
    fit_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    adapt_x = train_x.loc[adapt]
    prepared_adapt = model._prepare(adapt_x, fitting=False)
    prepared_valid = model._prepare(valid_x, fitting=False)
    adapt_leaves = np.asarray(
        model.estimator.calc_leaf_indexes(prepared_adapt, thread_count=6),
        dtype=np.int32,
    )
    valid_leaves = np.asarray(
        model.estimator.calc_leaf_indexes(prepared_valid, thread_count=6),
        dtype=np.int32,
    )
    base_probabilities = model.estimator.predict_proba(prepared_valid)
    classes = [str(value) for value in model.estimator.classes_]
    success_indices = [
        index for index, value in enumerate(classes)
        if value == "success" or value.startswith("success|")
    ]
    if not success_indices:
        raise ValueError(f"Stable outcome tree has no success class: {classes}")
    base_prediction = base_probabilities[:, success_indices].sum(axis=1).astype(
        np.float64
    )
    adapt_y = history.loc[adapt, TARGET].to_numpy(dtype=np.float64, copy=False)
    adapt_context = history.loc[adapt]

    def segment_codes(frame: pd.DataFrame) -> tuple[np.ndarray, int]:
        if refit_segment == "global":
            return np.zeros(len(frame), dtype=np.int32), 1
        type_code = frame["game_type"].astype(str).map({"R": 0, "F": 1}).fillna(2)
        type_code = type_code.to_numpy(dtype=np.int32)
        balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(0)
        strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(0)
        count_code = (
            balls.to_numpy(dtype=np.int32) * 3
            + strikes.to_numpy(dtype=np.int32)
        )
        if refit_segment == "game_type":
            return type_code, 3
        if refit_segment == "count":
            return count_code, 12
        return type_code * 12 + count_code, 36

    adapt_segment, segment_count = segment_codes(adapt_context)
    valid_segment, valid_segment_count = segment_codes(valid_context)
    if valid_segment_count != segment_count:
        raise AssertionError("Adaptation and validation segment dimensions differ")
    overall_prior = float(adapt_y.mean())
    segment_n = np.bincount(adapt_segment, minlength=segment_count).astype(np.float64)
    segment_s = np.bincount(
        adapt_segment, weights=adapt_y, minlength=segment_count
    ).astype(np.float64)
    segment_prior = (segment_s + refit_k * overall_prior) / (
        segment_n + refit_k
    )
    leaf_prediction = np.zeros(len(valid_x), dtype=np.float64)
    for tree_index in range(adapt_leaves.shape[1]):
        leaf_count = int(max(
            adapt_leaves[:, tree_index].max(initial=0),
            valid_leaves[:, tree_index].max(initial=0),
        )) + 1
        adapt_key = (
            adapt_segment * leaf_count + adapt_leaves[:, tree_index]
        )
        valid_key = valid_segment * leaf_count + valid_leaves[:, tree_index]
        size = segment_count * leaf_count
        count = np.bincount(adapt_key, minlength=size).astype(np.float64)
        success = np.bincount(
            adapt_key, weights=adapt_y, minlength=size
        ).astype(np.float64)
        prior = np.repeat(segment_prior, leaf_count)
        table = (success + refit_k * prior) / (count + refit_k)
        leaf_prediction += table[valid_key]
    leaf_prediction /= float(adapt_leaves.shape[1])
    prediction = np.clip(
        (1.0 - refit_blend) * base_prediction
        + refit_blend * leaf_prediction,
        1e-6,
        1.0 - 1e-6,
    )
    predict_seconds = time.perf_counter() - prediction_started
    details: dict[str, Any] = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_iter": int(settings["iterations"]),
        "model_params": settings,
        "stable_seasons": sorted(
            int(value) for value in np.unique(seasons[older])
        ),
        "adaptation_seasons": sorted(
            int(value) for value in np.unique(seasons[adapt])
        ),
        "stable_outcome_rows": int(stable.sum()),
        "adaptation_rows": int(adapt.sum()),
        "leaf_refit_k": refit_k,
        "leaf_refit_segment": refit_segment,
        "leaf_refit_blend": refit_blend,
        "tree_count": int(adapt_leaves.shape[1]),
        "prediction_std": float(np.std(prediction)),
        "base_prediction_mean": float(base_prediction.mean()),
        "leaf_prediction_mean": float(leaf_prediction.mean()),
        "row_independent_inference": True,
        "adaptation_contract": (
            "frozen older-season tree; latest completed-season leaf target tables"
        ),
        "_component_predictions": {
            "stable_base": base_prediction,
            "recent_leaf": leaf_prediction,
        },
    }
    del (
        model, base_probabilities, adapt_leaves, valid_leaves,
        base_prediction, leaf_prediction, outcome,
    )
    gc.collect()
    return prediction, details


def fit_brier_model(
    label: str,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: pd.DataFrame,
    params: dict[str, Any] | None = None,
    train_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a squared-error probability regressor aligned to the Brier metric."""
    from catboost import CatBoostRegressor

    settings: dict[str, Any] = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 12.0,
        "random_seed": RANDOM_SEED,
        "allow_writing_files": False,
        "thread_count": 6,
        "task_type": (
            "GPU" if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu" else "CPU"
        ),
    }
    settings.update(params or {})
    categorical = [column for column in BOOSTER_CATEGORICAL if column in train_x.columns]
    model = CategoricalFrameModel(
        CatBoostRegressor(**settings), categorical, "catboost"
    )
    print(
        f"[{label}] Brier-regression fit={len(train_x):,} rows, "
        f"features={train_x.shape[1]}",
        flush=True,
    )
    started = time.perf_counter()
    model.fit(train_x, train_y, sample_weight=train_weight)
    fit_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    raw_prediction = np.asarray(model.predict(valid_x), dtype=np.float64)
    prediction = np.clip(raw_prediction, 1e-6, 1.0 - 1e-6)
    predict_seconds = time.perf_counter() - prediction_started
    details: dict[str, Any] = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_iter": settings["iterations"],
        "prediction_std": float(np.std(prediction)),
        "raw_prediction_min": float(raw_prediction.min()),
        "raw_prediction_max": float(raw_prediction.max()),
        "clipped_rows": int(np.sum((raw_prediction < 0.0) | (raw_prediction > 1.0))),
        "model_params": settings,
        "sample_weighted": bool(train_weight is not None),
        "objective_alignment": "RMSE on binary target equals Brier minimization",
    }
    if hasattr(model.estimator, "get_feature_importance"):
        importance = np.asarray(model.estimator.get_feature_importance(), dtype=np.float64)
        if len(importance) == len(train_x.columns):
            details["feature_importance"] = [
                {"feature": feature, "importance": float(value)}
                for feature, value in sorted(
                    zip(train_x.columns, importance),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
    export_catboost_model(
        label,
        model,
        list(train_x.columns),
        "regressor",
        {"clip_prediction": [1e-6, 1.0 - 1e-6]},
    )
    del model, raw_prediction
    gc.collect()
    return prediction, details


def _leave_one_out_eb_rate(
    frame: pd.DataFrame,
    target: np.ndarray,
    keys: list[str],
    prior: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a leave-one-row-out empirical-Bayes rate for historical rows."""
    if strength <= 0.0:
        raise ValueError("empirical-Bayes strength must be positive")
    work = frame[keys].copy()
    work["_group_soft_target"] = np.asarray(target, dtype=np.float64)
    grouped = work.groupby(keys, sort=False, observed=True, dropna=False)[
        "_group_soft_target"
    ]
    group_sum = grouped.transform("sum").to_numpy(dtype=np.float64)
    group_n = grouped.transform("size").to_numpy(dtype=np.float64)
    loo_n = group_n - 1.0
    rate = (
        group_sum - np.asarray(target, dtype=np.float64)
        + strength * np.asarray(prior, dtype=np.float64)
    ) / (loo_n + strength)
    return np.clip(rate, 1e-4, 1.0 - 1e-4), group_n


def fit_group_soft_brier_model(
    label: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: pd.DataFrame,
    params: dict[str, Any] | None = None,
    train_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit Brier regression to a historical hierarchical soft target.

    Every target statistic is built exclusively inside the already-completed
    historical slice.  Each row is removed from its own aggregates.  The
    hierarchy denoises pitch-level Bernoulli labels into a stable probability
    forecast while inference remains a normal row-independent CatBoost model.
    """
    supplied = dict(params or {})
    hard_fraction = float(supplied.pop("group_soft_hard_fraction", 0.25))
    base_k = float(supplied.pop("group_soft_base_k", 512.0))
    pitcher_k = float(supplied.pop("group_soft_pitcher_k", 96.0))
    batter_k = float(supplied.pop("group_soft_batter_k", 160.0))
    context_k = float(supplied.pop("group_soft_context_k", 384.0))
    local_k = float(supplied.pop("group_soft_local_k", 32.0))
    if not 0.0 <= hard_fraction <= 1.0:
        raise ValueError("group_soft_hard_fraction must be in [0, 1]")
    required = {
        SEASON, PITCHER, "batter_id", "game_type", "balls_before",
        "strikes_before", "pitcher_hand", BATTER_HAND,
    }
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"group-soft history columns are missing: {missing}")

    hard = np.asarray(train_y, dtype=np.float64)
    # A row-specific game-type leave-one-out prior protects the smallest first
    # season fold without borrowing any future season or validation labels.
    neutral = np.full(len(history), 0.5, dtype=np.float64)
    game_type_prior, game_type_n = _leave_one_out_eb_rate(
        history, hard, ["game_type"], neutral, 4.0
    )
    base_rate, base_n = _leave_one_out_eb_rate(
        history, hard, [SEASON, "game_type"], game_type_prior, base_k
    )
    pitcher_rate, pitcher_n = _leave_one_out_eb_rate(
        history, hard, [SEASON, PITCHER, "game_type"], base_rate, pitcher_k
    )
    batter_rate, batter_n = _leave_one_out_eb_rate(
        history, hard, [SEASON, "batter_id", "game_type"], base_rate, batter_k
    )
    context_rate, context_n = _leave_one_out_eb_rate(
        history,
        hard,
        [
            SEASON, "game_type", "balls_before", "strikes_before",
            "pitcher_hand", BATTER_HAND,
        ],
        base_rate,
        context_k,
    )

    def logit(value: np.ndarray) -> np.ndarray:
        clipped = np.clip(value, 0.02, 0.98)
        return np.log(clipped / (1.0 - clipped))

    additive_prior = 1.0 / (
        1.0
        + np.exp(
            -(
                logit(pitcher_rate)
                + logit(batter_rate)
                + logit(context_rate)
                - 2.0 * logit(base_rate)
            )
        )
    )
    local_rate, local_n = _leave_one_out_eb_rate(
        history,
        hard,
        [
            SEASON, PITCHER, "game_type", "balls_before", "strikes_before",
            BATTER_HAND,
        ],
        additive_prior,
        local_k,
    )
    soft_target = hard_fraction * hard + (1.0 - hard_fraction) * local_rate
    prediction, details = fit_brier_model(
        label, train_x, soft_target, valid_x, supplied, train_weight
    )
    details.update(
        {
            "training_target": "hierarchical_leave_one_out_empirical_bayes",
            "hard_label_fraction": hard_fraction,
            "soft_target_mean": float(soft_target.mean()),
            "soft_target_std": float(soft_target.std()),
            "local_rate_mean": float(local_rate.mean()),
            "local_rate_std": float(local_rate.std()),
            "hard_soft_correlation": float(np.corrcoef(hard, local_rate)[0, 1]),
            "strengths": {
                "base": base_k,
                "pitcher": pitcher_k,
                "batter": batter_k,
                "context": context_k,
                "local": local_k,
            },
            "mean_group_sizes": {
                "game_type": float(game_type_n.mean()),
                "season_game_type": float(base_n.mean()),
                "pitcher_season_game_type": float(pitcher_n.mean()),
                "batter_season_game_type": float(batter_n.mean()),
                "season_count_hands": float(context_n.mean()),
                "pitcher_season_count_batter_hand": float(local_n.mean()),
            },
            "validation_labels_used_for_target_or_fit": False,
            "row_independent_inference": True,
            "objective_alignment": "RMSE on a denoised historical probability target",
        }
    )
    return prediction, details


def fit_state_residual_model(
    label: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: pd.DataFrame,
    params: dict[str, Any] | None = None,
    train_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a season-centered Brier residual around a row-local state prior.

    The base probability is reconstructed from the current row's official
    as-of counter and a pre-season frozen state (normally E14).  Centering the
    residual separately within each historical season prevents the regressor
    from learning a stale global intercept while retaining stable contextual
    deviations.  Validation receives no target-derived centering constant.
    """
    from catboost import CatBoostRegressor

    supplied = dict(params or {})
    base_column = str(supplied.pop("state_base_column", "e14_rate_season"))
    residual_scale = float(supplied.pop("residual_scale", 1.0))
    center_by = str(supplied.pop("residual_center_by", "season"))
    if base_column not in train_x or base_column not in valid_x:
        raise ValueError(f"state residual base column is missing: {base_column}")
    if center_by not in {"season", "season_game_type"}:
        raise ValueError(
            "residual_center_by must be 'season' or 'season_game_type'"
        )
    if not 0.0 < residual_scale <= 2.0:
        raise ValueError("residual_scale must be in (0, 2]")

    base_train = pd.to_numeric(train_x[base_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    base_valid = pd.to_numeric(valid_x[base_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(base_train).all() or not np.isfinite(base_valid).all():
        raise ValueError(f"non-finite state base probability in {base_column}")
    residual = np.asarray(train_y, dtype=np.float64) - base_train
    center_frame = pd.DataFrame(
        {
            "season": history[SEASON].to_numpy(dtype=np.int16, copy=False),
            "game_type": history["game_type"].astype(str).to_numpy(),
            "residual": residual,
        },
        index=history.index,
    )
    group_columns = ["season"]
    if center_by == "season_game_type":
        group_columns.append("game_type")
    centers = center_frame.groupby(
        group_columns, sort=True, observed=True
    )["residual"].transform("mean").to_numpy(dtype=np.float64)
    centered_residual = residual - centers
    center_table = center_frame.groupby(
        group_columns, sort=True, observed=True
    )["residual"].mean()

    settings: dict[str, Any] = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 500,
        "depth": 5,
        "learning_rate": 0.05,
        "l2_leaf_reg": 20.0,
        "random_seed": RANDOM_SEED,
        "allow_writing_files": False,
        "thread_count": 6,
        "task_type": (
            "GPU"
            if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu"
            else "CPU"
        ),
    }
    settings.update(supplied)
    categorical = [column for column in BOOSTER_CATEGORICAL if column in train_x]
    model = CategoricalFrameModel(
        CatBoostRegressor(**settings), categorical, "catboost"
    )
    print(
        f"[{label}] state-residual fit={len(train_x):,} rows, "
        f"features={train_x.shape[1]}, base={base_column}",
        flush=True,
    )
    started = time.perf_counter()
    model.fit(train_x, centered_residual, sample_weight=train_weight)
    fit_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    correction = np.asarray(model.predict(valid_x), dtype=np.float64)
    prediction = np.clip(
        base_valid + residual_scale * correction, 1e-6, 1.0 - 1e-6
    )
    predict_seconds = time.perf_counter() - prediction_started
    details: dict[str, Any] = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_iter": int(settings["iterations"]),
        "model_params": settings,
        "state_base_column": base_column,
        "residual_scale": residual_scale,
        "residual_center_by": center_by,
        "historical_centers": {
            "|".join(
                str(part) for part in (key if isinstance(key, tuple) else (key,))
            ): float(value)
            for key, value in center_table.items()
        },
        "centered_target_mean": float(centered_residual.mean()),
        "centered_target_std": float(centered_residual.std()),
        "base_prediction_mean": float(base_valid.mean()),
        "correction_mean": float(correction.mean()),
        "correction_std": float(correction.std()),
        "correction_max_abs": float(np.max(np.abs(correction))),
        "prediction_mean": float(prediction.mean()),
        "row_independent_inference": True,
        "validation_target_center_used": False,
        "_component_predictions": {
            "state_base": base_valid,
            "centered_correction": correction,
        },
    }
    if hasattr(model.estimator, "get_feature_importance"):
        importance = np.asarray(
            model.estimator.get_feature_importance(), dtype=np.float64
        )
        if len(importance) == len(train_x.columns):
            details["feature_importance"] = [
                {"feature": feature, "importance": float(value)}
                for feature, value in sorted(
                    zip(train_x.columns, importance),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
    del model, correction, centered_residual, residual, centers
    gc.collect()
    return prediction, details


def fit_multi_brier_model(
    label: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    valid_x: pd.DataFrame,
    params: dict[str, Any] | None = None,
    outcome_scheme: str = "reverse_any",
    train_weight: np.ndarray | None = None,
    outcome_labels: pd.Series | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Jointly regress one-hot outcome components with a squared-error loss."""
    from catboost import CatBoostRegressor

    outcome = (
        derive_control_outcome_labels(history, outcome_scheme)
        if outcome_labels is None
        else outcome_labels.reindex(history.index)
    )
    usable = outcome.notna().to_numpy(dtype=bool)
    classes = sorted(str(value) for value in outcome.loc[usable].unique())
    if "success" not in classes:
        raise ValueError(
            "catboost_multi_brier requires an unsuffixed success class; "
            f"got {classes}"
        )
    class_index = {value: index for index, value in enumerate(classes)}
    encoded = np.zeros((int(usable.sum()), len(classes)), dtype=np.float32)
    encoded[
        np.arange(len(encoded)),
        [class_index[str(value)] for value in outcome.loc[usable].to_numpy()],
    ] = 1.0
    supplied = dict(params or {})
    prediction_normalization = str(
        supplied.pop("prediction_normalization", "simplex")
    )
    if prediction_normalization not in {"simplex", "raw_success"}:
        raise ValueError(
            "prediction_normalization must be 'simplex' or 'raw_success'"
        )
    settings: dict[str, Any] = {
        "loss_function": "MultiRMSE",
        "eval_metric": "MultiRMSE",
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 12.0,
        "random_seed": RANDOM_SEED,
        "allow_writing_files": False,
        "thread_count": 6,
        "task_type": (
            "GPU" if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu" else "CPU"
        ),
    }
    settings.update(supplied)
    categorical = [column for column in BOOSTER_CATEGORICAL if column in train_x.columns]
    model = CategoricalFrameModel(
        CatBoostRegressor(**settings), categorical, "catboost"
    )
    print(
        f"[{label}] Multi-Brier fit={int(usable.sum()):,}/{len(train_x):,} rows, "
        f"features={train_x.shape[1]}, outcomes={len(classes)}",
        flush=True,
    )
    started = time.perf_counter()
    model.fit(
        train_x.loc[usable], encoded,
        sample_weight=(train_weight[usable] if train_weight is not None else None),
    )
    fit_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    raw = np.asarray(model.predict(valid_x), dtype=np.float64)
    clipped = np.clip(raw, 1e-9, None)
    normalized = clipped / clipped.sum(axis=1, keepdims=True)
    if prediction_normalization == "simplex":
        success_raw = normalized[:, class_index["success"]]
    else:
        success_raw = raw[:, class_index["success"]]
    prediction = np.clip(success_raw, 1e-6, 1.0 - 1e-6)
    predict_seconds = time.perf_counter() - prediction_started
    details: dict[str, Any] = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_iter": settings["iterations"],
        "prediction_std": float(np.std(prediction)),
        "outcome_scheme": outcome_scheme,
        "outcome_classes": classes,
        "outcome_usable_rows": int(usable.sum()),
        "outcome_dropped_rows": int((~usable).sum()),
        "raw_component_min": float(raw.min()),
        "raw_component_max": float(raw.max()),
        "raw_row_sum_mean": float(raw.sum(axis=1).mean()),
        "negative_component_rows": int(np.sum(np.any(raw < 0.0, axis=1))),
        "model_params": settings,
        "sample_weighted": bool(train_weight is not None),
        "objective_alignment": (
            "joint one-hot MultiRMSE; " + prediction_normalization
        ),
        "prediction_normalization": prediction_normalization,
    }
    if hasattr(model.estimator, "get_feature_importance"):
        importance = np.asarray(model.estimator.get_feature_importance(), dtype=np.float64)
        if len(importance) == len(train_x.columns):
            details["feature_importance"] = [
                {"feature": feature, "importance": float(value)}
                for feature, value in sorted(
                    zip(train_x.columns, importance),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
    del model, raw, clipped, normalized, encoded, outcome
    gc.collect()
    return prediction, details


def fit_dense_multitask_brier_model(
    label: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    valid_x: pd.DataFrame,
    component15_labels: pd.Series,
    params: dict[str, Any] | None = None,
    train_weight: np.ndarray | None = None,
    save_components: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Share CatBoost trees across control and dense auxiliary pitch tasks.

    The primary head is the official binary target.  Auxiliary heads are the
    four official as-of counter increments and the three dense pitch-group
    indicators reconstructed strictly inside completed training history.
    Unlike a Cartesian joint classifier, MultiRMSE can share splits without
    fragmenting rows into rare outcome-by-pitch classes.
    """
    from catboost import CatBoostRegressor

    components = component15_labels.reindex(history.index).astype("string")
    bits = components.str.extract(
        r"\|r([01])m([01])b([01])s([01])", expand=True
    )
    bits.columns = ["reverse", "middle", "ball", "strike"]
    groups = ("fastball", "breaking", "offspeed")
    group_labels = derive_dense_pitch_group_labels(history)
    usable_series = bits.notna().all(axis=1) & group_labels.isin(groups)
    usable = usable_series.to_numpy(dtype=bool)
    coverage = float(usable.mean())
    if coverage < 0.995:
        raise ValueError(
            f"dense multitask history coverage fell below 0.995: {coverage}"
        )

    supplied = dict(params or {})
    success_scale = float(supplied.pop("multitask_success_scale", 2.0))
    if not 1.0 <= success_scale <= 4.0:
        raise ValueError("multitask_success_scale must be in [1, 4]")
    target_names = [
        "success_scaled", "reverse", "middle", "ball", "strike",
        *groups,
    ]
    target = np.column_stack(
        [
            history.loc[usable_series, TARGET].to_numpy(dtype=np.float32)
            * success_scale,
            *[
                bits.loc[usable_series, column].to_numpy(dtype=np.float32)
                for column in ("reverse", "middle", "ball", "strike")
            ],
            *[
                group_labels.loc[usable_series].eq(group).to_numpy(
                    dtype=np.float32
                )
                for group in groups
            ],
        ]
    )
    settings: dict[str, Any] = {
        "loss_function": "MultiRMSE",
        "eval_metric": "MultiRMSE",
        "iterations": 700,
        "depth": 7,
        "learning_rate": 0.04,
        "l2_leaf_reg": 30.0,
        "random_seed": RANDOM_SEED,
        "allow_writing_files": False,
        "thread_count": 6,
        "task_type": (
            "GPU"
            if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu"
            else "CPU"
        ),
    }
    settings.update(supplied)
    categorical = [
        column for column in BOOSTER_CATEGORICAL if column in train_x.columns
    ]
    model = CategoricalFrameModel(
        CatBoostRegressor(**settings), categorical, "catboost"
    )
    print(
        f"[{label}] dense multitask fit={int(usable.sum()):,}/"
        f"{len(train_x):,} rows, features={train_x.shape[1]}, "
        f"heads={len(target_names)}",
        flush=True,
    )
    started = time.perf_counter()
    model.fit(
        train_x.loc[usable],
        target,
        sample_weight=(
            train_weight[usable] if train_weight is not None else None
        ),
    )
    fit_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    raw = np.asarray(model.predict(valid_x), dtype=np.float64)
    prediction = np.clip(
        raw[:, 0] / success_scale, 1e-6, 1.0 - 1e-6
    )
    predict_seconds = time.perf_counter() - prediction_started
    regular = history["game_type"].eq("R")
    details: dict[str, Any] = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_iter": int(settings["iterations"]),
        "model_params": settings,
        "architecture": "shared_tree_dense_multitask_multirmse",
        "target_heads": target_names,
        "success_scale": success_scale,
        "history_usable_rows": int(usable.sum()),
        "history_dropped_rows": int((~usable).sum()),
        "history_usable_coverage": coverage,
        "history_dense_group_coverage": float(group_labels.notna().mean()),
        "history_dense_group_coverage_R": float(
            group_labels.loc[regular].notna().mean()
        ),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "sample_weighted": bool(train_weight is not None),
        "current_pitch_group_used_at_inference": False,
        "validation_auxiliary_labels_used": False,
        "row_independent_inference": True,
    }
    if save_components:
        details["_component_predictions"] = {
            "raw_success_unscaled": raw[:, 0] / success_scale,
            **{
                f"aux_{name}": raw[:, index]
                for index, name in enumerate(target_names[1:], start=1)
            },
        }
    if hasattr(model.estimator, "get_feature_importance"):
        importance = np.asarray(
            model.estimator.get_feature_importance(), dtype=np.float64
        )
        if len(importance) == len(train_x.columns):
            details["feature_importance"] = [
                {"feature": feature, "importance": float(value)}
                for feature, value in sorted(
                    zip(train_x.columns, importance),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
    del model, raw, target, bits, group_labels, components
    gc.collect()
    return prediction, details


def fit_neural_dense_multitask_model(
    label: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    valid_x: pd.DataFrame,
    component15_labels: pd.Series,
    params: dict[str, Any] | None = None,
    train_weight: np.ndarray | None = None,
    save_components: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit eight Bernoulli tasks through one shared TabM representation."""
    components = component15_labels.reindex(history.index).astype("string")
    bits = components.str.extract(
        r"\|r([01])m([01])b([01])s([01])", expand=True
    )
    bits.columns = ["reverse", "middle", "ball", "strike"]
    groups = ("fastball", "breaking", "offspeed")
    group_labels = derive_dense_pitch_group_labels(history)
    usable_series = bits.notna().all(axis=1) & group_labels.isin(groups)
    usable = usable_series.to_numpy(dtype=bool)
    coverage = float(usable.mean())
    if coverage < 0.995:
        raise ValueError(
            f"neural dense multitask coverage fell below 0.995: {coverage}"
        )
    target_names = [
        "success", "reverse", "middle", "ball", "strike", *groups
    ]
    target = np.column_stack(
        [
            history.loc[usable_series, TARGET].to_numpy(dtype=np.float32),
            *[
                bits.loc[usable_series, column].to_numpy(dtype=np.float32)
                for column in ("reverse", "middle", "ball", "strike")
            ],
            *[
                group_labels.loc[usable_series].eq(group).to_numpy(
                    dtype=np.float32
                )
                for group in groups
            ],
        ]
    ).astype(np.float32, copy=False)
    settings = dict(params or {})
    configured_weights = settings.get(
        "multilabel_head_weights",
        [4.0, 0.75, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5],
    )
    if len(configured_weights) != len(target_names):
        raise ValueError("dense multitask head-weight count mismatch")
    settings["multilabel_head_weights"] = [
        float(value) for value in configured_weights
    ]
    settings["_multilabel_outputs"] = len(target_names)
    model = TorchTabularModel(list(train_x.columns), "tabm", settings)
    print(
        f"[{label}] neural dense multitask fit={int(usable.sum()):,}/"
        f"{len(train_x):,} rows, features={train_x.shape[1]}, "
        f"heads={len(target_names)}",
        flush=True,
    )
    started = time.perf_counter()
    model.fit(
        train_x.loc[usable],
        target,
        sample_weight=(
            train_weight[usable] if train_weight is not None else None
        ),
    )
    fit_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    probabilities = np.asarray(model.predict_proba(valid_x), dtype=np.float64)
    if probabilities.shape != (len(valid_x), len(target_names)):
        raise RuntimeError(
            "neural dense multitask probability shape mismatch: "
            f"{probabilities.shape}"
        )
    prediction = np.clip(probabilities[:, 0], 1e-6, 1.0 - 1e-6)
    predict_seconds = time.perf_counter() - prediction_started
    regular = history["game_type"].eq("R")
    details: dict[str, Any] = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_iter": int(model.n_iter_ or settings.get("epochs", 0)),
        "model_params": {
            key: value for key, value in settings.items()
            if not key.startswith("_")
        },
        "architecture": "tabm_shared_representation_eight_bernoulli_heads",
        "target_heads": target_names,
        "head_weights": settings["multilabel_head_weights"],
        "history_usable_rows": int(usable.sum()),
        "history_dropped_rows": int((~usable).sum()),
        "history_usable_coverage": coverage,
        "history_dense_group_coverage": float(group_labels.notna().mean()),
        "history_dense_group_coverage_R": float(
            group_labels.loc[regular].notna().mean()
        ),
        "training_history": model.training_history_,
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "sample_weighted": bool(train_weight is not None),
        "current_pitch_group_used_at_inference": False,
        "validation_auxiliary_labels_used": False,
        "row_independent_inference": True,
    }
    if save_components:
        details["_component_predictions"] = {
            f"head_{name}": probabilities[:, index]
            for index, name in enumerate(target_names)
        }
    del model, probabilities, target, components, bits, group_labels
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return prediction, details


def fit_pitch_gated_control_model(
    label: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    valid_x: pd.DataFrame,
    params: dict[str, Any] | None = None,
    train_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train a supervised soft pitch gate and pitch-specific control experts.

    Completed-history pitch groups supervise both the gate and the matching
    expert.  The deployable probability is optimized end to end as
    ``sum(gate_probability * expert_success_probability)`` and needs only the
    row-local legal feature vector at inference.
    """
    groups = ("fastball", "breaking", "offspeed")
    group_labels = derive_dense_pitch_group_labels(history)
    usable_series = group_labels.isin(groups)
    usable = usable_series.to_numpy(dtype=bool)
    coverage = float(usable.mean())
    if coverage < 0.995:
        raise ValueError(f"pitch-gated history coverage fell below 0.995: {coverage}")
    group_code = group_labels.loc[usable_series].map(
        {name: index for index, name in enumerate(groups)}
    ).to_numpy(dtype=np.float32)
    target = np.column_stack([
        history.loc[usable_series, TARGET].to_numpy(dtype=np.float32),
        group_code,
    ]).astype(np.float32, copy=False)
    settings = dict(params or {})
    settings["_pitch_gated"] = True
    model = TorchTabularModel(
        list(train_x.columns), "tabm_pitch_gated", settings
    )
    print(
        f"[{label}] supervised pitch-gated TabM fit={int(usable.sum()):,}/"
        f"{len(train_x):,} rows, features={train_x.shape[1]}",
        flush=True,
    )
    started = time.perf_counter()
    model.fit(
        train_x.loc[usable],
        target,
        sample_weight=(
            train_weight[usable] if train_weight is not None else None
        ),
    )
    fit_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    probability = np.asarray(model.predict_proba(valid_x), dtype=np.float64)
    if probability.shape != (len(valid_x), 2):
        raise RuntimeError(f"pitch-gated probability shape mismatch: {probability.shape}")
    prediction = np.clip(probability[:, 1], 1e-6, 1.0 - 1e-6)
    predict_seconds = time.perf_counter() - prediction_started
    details: dict[str, Any] = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_iter": int(model.n_iter_ or settings.get("epochs", 0)),
        "model_params": {
            key: value for key, value in settings.items()
            if not key.startswith("_")
        },
        "architecture": "supervised_soft_pitch_gate_times_control_experts",
        "pitch_groups": list(groups),
        "probability_formula": "sum_g softmax(gate)_g * sigmoid(expert_g)",
        "history_usable_rows": int(usable.sum()),
        "history_dropped_rows": int((~usable).sum()),
        "history_usable_coverage": coverage,
        "training_history": model.training_history_,
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "sample_weighted": bool(train_weight is not None),
        "current_pitch_group_used_at_inference": False,
        "validation_pitch_group_used": False,
        "validation_target_used_in_training": False,
        "row_independent_inference": True,
    }
    del model, probability, target, group_labels
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return prediction, details


def fit_component_pattern_moe_model(
    label: str,
    train_x: pd.DataFrame,
    history: pd.DataFrame,
    valid_x: pd.DataFrame,
    component15_labels: pd.Series,
    params: dict[str, Any] | None = None,
    train_weight: np.ndarray | None = None,
    save_components: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Marginalize conditional success over coherent counter patterns."""
    from catboost import CatBoostClassifier

    labels = component15_labels.reindex(history.index).astype("string")
    patterns = labels.str.extract(
        r"\|(r[01]m[01]b[01]s[01])", expand=False
    ).astype("string")
    usable = patterns.notna().to_numpy(dtype=bool)
    coverage = float(usable.mean())
    if coverage < 0.995:
        raise ValueError(
            f"component-pattern history coverage below 0.995: {coverage}"
        )
    supplied = dict(params or {})
    expert_seed_offset = int(supplied.pop("component_expert_seed_offset", 9100))
    base_seed = int(supplied.get("random_seed", RANDOM_SEED))
    common: dict[str, Any] = {
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 12.0,
        "random_seed": base_seed,
        "allow_writing_files": False,
        "thread_count": 6,
        "task_type": (
            "GPU"
            if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu"
            else "CPU"
        ),
    }
    common.update(supplied)
    categorical = [
        column for column in BOOSTER_CATEGORICAL if column in train_x.columns
    ]
    pattern_settings = {**common, "loss_function": "MultiClass"}
    gate = CategoricalFrameModel(
        CatBoostClassifier(**pattern_settings), categorical, "catboost"
    )
    print(
        f"[{label}] component gate fit={int(usable.sum()):,}/"
        f"{len(train_x):,} rows, features={train_x.shape[1]}",
        flush=True,
    )
    started = time.perf_counter()
    gate.fit(
        train_x.loc[usable],
        patterns.loc[usable].astype(str).to_numpy(),
        sample_weight=(
            train_weight[usable] if train_weight is not None else None
        ),
    )
    gate_fit_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    gate_probability = np.asarray(
        gate.predict_proba(valid_x), dtype=np.float64
    )
    gate_predict_seconds = time.perf_counter() - prediction_started
    gate_classes = [str(value) for value in gate.estimator.classes_]
    expected_classes = [
        f"r{reverse}m{middle}b{ball}s{strike}"
        for reverse in (0, 1)
        for middle in (0, 1)
        for ball, strike in ((0, 0), (0, 1), (1, 0))
    ]
    if sorted(gate_classes) != sorted(expected_classes):
        raise ValueError(f"unexpected component classes: {gate_classes}")

    eligible_patterns = ("r0m0b0s0", "r0m0b0s1", "r0m0b1s0")
    prediction = np.zeros(len(valid_x), dtype=np.float64)
    expert_predictions: dict[str, np.ndarray] = {}
    expert_details: dict[str, Any] = {}
    total_fit_seconds = gate_fit_seconds
    total_predict_seconds = gate_predict_seconds
    for expert_index, pattern in enumerate(eligible_patterns):
        mask = patterns.eq(pattern).fillna(False).to_numpy(dtype=bool)
        target = history.loc[mask, TARGET].to_numpy(dtype=np.int8)
        if len(target) < 10_000 or np.unique(target).size != 2:
            raise ValueError(
                f"component expert {pattern} is not estimable: rows={len(target)}"
            )
        expert_settings = {
            **common,
            "loss_function": "Logloss",
            "random_seed": base_seed + expert_seed_offset + expert_index,
        }
        expert = CategoricalFrameModel(
            CatBoostClassifier(**expert_settings), categorical, "catboost"
        )
        expert_started = time.perf_counter()
        expert.fit(
            train_x.loc[mask],
            target,
            sample_weight=(
                train_weight[mask] if train_weight is not None else None
            ),
        )
        expert_fit = time.perf_counter() - expert_started
        expert_prediction_started = time.perf_counter()
        expert_probability = np.asarray(
            expert.predict_proba(valid_x)[:, 1], dtype=np.float64
        )
        expert_predict = time.perf_counter() - expert_prediction_started
        gate_index = gate_classes.index(pattern)
        prediction += gate_probability[:, gate_index] * expert_probability
        expert_predictions[pattern] = expert_probability
        expert_details[pattern] = {
            "fit_rows": int(mask.sum()),
            "success_rate": float(target.mean()),
            "fit_seconds": expert_fit,
            "predict_seconds": expert_predict,
            "random_seed": int(expert_settings["random_seed"]),
        }
        total_fit_seconds += expert_fit
        total_predict_seconds += expert_predict
        del expert
        gc.collect()
    prediction = np.clip(prediction, 1e-6, 1.0 - 1e-6)
    details: dict[str, Any] = {
        "fit_seconds": total_fit_seconds,
        "predict_seconds": total_predict_seconds,
        "architecture": "component_pattern_gate_with_conditional_success_experts",
        "pattern_classes": sorted(gate_classes),
        "success_eligible_patterns": list(eligible_patterns),
        "history_pattern_coverage": coverage,
        "history_pattern_rows": int(usable.sum()),
        "history_pattern_dropped_rows": int((~usable).sum()),
        "experts": expert_details,
        "model_params": common,
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "current_validation_pattern_used": False,
        "row_independent_inference": True,
        "training_label_source": (
            "completed-history next same-pitcher official counter increments"
        ),
    }
    if save_components:
        components: dict[str, np.ndarray] = {
            f"gate_{pattern}": gate_probability[:, index]
            for index, pattern in enumerate(gate_classes)
        }
        components.update(
            {
                f"expert_{pattern}": value
                for pattern, value in expert_predictions.items()
            }
        )
        details["_component_predictions"] = components
    if hasattr(gate.estimator, "get_feature_importance"):
        importance = np.asarray(
            gate.estimator.get_feature_importance(), dtype=np.float64
        )
        if len(importance) == len(train_x.columns):
            details["gate_feature_importance"] = [
                {"feature": feature, "importance": float(value)}
                for feature, value in sorted(
                    zip(train_x.columns, importance),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
    del gate, gate_probability, patterns, labels
    gc.collect()
    return prediction, details


def run_fold(
    frame: pd.DataFrame,
    season: int,
    args: argparse.Namespace,
    params: dict | None,
    joined_trackman: pd.DataFrame | None = None,
    raw_trackman: pd.DataFrame | None = None,
    main_linkage_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    history_all = frame.loc[frame[SEASON] < season].copy()
    partial_trackman_meta: dict[str, Any] = {"enabled": False}
    expanded_trackman_profile_meta: dict[str, Any] = {"enabled": False}
    if (
        "partial_expanded_auto_pitch_latent" in args.features
        and "expanded_trackman_profiles" in args.features
    ):
        raise ValueError(
            "partial row linkage and full raw profile source are separate axes"
        )
    if "partial_expanded_auto_pitch_latent" in args.features:
        if (
            joined_trackman is None
            or raw_trackman is None
            or main_linkage_frame is None
        ):
            raise ValueError(
                "partial expanded auto pitch requires all linkage sources"
            )
        from experiments.v5_partial_trackman_linkage import (  # noqa: WPS433
            build_augmented_trackman_linkage,
        )

        joined_trackman, partial_trackman_meta = (
            build_augmented_trackman_linkage(
                main_linkage_frame,
                joined_trackman,
                raw_trackman,
                sorted(int(value) for value in history_all[SEASON].unique()),
            )
        )
        partial_trackman_meta = {"enabled": True, **partial_trackman_meta}
        partial_labels = (
            joined_trackman[["row_id", "auto_pitch_type"]]
            .drop_duplicates("row_id")
            .set_index("row_id")["auto_pitch_type"]
        )
        normalized_partial = (
            history_all["row_id"].astype(str).map(partial_labels)
            .astype("string")
            .replace(
                {
                    "Changeup": "ChangeUp",
                    "Four-Seam": "Fastball",
                    "SInker": "Sinker",
                }
            )
        )
        normalized_partial = normalized_partial.where(
            normalized_partial.isin(FINE_PITCH_TYPES[:-1]), "Other"
        ).where(normalized_partial.notna())
        history_all["auto_fine_pitch_type"] = normalized_partial.combine_first(
            history_all["auto_fine_pitch_type"].astype("string")
        )
        partial_trackman_meta["history_auto_label_coverage"] = float(
            history_all["auto_fine_pitch_type"].isin(FINE_PITCH_TYPES).mean()
        )
        del partial_labels, normalized_partial
    if "expanded_trackman_profiles" in args.features:
        if joined_trackman is None or raw_trackman is None:
            raise ValueError(
                "expanded TrackMan profiles require exact and raw sources"
            )
        from experiments.v5_expanded_trackman_profiles import (  # noqa: WPS433
            build_expanded_trackman_profile_source,
        )

        joined_trackman, expanded_trackman_profile_meta = (
            build_expanded_trackman_profile_source(
                joined_trackman,
                raw_trackman,
                sorted(int(value) for value in history_all[SEASON].unique()),
            )
        )
        expanded_trackman_profile_meta = {
            "enabled": True,
            **expanded_trackman_profile_meta,
        }
    outcome_labels_full = None
    component15_labels_full = None
    if any(
        name in {
            "catboost_outcome", "lgbm_outcome", "catboost_count_moe", "catboost_pitchtype_moe",
            "catboost_dense_pitchtype_moe", "catboost_dense_pitch_joint",
            "catboost_fine_pitch_joint",
            "catboost_physics_joint",
            "catboost_hier_pitch_joint",
            "catboost_auto_pitch_joint",
            "catboost_fine_pitch_moe",
            "catboost_dense_multitask",
            "tabm_dense_multitask",
            "catboost_multi_brier",
            "catboost_failure_chain",
            "deep_mlp_outcome", "deepfm_outcome", "tabtransformer_outcome",
            "tabm_outcome", "tabm_periodic_outcome",
            "tabm_piecewise_outcome",
            "realmlp_outcome",
            "tabr_outcome", "realtabr_outcome",
            "catboost_leaf_refit",
        }
        for name in args.models
    ):
        outcome_labels_full = derive_control_outcome_labels(
            history_all, args.outcome_scheme
        )
    if (
        "outcome_context" in args.features
        or "catboost_failure_decomp" in args.models
        or "catboost_dense_multitask" in args.models
        or "tabm_dense_multitask" in args.models
        or "catboost_component_pattern_moe" in args.models
    ):
        component15_labels_full = (
            outcome_labels_full
            if outcome_labels_full is not None and args.outcome_scheme == "component15"
            else derive_control_outcome_labels(history_all, "component15")
        )
    history = history_all
    if args.fit_game_types:
        history = history.loc[history["game_type"].isin(args.fit_game_types)]
    if args.history_window is not None:
        if args.history_window < 1:
            raise ValueError("--history-window must be >= 1")
        history = history.loc[history[SEASON] >= season - args.history_window]
    if args.fit_count_states:
        count_state = (
            history["balls_before"].astype(str)
            + "-"
            + history["strikes_before"].astype(str)
        )
        history = history.loc[count_state.isin(args.fit_count_states)]
    if not 0.0 < args.season_decay <= 1.0:
        raise ValueError("--season-decay must be in (0, 1]")
    if not 0.0 <= args.f_pre_regime_weight <= 1.0:
        raise ValueError("--f-pre-regime-weight must be in [0, 1]")
    if (
        args.f_regime_start is not None
        and season > args.f_regime_start
        and args.f_pre_regime_weight == 0.0
    ):
        history = history.loc[
            history["game_type"].ne("F") | history[SEASON].ge(args.f_regime_start)
        ]
    history = subsample(history.copy(), args.max_history_rows)
    valid = subsample(frame.loc[frame[SEASON] == season].copy(), args.max_valid_rows)
    if history.empty or valid.empty:
        raise ValueError(f"Empty fold for season {season}")

    prior = float(candidate_priors(history_all, season)[args.prior_mode])
    use_e14_hand_cells = "e14_hand_cells" in args.features
    use_e14_count_cells = "e14_count_cells" in args.features
    use_e14_type_count_cells = "e14_type_count_cells" in args.features
    use_pitcher_batter_interactions = (
        "pitcher_batter_season_interactions" in args.features
    )
    use_e14_rate_hand_bin = "e14_rate_hand_bin" in args.features
    use_e14_n_hand_bin = "e14_n_hand_bin" in args.features
    use_hierarchical_e14 = "hierarchical_e14" in args.features
    use_recent_form = (
        "recent_form" in args.features
        or "recent_form_count_cells" in args.features
    )
    use_recent_form_count_cells = "recent_form_count_cells" in args.features
    use_recent_denominators = "recent_denominators" in args.features
    use_recent_workload_decoder = "recent_workload_decoder" in args.features
    if use_recent_workload_decoder and not use_recent_denominators:
        raise ValueError(
            "recent_workload_decoder requires recent_denominators for its locked ablation"
        )
    use_current_state_context = "current_state_context" in args.features
    use_current_state_level = "current_state_level" in args.features
    use_current_state_full = (
        "current_state_full" in args.features
        or use_current_state_context
        or use_current_state_level
    )
    use_e14 = (
        "e14" in args.features
        or "e14_multi" in args.features
        or use_e14_hand_cells
        or use_e14_count_cells
        or use_e14_type_count_cells
        or use_pitcher_batter_interactions
        or use_e14_rate_hand_bin
        or use_e14_n_hand_bin
        or use_hierarchical_e14
        or use_recent_form
        or use_current_state_full
    )
    include_e14_base = "e14" in args.features
    use_e14_multi = "e14_multi" in args.features
    use_consistent_prior = "consistent_prior" in args.features
    use_platoon = "platoon" in args.features
    use_pitcher_te = "pitcher_te" in args.features
    use_trackman = "trackman" in args.features
    use_trackman_rich = "trackman_rich" in args.features
    use_trackman_stability = "trackman_stability" in args.features
    use_trackman_group_stability = "trackman_group_stability" in args.features
    use_trackman_game_repeatability = "trackman_game_repeatability" in args.features
    use_trackman_inning_physics = "trackman_inning_physics" in args.features
    use_trackman_trend = "trackman_trend" in args.features
    use_trackman_platoon = "trackman_platoon" in args.features
    use_trackman_count = "trackman_count" in args.features
    use_trackman_workload = "trackman_workload" in args.features
    use_trackman_teacher = "trackman_teacher" in args.features
    use_trackman_lupi = "trackman_lupi" in args.features
    use_trackman_archetype = "trackman_archetype" in args.features
    use_trackman_batter_rich = "trackman_batter_rich" in args.features
    if use_trackman_lupi and not use_trackman_rich:
        raise ValueError("trackman_lupi requires trackman_rich")
    if use_trackman_archetype and not use_trackman_rich:
        raise ValueError("trackman_archetype requires trackman_rich")
    use_e22_probs = "e22_probs" in args.features
    use_e22_cat = "e22_cat" in args.features
    use_partial_expanded_auto_pitch_latent = (
        "partial_expanded_auto_pitch_latent" in args.features
    )
    use_matchup_hand_auto_pitch_latent = (
        "matchup_hand_auto_pitch_latent" in args.features
    )
    use_expanded_auto_pitch_latent = (
        "expanded_auto_pitch_latent" in args.features
        or use_partial_expanded_auto_pitch_latent
    )
    use_auto_pitch_profile_latent = "auto_pitch_profile_latent" in args.features
    use_auto_pitch_latent = (
        "auto_pitch_latent" in args.features
        or use_auto_pitch_profile_latent
        or use_expanded_auto_pitch_latent
        or use_matchup_hand_auto_pitch_latent
    )
    use_fine_pitch_latent = (
        "fine_pitch_latent" in args.features or use_auto_pitch_latent
    )
    if "fine_pitch_latent" in args.features and use_auto_pitch_latent:
        raise ValueError(
            "choose exactly one of fine_pitch_latent/auto_pitch_latent variants"
        )
    if (
        "auto_pitch_latent" in args.features
        and use_auto_pitch_profile_latent
    ):
        raise ValueError(
            "choose exactly one of auto_pitch_latent/auto_pitch_profile_latent"
        )
    if use_expanded_auto_pitch_latent and (
        "auto_pitch_latent" in args.features or use_auto_pitch_profile_latent
    ):
        raise ValueError(
            "choose exactly one auto-pitch latent selector variant"
        )
    if use_matchup_hand_auto_pitch_latent and (
        use_expanded_auto_pitch_latent
        or "auto_pitch_latent" in args.features
        or use_auto_pitch_profile_latent
    ):
        raise ValueError(
            "matchup_hand_auto_pitch_latent is a separately locked selector"
        )
    if use_matchup_hand_auto_pitch_latent and set(args.models) != {
        "catboost_fine_pitch_moe"
    }:
        raise ValueError(
            "the locked matchup selector may only feed catboost_fine_pitch_moe; "
            "its training e92 columns are intentionally inert"
        )
    include_components_base = (
        "components" in args.features or "reverse_component" in args.features
    )
    use_components = include_components_base or use_current_state_full
    reverse_component_only = "reverse_component" in args.features
    use_centered_platoon = "platoon_centered" in args.features
    use_pitcher_hand_category = "pitcher_hand_cat" in args.features
    use_f_regime = "f_regime" in args.features
    use_hand_matchup = "hand_matchup" in args.features
    use_semantic_row = "semantic_row" in args.features
    use_count_state = "count_state" in args.features
    use_type_count = "type_count" in args.features
    use_type_month = "type_month" in args.features
    use_team_matchup = "team_matchup" in args.features
    use_venue = "venue" in args.features
    use_pitcher_profile = "pitcher_profile" in args.features
    use_batter_e14_count_cells = "batter_e14_count_cells" in args.features
    use_hierarchical_batter_e14 = "hierarchical_batter_e14" in args.features
    include_batter_e14_base = (
        "batter_e14" in args.features
        or use_batter_e14_count_cells
        or use_pitcher_batter_interactions
        or use_hierarchical_batter_e14
    )
    use_batter_e14 = include_batter_e14_base or use_current_state_full
    include_batter_middle_base = "batter_middle_e14" in args.features
    include_pitchmix_base = "pitchmix_e14" in args.features
    use_batter_middle_e14 = include_batter_middle_base or use_current_state_full
    use_pitchmix_e14 = include_pitchmix_base or use_current_state_full
    history_group_specs = [
        name for name in HISTORICAL_GROUP_RATE_SPECS if name in args.features
    ]
    use_temporal_stable_joint = "temporal_stable_joint" in args.features
    use_outcome_context = "outcome_context" in args.features
    use_pitcher_context_profile = "pitcher_context_profile" in args.features
    use_batter_context_profile = "batter_context_profile" in args.features
    use_reverse_hand_cells = "reverse_hand_cells" in args.features
    use_fastball_hand_cells = "fastball_hand_cells" in args.features

    train_e14 = valid_e14 = None
    train_e14_multi = valid_e14_multi = None
    train_hierarchical_e14 = valid_hierarchical_e14 = None
    train_recent_form = valid_recent_form = None
    train_recent_denominators = valid_recent_denominators = None
    train_recent_workload = valid_recent_workload = None
    recent_workload_meta: dict[str, Any] = {"enabled": False}
    hierarchical_e14_meta: dict[str, Any] = {"enabled": False}
    train_e14_hand_cells = valid_e14_hand_cells = None
    train_e14_hand_bins = valid_e14_hand_bins = None
    if use_e14:
        states_before, final_state = season_end_state(history_all)
        train_priors = (
            candidate_priors_before_each_season(history_all, args.prior_mode)
            if use_consistent_prior
            else prior_before_each_season(history_all)
        )
        train_e14, _ = build_e14_features(
            history, states_before, train_priors, prior, k=args.e14_k
        )
        valid_e14, _ = build_e14_features(
            valid, {season: final_state}, {season: prior}, prior, k=args.e14_k
        )
        if use_hierarchical_e14:
            train_hierarchical_e14, train_hierarchical_meta = (
                build_hierarchical_entity_features(
                    history, states_before, train_priors, prior,
                    PITCHER, "asof_pitcher_n", "asof_pitcher_success_rate",
                    "e60_pitcher",
                )
            )
            valid_hierarchical_e14, valid_hierarchical_meta = (
                build_hierarchical_entity_features(
                    valid, {season: final_state}, {season: prior}, prior,
                    PITCHER, "asof_pitcher_n", "asof_pitcher_success_rate",
                    "e60_pitcher",
                )
            )
            hierarchical_e14_meta = {
                "enabled": True,
                "train": train_hierarchical_meta,
                "valid": valid_hierarchical_meta,
            }
        if use_recent_form:
            train_recent_form = build_recent_form_features(
                history, train_e14, use_recent_form_count_cells
            )
            valid_recent_form = build_recent_form_features(
                valid, valid_e14, use_recent_form_count_cells
            )
        if use_e14_multi:
            train_e14_multi = build_e14_multi_features(
                history, train_e14, train_priors, prior
            )
            valid_e14_multi = build_e14_multi_features(
                valid, valid_e14, {season: prior}, prior
            )

        e14_interaction_train_parts: list[pd.DataFrame] = []
        e14_interaction_valid_parts: list[pd.DataFrame] = []
        if use_e14_hand_cells:
            e14_interaction_train_parts.append(
                build_e14_hand_cell_features(history, train_e14)
            )
            e14_interaction_valid_parts.append(
                build_e14_hand_cell_features(valid, valid_e14)
            )
        if use_e14_count_cells:
            e14_interaction_train_parts.append(
                build_e14_count_cell_features(history, train_e14, False)
            )
            e14_interaction_valid_parts.append(
                build_e14_count_cell_features(valid, valid_e14, False)
            )
        if use_e14_type_count_cells:
            e14_interaction_train_parts.append(
                build_e14_count_cell_features(history, train_e14, True)
            )
            e14_interaction_valid_parts.append(
                build_e14_count_cell_features(valid, valid_e14, True)
            )
        if e14_interaction_train_parts:
            train_e14_hand_cells = pd.concat(e14_interaction_train_parts, axis=1)
            valid_e14_hand_cells = pd.concat(e14_interaction_valid_parts, axis=1)
        if use_e14_rate_hand_bin or use_e14_n_hand_bin:
            train_e14_hand_bins = build_e14_hand_bin_features(
                history,
                train_e14,
                include_rate=use_e14_rate_hand_bin,
                include_n=use_e14_n_hand_bin,
            )
            valid_e14_hand_bins = build_e14_hand_bin_features(
                valid,
                valid_e14,
                include_rate=use_e14_rate_hand_bin,
                include_n=use_e14_n_hand_bin,
            )

    if use_recent_denominators:
        train_recent_denominators = build_recent_denominator_features(history)
        valid_recent_denominators = build_recent_denominator_features(valid)
    if use_recent_workload_decoder:
        from experiments.v5_recent_workload_decoder_features import (  # noqa: WPS433
            build_recent_workload_decoder_fold_features,
        )

        (
            train_recent_workload,
            valid_recent_workload,
            recent_workload_meta,
        ) = build_recent_workload_decoder_fold_features(
            history_all,
            history,
            valid,
            season,
            ROOT / "experiments/params/v5_recent_workload_decoder_preregister.json",
            build_recent_denominator_features,
        )

    train_platoon = valid_platoon = None
    platoon_meta: dict[str, Any] = {}
    if use_platoon:
        priors_by_season = prior_before_each_season(history_all)
        platoon_before, platoon_final = platoon_states_before_each_season(
            history_all, priors_by_season, args.k_pitcher, args.k_platoon
        )
        train_platoon = build_platoon_frame(history, platoon_before, platoon_final)
        valid_platoon = build_platoon_frame(valid, {season: platoon_final}, platoon_final)
        platoon_meta = {
            "k_pitcher": args.k_pitcher,
            "k_platoon": args.k_platoon,
            "state_cells": len(platoon_final["platoon_delta"]),
            "state_pitchers": len(platoon_final["pitcher_rate"]),
            "valid_unseen_rate": float(valid_platoon["e30_platoon_unseen"].mean()),
        }

    train_pitcher_te = build_pitcher_te_features(history) if use_pitcher_te else None
    valid_pitcher_te = build_pitcher_te_features(valid) if use_pitcher_te else None

    train_trackman = valid_trackman = None
    trackman_meta: dict[str, Any] = {"enabled": False}
    if (
        use_trackman
        or use_trackman_rich
        or use_trackman_stability
        or use_trackman_group_stability
        or use_trackman_game_repeatability
        or use_trackman_inning_physics
        or use_trackman_trend
        or use_trackman_platoon
        or use_trackman_count
        or use_trackman_workload
        or use_trackman_teacher
        or use_trackman_lupi
        or use_trackman_archetype
        or use_trackman_batter_rich
    ):
        if joined_trackman is None:
            raise ValueError("Trackman features requested but joined Trackman rows were not loaded")
        # Import lazily so ordinary V2/V3 runs do not pay the structural-join cost.
        from experiments.run_e20r_rolling import (  # noqa: WPS433
            build_profile_features,
            build_rich_profile_features,
            build_stability_profile_features,
            build_group_stability_profile_features,
            build_trend_profile_features,
            build_trackman_count_features,
            build_trackman_platoon_features,
            profile_states_before_each_season,
            rich_profile_states_before_each_season,
            stability_profile_states_before_each_season,
            group_stability_profile_states_before_each_season,
            trend_profile_states_before_each_season,
            trackman_count_states_before_each_season,
            trackman_platoon_states_before_each_season,
        )
        if use_trackman_game_repeatability:
            from experiments.v5_trackman_game_repeatability_features import (  # noqa: WPS433
                build_game_repeatability_features,
                game_repeatability_states_before_each_season,
            )
        if use_trackman_inning_physics:
            from experiments.v5_trackman_inning_physics_features import (  # noqa: WPS433
                build_inning_physics_features,
                inning_physics_states_before_each_season,
            )
        if use_trackman_workload:
            from experiments.v5_trackman_workload_features import (  # noqa: WPS433
                build_workload_profile_features,
                workload_profile_states_before_each_season,
            )
        if use_trackman_teacher:
            from experiments.v5_trackman_teacher_profiles import (  # noqa: WPS433
                build_teacher_profile_features,
                teacher_profile_states_before_each_season,
            )
        if use_trackman_batter_rich:
            if raw_trackman is None:
                raise RuntimeError(
                    "trackman_batter_rich requires the full official TrackMan history"
                )
            from experiments.v5_batter_trackman_profiles import (  # noqa: WPS433
                build_batter_trackman_fold_features,
            )
        tm_history = joined_trackman.loc[joined_trackman[SEASON] < season]
        profile_seasons = sorted(int(value) for value in tm_history[SEASON].unique())
        requested_profiles = []
        if use_trackman:
            requested_profiles.append(
                ("simple", profile_states_before_each_season, build_profile_features)
            )
        if use_trackman_rich:
            requested_profiles.append(
                (
                    "rich",
                    rich_profile_states_before_each_season,
                    build_rich_profile_features,
                )
            )
        if use_trackman_stability:
            requested_profiles.append(
                (
                    "stability",
                    stability_profile_states_before_each_season,
                    build_stability_profile_features,
                )
            )
        if use_trackman_group_stability:
            requested_profiles.append(
                (
                    "group_stability",
                    group_stability_profile_states_before_each_season,
                    build_group_stability_profile_features,
                )
            )
        if use_trackman_game_repeatability:
            requested_profiles.append(
                (
                    "game_repeatability",
                    game_repeatability_states_before_each_season,
                    build_game_repeatability_features,
                )
            )
        if use_trackman_inning_physics:
            requested_profiles.append(
                (
                    "inning_physics",
                    inning_physics_states_before_each_season,
                    build_inning_physics_features,
                )
            )
        if use_trackman_trend:
            requested_profiles.append(
                (
                    "trend",
                    lambda joined, seasons, window=None: (
                        trend_profile_states_before_each_season(
                            joined, seasons, window=args.trackman_trend_window
                        )
                    ),
                    build_trend_profile_features,
                )
            )
        if use_trackman_teacher:
            requested_profiles.append(
                (
                    "teacher",
                    teacher_profile_states_before_each_season,
                    build_teacher_profile_features,
                )
            )
        if use_trackman_workload:
            requested_profiles.append(
                (
                    "workload",
                    workload_profile_states_before_each_season,
                    build_workload_profile_features,
                )
            )
        train_profile_parts: list[pd.DataFrame] = []
        valid_profile_parts: list[pd.DataFrame] = []
        profile_details: dict[str, Any] = {}
        final_profile_sizes: dict[str, int] = {}
        rich_profiles_before: dict[int, pd.DataFrame] | None = None
        rich_profile_final: pd.DataFrame | None = None
        for profile_name, state_builder, feature_builder in requested_profiles:
            profiles_before, profile_final = state_builder(
                tm_history, profile_seasons, window=args.trackman_window
            )
            train_part, train_meta = feature_builder(history, profiles_before)
            valid_part, valid_meta = feature_builder(valid, {season: profile_final})
            train_profile_parts.append(train_part)
            valid_profile_parts.append(valid_part)
            profile_details[profile_name] = {"train": train_meta, "valid": valid_meta}
            final_profile_sizes[profile_name] = int(len(profile_final))
            if profile_name == "rich":
                rich_profiles_before = profiles_before
                rich_profile_final = profile_final
        if use_trackman_platoon:
            profiles_before, profile_final = trackman_platoon_states_before_each_season(
                tm_history,
                profile_seasons,
                k=args.trackman_platoon_k,
                window=args.trackman_window,
            )
            train_part, train_meta = build_trackman_platoon_features(
                history, profiles_before
            )
            valid_part, valid_meta = build_trackman_platoon_features(
                valid, {season: profile_final}
            )
            train_profile_parts.append(train_part)
            valid_profile_parts.append(valid_part)
            profile_details["platoon"] = {
                "train": train_meta,
                "valid": valid_meta,
                "k": args.trackman_platoon_k,
            }
            final_profile_sizes["platoon"] = int(len(profile_final))
        if use_trackman_count:
            profiles_before, profile_final = trackman_count_states_before_each_season(
                tm_history,
                profile_seasons,
                k=args.trackman_count_k,
                window=args.trackman_window,
            )
            train_part, train_meta = build_trackman_count_features(
                history, profiles_before
            )
            valid_part, valid_meta = build_trackman_count_features(
                valid, {season: profile_final}
            )
            train_profile_parts.append(train_part)
            valid_profile_parts.append(valid_part)
            profile_details["count"] = {
                "train": train_meta,
                "valid": valid_meta,
                "k": args.trackman_count_k,
            }
            final_profile_sizes["count"] = int(len(profile_final))
        if use_trackman_batter_rich:
            train_part, valid_part, batter_trackman_meta = (
                build_batter_trackman_fold_features(
                    history,
                    valid,
                    joined_trackman,
                    raw_trackman,
                    season,
                )
            )
            train_profile_parts.append(train_part)
            valid_profile_parts.append(valid_part)
            profile_details["batter_rich"] = batter_trackman_meta
            final_profile_sizes["batter_rich"] = int(
                batter_trackman_meta["states"][str(season)]["mapped_batters"]
            )
        train_trackman = pd.concat(train_profile_parts, axis=1)
        valid_trackman = pd.concat(valid_profile_parts, axis=1)
        trackman_meta = {
            "enabled": True,
            "rich": use_trackman_rich,
            "stability": use_trackman_stability,
            "group_stability": use_trackman_group_stability,
            "game_repeatability": use_trackman_game_repeatability,
            "inning_physics": use_trackman_inning_physics,
            "trend": use_trackman_trend,
            "trend_window": args.trackman_trend_window,
            "simple": use_trackman,
            "platoon": use_trackman_platoon,
            "count": use_trackman_count,
            "workload": use_trackman_workload,
            "teacher": use_trackman_teacher,
            "lupi": use_trackman_lupi,
            "archetype": use_trackman_archetype,
            "batter_rich": use_trackman_batter_rich,
            "source_rows": int(len(tm_history)),
            "window": args.trackman_window,
            "state_pitchers": final_profile_sizes,
            "profiles": profile_details,
        }
        if use_trackman_lupi:
            from experiments.v5_trackman_lupi_features import (  # noqa: WPS433
                add_profile_deltas,
                build_cross_season_lupi_features,
            )

            train_lupi, valid_lupi, lupi_meta = build_cross_season_lupi_features(
                history,
                valid,
                joined_trackman,
                smoke_source_rows=(5000 if args.max_history_rows else None),
            )
            train_lupi = add_profile_deltas(train_lupi, train_trackman)
            valid_lupi = add_profile_deltas(valid_lupi, valid_trackman)
            train_trackman = pd.concat([train_trackman, train_lupi], axis=1)
            valid_trackman = pd.concat([valid_trackman, valid_lupi], axis=1)
            trackman_meta["lupi_details"] = lupi_meta
            trackman_meta["lupi_feature_columns"] = list(train_lupi.columns)
            del train_lupi, valid_lupi
        if use_trackman_archetype:
            if rich_profiles_before is None or rich_profile_final is None:
                raise RuntimeError("trackman_archetype rich profiles were not retained")
            from experiments.v5_trackman_archetype_features import (  # noqa: WPS433
                build_archetype_features,
                fit_archetype_basis,
            )

            archetype_basis = fit_archetype_basis(joined_trackman)
            train_archetype, train_archetype_meta = build_archetype_features(
                history, rich_profiles_before, archetype_basis
            )
            valid_archetype, valid_archetype_meta = build_archetype_features(
                valid, {season: rich_profile_final}, archetype_basis
            )
            train_trackman = pd.concat([train_trackman, train_archetype], axis=1)
            valid_trackman = pd.concat([valid_trackman, valid_archetype], axis=1)
            trackman_meta["archetype_details"] = {
                "train": train_archetype_meta,
                "valid": valid_archetype_meta,
                "feature_columns": list(train_archetype.columns),
            }
            del train_archetype, valid_archetype, archetype_basis

    train_e22 = valid_e22 = None
    e22_meta: dict[str, Any] = {"enabled": False}
    if use_e22_probs or use_e22_cat:
        if "e22_pitch_type_group" not in history.columns:
            raise ValueError("E22 probabilities requested but group labels were not loaded")
        if use_e22_cat:
            train_e22, valid_e22, stage1_meta = build_e22_catboost_probabilities(
                history, valid
            )
        else:
            from experiments.run_e22r_probs_rolling import (  # noqa: WPS433
                group_features_for_fold,
            )

            train_e22, valid_e22, stage1_meta = group_features_for_fold(history, valid)
        e22_meta = {"enabled": True, **stage1_meta}

    train_fine_pitch = valid_fine_pitch = None
    fine_pitch_meta: dict[str, Any] = {"enabled": False}
    if use_fine_pitch_latent:
        if use_matchup_hand_auto_pitch_latent:
            if joined_trackman is None or raw_trackman is None:
                raise RuntimeError(
                    "matchup_hand_auto_pitch_latent requires exact and raw TrackMan"
                )
            from experiments.v5_matchup_hand_selector import (  # noqa: WPS433
                build_locked_matchup_hand_probabilities,
            )

            train_fine_pitch, valid_fine_pitch, fine_pitch_meta = (
                build_locked_matchup_hand_probabilities(
                    history,
                    valid,
                    joined_trackman,
                    raw_trackman,
                    list(BASE_FEATURES),
                    list(BOOSTER_CATEGORICAL),
                    RANDOM_SEED,
                    os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu",
                )
            )
            fine_pitch_meta = {
                "enabled": True,
                "label_column": "auto_fine_pitch_type",
                **fine_pitch_meta,
            }
        else:
            train_fine_pitch, valid_fine_pitch, fine_pitch_meta = (
                build_fine_pitch_latent_probabilities(
                    history,
                    valid,
                    "auto_fine_pitch_type"
                    if use_auto_pitch_latent
                    else "fine_pitch_type",
                    use_profile_features=use_auto_pitch_profile_latent,
                    fit_full_validation_model=use_expanded_auto_pitch_latent,
                )
            )
        if use_expanded_auto_pitch_latent:
            if joined_trackman is None or raw_trackman is None:
                raise RuntimeError(
                    "expanded_auto_pitch_latent requires joined and raw TrackMan"
                )
            baseline_meta = fine_pitch_meta
            (
                train_fine_pitch,
                valid_fine_pitch,
                expanded_meta,
            ) = build_expanded_auto_pitch_probabilities(
                history,
                valid,
                train_fine_pitch,
                valid_fine_pitch,
                joined_trackman,
                raw_trackman,
            )
            fine_pitch_meta = {
                "enabled": True,
                "label_column": "auto_fine_pitch_type",
                "baseline_crossfit": baseline_meta,
                **expanded_meta,
            }

    train_components = valid_components = None
    component_meta: dict[str, Any] = {"enabled": False}
    if use_components:
        (
            component_before,
            component_priors,
            component_final,
            component_final_priors,
        ) = component_states_before_each_season(history_all)
        train_components, train_component_meta = build_component_features(
            history,
            component_before,
            component_priors,
            component_final_priors,
            args.component_k,
        )
        valid_components, valid_component_meta = build_component_features(
            valid,
            {season: component_final},
            {season: component_final_priors},
            component_final_priors,
            args.component_k,
        )
        if reverse_component_only:
            reverse_columns = [
                "e31_reverse_rate_season",
                "e31_reverse_delta_career",
            ]
            train_components = train_components[reverse_columns]
            valid_components = valid_components[reverse_columns]
        component_meta = {
            "enabled": True,
            "reverse_only": reverse_component_only,
            "train": train_component_meta,
            "valid": valid_component_meta,
            "final_priors": component_final_priors,
            "state_pitchers": len(component_final),
        }

    train_centered = valid_centered = None
    centered_meta: dict[str, Any] = {"enabled": False}
    if use_centered_platoon:
        centered_before, centered_final = centered_platoon_states_before_each_season(
            history_all, args.centered_platoon_k, args.centered_platoon_window
        )
        train_centered = build_centered_platoon_frame(
            history, centered_before, centered_final
        )
        valid_centered = build_centered_platoon_frame(
            valid, {season: centered_final}, centered_final
        )
        centered_meta = {
            "enabled": True,
            "k": args.centered_platoon_k,
            "season_window": args.centered_platoon_window,
            "state_cells": len(centered_final["delta"]),
            "state_seasons": centered_final["seasons"],
            "valid_unseen_rate": float(
                valid_centered["e32_platoon_centered_unseen"].mean()
            ),
        }

    train_pitcher_hand_category = (
        build_pitcher_hand_category(history) if use_pitcher_hand_category else None
    )
    valid_pitcher_hand_category = (
        build_pitcher_hand_category(valid) if use_pitcher_hand_category else None
    )
    train_f_regime = build_f_regime_feature(history) if use_f_regime else None
    valid_f_regime = build_f_regime_feature(valid) if use_f_regime else None
    train_hand_matchup = build_hand_matchup_features(history) if use_hand_matchup else None
    valid_hand_matchup = build_hand_matchup_features(valid) if use_hand_matchup else None
    train_semantic_row = (
        build_semantic_row_features(history) if use_semantic_row else None
    )
    valid_semantic_row = (
        build_semantic_row_features(valid) if use_semantic_row else None
    )
    train_count_state = build_count_state_feature(history) if use_count_state else None
    valid_count_state = build_count_state_feature(valid) if use_count_state else None
    train_type_count = build_type_count_feature(history) if use_type_count else None
    valid_type_count = build_type_count_feature(valid) if use_type_count else None
    train_type_month = build_type_month_feature(history) if use_type_month else None
    valid_type_month = build_type_month_feature(valid) if use_type_month else None
    train_team_matchup = build_team_matchup_feature(history) if use_team_matchup else None
    valid_team_matchup = build_team_matchup_feature(valid) if use_team_matchup else None
    train_venue = build_venue_features(history) if use_venue else None
    valid_venue = build_venue_features(valid) if use_venue else None
    train_pitcher_profile = valid_pitcher_profile = None
    pitcher_profile_meta: dict[str, Any] = {"enabled": False}
    if use_pitcher_profile:
        profile_before, profile_final = pitcher_profile_states_before_each_season(
            history_all
        )
        train_pitcher_profile = build_pitcher_profile_frame(
            history, profile_before, profile_final, args.pitcher_profile_k
        )
        valid_pitcher_profile = build_pitcher_profile_frame(
            valid, {season: profile_final}, profile_final, args.pitcher_profile_k
        )
        pitcher_profile_meta = {
            "enabled": True,
            "k": args.pitcher_profile_k,
            "state_pitchers": int(len(profile_final["table"])),
            "valid_unseen_rate": float(
                valid_pitcher_profile["c47_profile_unseen"].mean()
            ),
            "target_free": True,
            "cutoff": "prior seasons for training rows; full outer history for validation",
        }
    train_batter_e14 = valid_batter_e14 = None
    train_hierarchical_batter = valid_hierarchical_batter = None
    hierarchical_batter_meta: dict[str, Any] = {"enabled": False}
    train_batter_interactions = valid_batter_interactions = None
    batter_e14_meta: dict[str, Any] = {"enabled": False}
    if use_batter_e14:
        batter_before, batter_final = entity_season_end_state(
            history_all,
            "batter_id",
            "asof_batter_n",
            "asof_batter_success_rate",
        )
        batter_priors = (
            candidate_priors_before_each_season(history_all, args.prior_mode)
            if use_consistent_prior
            else prior_before_each_season(history_all)
        )
        train_batter_e14, train_batter_meta = build_entity_season_features(
            history, batter_before, batter_priors, prior,
            "batter_id", "asof_batter_n", "asof_batter_success_rate",
            "e49_batter", args.batter_e14_k,
        )
        valid_batter_e14, valid_batter_meta = build_entity_season_features(
            valid, {season: batter_final}, {season: prior}, prior,
            "batter_id", "asof_batter_n", "asof_batter_success_rate",
            "e49_batter", args.batter_e14_k,
        )
        batter_e14_meta = {
            "enabled": True,
            "state_batters": int(len(batter_final)),
            "train": train_batter_meta,
            "valid": valid_batter_meta,
            "row_independent": True,
        }
        if use_hierarchical_batter_e14:
            train_hierarchical_batter, train_hierarchical_batter_meta = (
                build_hierarchical_entity_features(
                    history, batter_before, batter_priors, prior,
                    "batter_id", "asof_batter_n", "asof_batter_success_rate",
                    "e61_batter",
                )
            )
            valid_hierarchical_batter, valid_hierarchical_batter_meta = (
                build_hierarchical_entity_features(
                    valid, {season: batter_final}, {season: prior}, prior,
                    "batter_id", "asof_batter_n", "asof_batter_success_rate",
                    "e61_batter",
                )
            )
            hierarchical_batter_meta = {
                "enabled": True,
                "train": train_hierarchical_batter_meta,
                "valid": valid_hierarchical_batter_meta,
            }
        interaction_train_parts: list[pd.DataFrame] = []
        interaction_valid_parts: list[pd.DataFrame] = []
        if use_batter_e14_count_cells:
            interaction_train_parts.append(
                build_rate_count_cell_features(
                    history, train_batter_e14["e49_batter_rate_season"],
                    "e50_batter_rate",
                )
            )
            interaction_valid_parts.append(
                build_rate_count_cell_features(
                    valid, valid_batter_e14["e49_batter_rate_season"],
                    "e50_batter_rate",
                )
            )
        if use_pitcher_batter_interactions:
            if train_e14 is None or valid_e14 is None:
                raise ValueError("pitcher/batter interactions require base E14 features")
            interaction_train_parts.append(
                build_pitcher_batter_season_interactions(
                    train_e14, train_batter_e14
                )
            )
            interaction_valid_parts.append(
                build_pitcher_batter_season_interactions(
                    valid_e14, valid_batter_e14
                )
            )
        if interaction_train_parts:
            train_batter_interactions = pd.concat(interaction_train_parts, axis=1)
            valid_batter_interactions = pd.concat(interaction_valid_parts, axis=1)
    train_aux_components = valid_aux_components = None
    aux_component_meta: dict[str, Any] = {}
    aux_train_parts: list[pd.DataFrame] = []
    aux_valid_parts: list[pd.DataFrame] = []
    if use_batter_middle_e14:
        batter_middle_columns = {"middle": "asof_batter_middle_rate"}
        bm_before, bm_priors, bm_final, bm_final_priors = (
            generic_component_states_before_each_season(
                history_all, "batter_id", "asof_batter_n", batter_middle_columns
            )
        )
        train_bm, train_bm_meta = build_generic_component_features(
            history, bm_before, bm_priors, bm_final_priors,
            "batter_id", "asof_batter_n", batter_middle_columns,
            "e52_batter", args.batter_middle_k,
            include_raw=use_current_state_full,
        )
        valid_bm, valid_bm_meta = build_generic_component_features(
            valid, {season: bm_final}, {season: bm_final_priors}, bm_final_priors,
            "batter_id", "asof_batter_n", batter_middle_columns,
            "e52_batter", args.batter_middle_k,
            include_raw=use_current_state_full,
        )
        aux_train_parts.append(train_bm)
        aux_valid_parts.append(valid_bm)
        aux_component_meta["batter_middle"] = {
            "enabled": True,
            "state_batters": len(bm_final),
            "train": train_bm_meta,
            "valid": valid_bm_meta,
        }
    if use_pitchmix_e14:
        pitchmix_columns = {
            "fastball": "asof_pitcher_fastball_rate",
            "breaking": "asof_pitcher_breaking_rate",
            "offspeed": "asof_pitcher_offspeed_rate",
        }
        pm_before, pm_priors, pm_final, pm_final_priors = (
            generic_component_states_before_each_season(
                history_all, PITCHER, "asof_pitcher_pitchmix_n", pitchmix_columns
            )
        )
        train_pm, train_pm_meta = build_generic_component_features(
            history, pm_before, pm_priors, pm_final_priors,
            PITCHER, "asof_pitcher_pitchmix_n", pitchmix_columns,
            "e53_pitchmix", args.pitchmix_k,
            include_raw=use_current_state_full,
        )
        valid_pm, valid_pm_meta = build_generic_component_features(
            valid, {season: pm_final}, {season: pm_final_priors}, pm_final_priors,
            PITCHER, "asof_pitcher_pitchmix_n", pitchmix_columns,
            "e53_pitchmix", args.pitchmix_k,
            include_raw=use_current_state_full,
        )
        aux_train_parts.append(train_pm)
        aux_valid_parts.append(valid_pm)
        aux_component_meta["pitchmix"] = {
            "enabled": True,
            "state_pitchers": len(pm_final),
            "train": train_pm_meta,
            "valid": valid_pm_meta,
        }
    if aux_train_parts:
        train_aux_components = pd.concat(aux_train_parts, axis=1)
        valid_aux_components = pd.concat(aux_valid_parts, axis=1)
    train_current_state = valid_current_state = None
    current_state_meta: dict[str, Any] = {"enabled": False}
    if use_current_state_full:
        dependencies = {
            "train_e14": train_e14,
            "valid_e14": valid_e14,
            "train_components": train_components,
            "valid_components": valid_components,
            "train_batter_e14": train_batter_e14,
            "valid_batter_e14": valid_batter_e14,
            "train_aux_components": train_aux_components,
            "valid_aux_components": valid_aux_components,
        }
        missing_dependencies = [
            name for name, value in dependencies.items() if value is None
        ]
        if missing_dependencies:
            raise RuntimeError(
                "current_state_full dependencies were not constructed: "
                + ", ".join(missing_dependencies)
            )
        train_current_state = build_current_state_full_features(
            train_e14, train_components, train_batter_e14, train_aux_components
        )
        valid_current_state = build_current_state_full_features(
            valid_e14, valid_components, valid_batter_e14, valid_aux_components
        )
        if use_current_state_context or use_current_state_level:
            train_interactions = build_current_state_interaction_features(
                history,
                train_current_state,
                include_context=use_current_state_context,
                include_level=use_current_state_level,
            )
            valid_interactions = build_current_state_interaction_features(
                valid,
                valid_current_state,
                include_context=use_current_state_context,
                include_level=use_current_state_level,
            )
            train_current_state = pd.concat(
                [train_current_state, train_interactions], axis=1
            )
            valid_current_state = pd.concat(
                [valid_current_state, valid_interactions], axis=1
            )
        current_state_meta = {
            "enabled": True,
            "context_interactions": use_current_state_context,
            "level_interactions": use_current_state_level,
            "feature_columns": list(train_current_state.columns),
            "count": int(train_current_state.shape[1]),
            "cutoff": "prior-season player constants plus current row official as-of counters",
            "row_independent": True,
        }
    train_history_groups = valid_history_groups = None
    history_group_meta: dict[str, Any] = {"enabled": []}
    if history_group_specs:
        train_history_groups, train_history_meta = build_historical_group_rate_features(
            history,
            history_all,
            history_group_specs,
            args.history_group_k,
            args.history_group_window,
            prior,
        )
        valid_history_groups, valid_history_meta = build_historical_group_rate_features(
            valid,
            history_all,
            history_group_specs,
            args.history_group_k,
            args.history_group_window,
            prior,
        )
        history_group_meta = {
            "enabled": history_group_specs,
            "train": train_history_meta,
            "valid": valid_history_meta,
            "row_independent": True,
        }
    train_temporal_stable = valid_temporal_stable = None
    temporal_stable_meta: dict[str, Any] = {"enabled": False}
    if use_temporal_stable_joint:
        train_temporal_stable, train_temporal_stable_meta = (
            build_temporal_stable_joint_features(history, history_all)
        )
        valid_temporal_stable, valid_temporal_stable_meta = (
            build_temporal_stable_joint_features(valid, history_all)
        )
        temporal_stable_meta = {
            "enabled": True,
            "train": train_temporal_stable_meta,
            "valid": valid_temporal_stable_meta,
            "preregistered_strengths": {
                "pitcher": 100.0,
                "hand": 38.0,
                "pressure_hand": 30.0,
            },
        }
    train_outcome_context = valid_outcome_context = None
    outcome_context_meta: dict[str, Any] = {"enabled": False}
    if use_outcome_context:
        if component15_labels_full is None:
            raise RuntimeError("outcome_context requires reconstructed component15 labels")
        train_outcome_context, train_outcome_context_meta = (
            build_outcome_context_features(
                history, history_all, component15_labels_full,
                args.outcome_context_k,
            )
        )
        valid_outcome_context, valid_outcome_context_meta = (
            build_outcome_context_features(
                valid, history_all, component15_labels_full,
                args.outcome_context_k,
            )
        )
        outcome_context_meta = {
            "enabled": True,
            "train": train_outcome_context_meta,
            "valid": valid_outcome_context_meta,
            "row_independent": True,
        }
    train_entity_profiles = valid_entity_profiles = None
    entity_profile_meta: dict[str, Any] = {"enabled": []}
    profile_train_parts: list[pd.DataFrame] = []
    profile_valid_parts: list[pd.DataFrame] = []
    if use_pitcher_context_profile:
        train_part, train_meta = build_completed_entity_context_profile(
            history, history_all, PITCHER, "e68_pitcher", args.history_group_k, prior
        )
        valid_part, valid_meta = build_completed_entity_context_profile(
            valid, history_all, PITCHER, "e68_pitcher", args.history_group_k, prior
        )
        profile_train_parts.append(train_part)
        profile_valid_parts.append(valid_part)
        entity_profile_meta["pitcher"] = {"train": train_meta, "valid": valid_meta}
        entity_profile_meta["enabled"].append("pitcher")
    if use_batter_context_profile:
        train_part, train_meta = build_completed_entity_context_profile(
            history, history_all, "batter_id", "e69_batter", args.history_group_k, prior
        )
        valid_part, valid_meta = build_completed_entity_context_profile(
            valid, history_all, "batter_id", "e69_batter", args.history_group_k, prior
        )
        profile_train_parts.append(train_part)
        profile_valid_parts.append(valid_part)
        entity_profile_meta["batter"] = {"train": train_meta, "valid": valid_meta}
        entity_profile_meta["enabled"].append("batter")
    if profile_train_parts:
        train_entity_profiles = pd.concat(profile_train_parts, axis=1)
        valid_entity_profiles = pd.concat(profile_valid_parts, axis=1)
    train_rate_hand_parts: list[pd.DataFrame] = []
    valid_rate_hand_parts: list[pd.DataFrame] = []
    if use_reverse_hand_cells:
        train_rate_hand_parts.append(
            build_rate_hand_cell_features(
                history, "asof_pitcher_reverse_rate", "c40_reverse"
            )
        )
        valid_rate_hand_parts.append(
            build_rate_hand_cell_features(
                valid, "asof_pitcher_reverse_rate", "c40_reverse"
            )
        )
    if use_fastball_hand_cells:
        train_rate_hand_parts.append(
            build_rate_hand_cell_features(
                history, "asof_pitcher_fastball_rate", "c41_fastball"
            )
        )
        valid_rate_hand_parts.append(
            build_rate_hand_cell_features(
                valid, "asof_pitcher_fastball_rate", "c41_fastball"
            )
        )
    train_rate_hand_cells = (
        pd.concat(train_rate_hand_parts, axis=1) if train_rate_hand_parts else None
    )
    valid_rate_hand_cells = (
        pd.concat(valid_rate_hand_parts, axis=1) if valid_rate_hand_parts else None
    )

    train_x = assemble(
        history,
        train_e14 if include_e14_base else None,
        train_e14_multi,
        train_platoon,
        train_pitcher_te,
        train_trackman,
        train_e22,
        train_components if include_components_base else None,
        train_centered,
        train_pitcher_hand_category,
        train_f_regime,
        train_hand_matchup,
        train_count_state,
        train_type_count,
        train_type_month,
        train_e14_hand_cells,
        train_rate_hand_cells,
        train_e14_hand_bins,
        train_team_matchup,
    )
    valid_x = assemble(
        valid,
        valid_e14 if include_e14_base else None,
        valid_e14_multi,
        valid_platoon,
        valid_pitcher_te,
        valid_trackman,
        valid_e22,
        valid_components if include_components_base else None,
        valid_centered,
        valid_pitcher_hand_category,
        valid_f_regime,
        valid_hand_matchup,
        valid_count_state,
        valid_type_count,
        valid_type_month,
        valid_e14_hand_cells,
        valid_rate_hand_cells,
        valid_e14_hand_bins,
        valid_team_matchup,
    )
    extra_train = [
        part for part in (
            train_venue, train_pitcher_profile,
            train_fine_pitch,
            train_semantic_row,
            train_batter_e14 if include_batter_e14_base else None,
            train_batter_interactions,
            train_aux_components
            if (include_batter_middle_base or include_pitchmix_base) else None,
            train_current_state, train_history_groups, train_temporal_stable,
            train_outcome_context,
            train_hierarchical_e14, train_hierarchical_batter, train_recent_form,
            train_recent_denominators,
            train_recent_workload,
            train_entity_profiles,
        )
        if part is not None
    ]
    extra_valid = [
        part for part in (
            valid_venue, valid_pitcher_profile,
            valid_fine_pitch,
            valid_semantic_row,
            valid_batter_e14 if include_batter_e14_base else None,
            valid_batter_interactions,
            valid_aux_components
            if (include_batter_middle_base or include_pitchmix_base) else None,
            valid_current_state, valid_history_groups, valid_temporal_stable,
            valid_outcome_context,
            valid_hierarchical_e14, valid_hierarchical_batter, valid_recent_form,
            valid_recent_denominators,
            valid_recent_workload,
            valid_entity_profiles,
        )
        if part is not None
    ]
    if extra_train:
        train_x = pd.concat([train_x, *extra_train], axis=1)
        valid_x = pd.concat([valid_x, *extra_valid], axis=1)
    train_x = apply_feature_view(train_x, args.feature_view)
    valid_x = apply_feature_view(valid_x, args.feature_view)
    if args.drop_features:
        missing_drop = sorted(set(args.drop_features) - set(train_x.columns))
        if missing_drop:
            raise ValueError(f"--drop-features columns not assembled: {missing_drop}")
        train_x = train_x.drop(columns=args.drop_features)
        valid_x = valid_x.drop(columns=args.drop_features)
    validation_feature_dir = os.environ.get("V2_EXPORT_VALID_FEATURE_DIR")
    if validation_feature_dir:
        feature_output = Path(validation_feature_dir)
        feature_output.mkdir(parents=True, exist_ok=True)
        feature_path = feature_output / f"{args.stage}_{season}.pkl"
        valid_x.to_pickle(feature_path)
        print(f"[{season}/{args.stage}] exported validation features {feature_path}", flush=True)
    train_y = history[TARGET].to_numpy(dtype=np.int8, copy=False)
    valid_y = valid[TARGET].to_numpy(dtype=np.int8, copy=False)
    train_weight: np.ndarray | None = None
    if args.season_decay != 1.0 or (
        args.f_regime_start is not None and args.f_pre_regime_weight != 1.0
    ):
        ages = (season - 1 - history[SEASON].to_numpy(dtype=np.int16)).astype(np.float64)
        train_weight = np.power(args.season_decay, ages)
        if args.f_regime_start is not None and season > args.f_regime_start:
            pre_f = (
                history["game_type"].eq("F").to_numpy(dtype=bool, na_value=False)
                & history[SEASON].lt(args.f_regime_start).to_numpy(
                    dtype=bool, na_value=False
                )
            )
            train_weight[pre_f] *= args.f_pre_regime_weight

    teacher_target: np.ndarray | None = None
    teacher_mask: np.ndarray | None = None
    teacher_anchor_valid: np.ndarray | None = None
    teacher_details: dict[str, Any] = {}
    if "catboost_teacher" in args.models:
        if not args.teacher_stage or not args.teacher_years:
            raise ValueError(
                "catboost_teacher requires --teacher-stage and --teacher-years"
            )
        if not 0.0 <= args.teacher_alpha <= 1.0:
            raise ValueError("--teacher-alpha must be in [0, 1]")
        invalid_years = [year for year in args.teacher_years if year >= season]
        if invalid_years:
            raise ValueError(
                f"Teacher years must precede validation season {season}: {invalid_years}"
            )
        teacher_series = pd.Series(np.nan, index=history.index, dtype=np.float64)
        anchor_series = pd.Series(np.nan, index=history.index, dtype=np.float64)
        loaded_teacher_rows: dict[int, int] = {}
        prediction_dir = args.save_predictions or (
            ROOT / "experiments/results/predictions"
        )
        for teacher_year in args.teacher_years:
            teacher_path = prediction_dir / f"{args.teacher_stage}_{teacher_year}.npz"
            if not teacher_path.is_file():
                raise FileNotFoundError(f"Teacher artifact not found: {teacher_path}")
            with np.load(teacher_path) as artifact:
                if args.teacher_key not in artifact:
                    raise KeyError(
                        f"{teacher_path.name} has no {args.teacher_key!r}; "
                        f"available: {sorted(artifact.files)}"
                    )
                row_index = artifact["row_index"].astype(np.int64)
                values = artifact[args.teacher_key].astype(np.float64)
                teacher_y = artifact["y"].astype(np.int8)
            if len(row_index) != len(values) or not np.isfinite(values).all():
                raise ValueError(f"Invalid teacher values: {teacher_path}")
            if not np.array_equal(
                frame.loc[row_index, TARGET].to_numpy(dtype=np.int8), teacher_y
            ):
                raise ValueError(f"Teacher target alignment mismatch: {teacher_path}")
            available = np.isin(row_index, history.index.to_numpy(dtype=np.int64))
            selected_index = row_index[available]
            teacher_series.loc[selected_index] = values[available]
            if args.teacher_anchor_stage:
                anchor_path = (
                    prediction_dir
                    / f"{args.teacher_anchor_stage}_{teacher_year}.npz"
                )
                if not anchor_path.is_file():
                    raise FileNotFoundError(f"Teacher anchor not found: {anchor_path}")
                with np.load(anchor_path) as anchor_artifact:
                    if args.teacher_anchor_key not in anchor_artifact:
                        raise KeyError(
                            f"{anchor_path.name} has no {args.teacher_anchor_key!r}"
                        )
                    anchor_index = anchor_artifact["row_index"].astype(np.int64)
                    anchor_value = anchor_artifact[args.teacher_anchor_key].astype(
                        np.float64
                    )
                if not np.array_equal(row_index, anchor_index):
                    raise ValueError(f"Teacher anchor alignment mismatch: {anchor_path}")
                anchor_series.loc[selected_index] = anchor_value[available]
            loaded_teacher_rows[int(teacher_year)] = int(available.sum())
        teacher_mask = teacher_series.notna().to_numpy(dtype=bool)
        if not teacher_mask.any():
            raise ValueError("No teacher rows overlap the current training history")
        teacher_soft = teacher_series.to_numpy(dtype=np.float64)[teacher_mask]
        actual_soft = train_y.astype(np.float64)[teacher_mask]
        mixed_teacher = (
            (1.0 - args.teacher_alpha) * teacher_soft
            + args.teacher_alpha * actual_soft
        )
        if args.teacher_anchor_stage:
            if anchor_series.loc[teacher_series.notna()].isna().any():
                raise ValueError("Teacher anchor is missing one or more teacher rows")
            anchor_train = anchor_series.to_numpy(dtype=np.float64)[teacher_mask]
            teacher_residual = mixed_teacher - anchor_train
            center_values: dict[str, float] = {}
            if args.teacher_center != "none":
                center_frame = history.loc[teacher_mask, [SEASON, "game_type"]].copy()
                center_frame["_teacher_residual"] = teacher_residual
                group_columns = [SEASON]
                if args.teacher_center == "year_game_type":
                    group_columns.append("game_type")
                centers = center_frame.groupby(
                    group_columns, sort=True, observed=True
                )["_teacher_residual"].transform("mean").to_numpy(dtype=np.float64)
                teacher_residual = teacher_residual - centers
                grouped_centers = center_frame.groupby(
                    group_columns, sort=True, observed=True
                )["_teacher_residual"].mean()
                center_values = {
                    "|".join(str(part) for part in (key if isinstance(key, tuple) else (key,))): float(value)
                    for key, value in grouped_centers.items()
                }
            teacher_target = 0.5 + teacher_residual
            if not args.teacher_residual_output:
                anchor_valid_path = (
                    prediction_dir / f"{args.teacher_anchor_stage}_{season}.npz"
                )
                if not anchor_valid_path.is_file():
                    raise FileNotFoundError(
                        f"Validation teacher anchor not found: {anchor_valid_path}"
                    )
                with np.load(anchor_valid_path) as anchor_artifact:
                    if args.teacher_anchor_key not in anchor_artifact:
                        raise KeyError(
                            f"{anchor_valid_path.name} has no "
                            f"{args.teacher_anchor_key!r}"
                        )
                    if not np.array_equal(
                        anchor_artifact["row_index"].astype(np.int64),
                        valid.index.to_numpy(dtype=np.int64),
                    ):
                        raise ValueError(
                            f"Validation teacher anchor mismatch: {anchor_valid_path}"
                        )
                    teacher_anchor_valid = anchor_artifact[
                        args.teacher_anchor_key
                    ].astype(np.float64)
        else:
            teacher_target = mixed_teacher
        teacher_details = {
            "teacher_stage": args.teacher_stage,
            "teacher_key": args.teacher_key,
            "teacher_years": [int(value) for value in args.teacher_years],
            "loaded_rows": loaded_teacher_rows,
            "fit_rows": int(teacher_mask.sum()),
            "teacher_alpha": float(args.teacher_alpha),
            "anchor_stage": args.teacher_anchor_stage,
            "anchor_key": args.teacher_anchor_key if args.teacher_anchor_stage else None,
            "center": args.teacher_center,
            "residual_output": bool(args.teacher_residual_output),
            "historical_residual_centers": center_values if args.teacher_anchor_stage else {},
            "teacher_target_mean": float(teacher_soft.mean()),
            "teacher_target_std": float(teacher_soft.std()),
            "outer_oof_teacher_only": True,
        }
        if args.teacher_fill_hard_labels:
            if args.teacher_anchor_stage:
                raise ValueError(
                    "--teacher-fill-hard-labels is incompatible with teacher anchors"
                )
            teacher_rows = int(teacher_mask.sum())
            full_teacher_target = train_y.astype(np.float64, copy=True)
            full_teacher_target[teacher_mask] = teacher_target
            teacher_target = full_teacher_target
            teacher_mask = np.ones(len(history), dtype=bool)
            teacher_details.update(
                {
                    "outer_oof_teacher_only": False,
                    "teacher_rows": teacher_rows,
                    "hard_label_fallback_rows": int(len(history) - teacher_rows),
                    "fit_rows": int(len(history)),
                    "full_history_hybrid_target": True,
                }
            )

    predictions: dict[str, np.ndarray] = {}
    auxiliary_predictions: dict[str, np.ndarray] = {}
    fit_details: dict[str, Any] = {}
    for name in args.models:
        if name == "tabicl":
            if train_weight is not None:
                raise ValueError(
                    "tabicl preregistration does not allow season/sample weights"
                )
            prediction, details = fit_tabicl_model(
                f"{season}/{args.stage}/{name}",
                train_x,
                train_y,
                valid_x,
                history[SEASON],
                params,
            )
        elif name == "catboost_outcome":
            prediction, details = fit_outcome_model(
                f"{season}/{args.stage}/{name}", train_x, history, valid_x, params,
                args.outcome_scheme, train_weight, outcome_labels_full,
                args.save_outcome_components,
            )
        elif name == "catboost_game_centered_brier":
            prediction, details = fit_game_centered_brier_model(
                f"{season}/{args.stage}/{name}",
                train_x,
                history,
                valid_x,
                params,
                train_weight,
            )
        elif name == "catboost_game_pairwise_rank":
            prediction, details = fit_game_pairwise_rank_model(
                f"{season}/{args.stage}/{name}",
                train_x,
                history,
                valid_x,
                params,
                train_weight,
            )
        elif name == "lgbm_outcome":
            prediction, details = fit_lgbm_outcome_model(
                f"{season}/{args.stage}/{name}", train_x, history, valid_x, params,
                args.outcome_scheme, train_weight, outcome_labels_full,
                args.save_outcome_components,
            )
        elif name == "catboost_count_moe":
            if args.save_outcome_components:
                raise ValueError(
                    "catboost_count_moe does not support --save-outcome-components"
                )
            prediction, details = fit_count_moe_outcome_model(
                f"{season}/{args.stage}/{name}",
                train_x,
                history,
                valid_x,
                valid,
                params,
                args.outcome_scheme,
                train_weight,
                outcome_labels_full,
            )
        elif name == "catboost_pitchtype_moe":
            prediction, details = fit_pitchtype_moe_outcome_model(
                f"{season}/{args.stage}/{name}",
                train_x,
                history,
                valid_x,
                valid,
                params,
                args.outcome_scheme,
                train_weight,
                outcome_labels_full,
                args.save_outcome_components,
            )
        elif name == "catboost_dense_pitchtype_moe":
            prediction, details = fit_dense_pitchtype_moe_outcome_model(
                f"{season}/{args.stage}/{name}",
                train_x,
                history,
                valid_x,
                valid,
                params,
                args.outcome_scheme,
                train_weight,
                outcome_labels_full,
                args.save_outcome_components,
            )
        elif name == "catboost_dense_pitch_joint":
            prediction, details = fit_dense_pitch_joint_outcome_model(
                f"{season}/{args.stage}/{name}",
                train_x,
                history,
                valid_x,
                params,
                args.outcome_scheme,
                train_weight,
                outcome_labels_full,
                args.save_outcome_components,
            )
        elif name == "catboost_fine_pitch_joint":
            prediction, details = fit_fine_pitch_joint_outcome_model(
                f"{season}/{args.stage}/{name}",
                train_x,
                history,
                valid_x,
                params,
                args.outcome_scheme,
                train_weight,
                outcome_labels_full,
                args.save_outcome_components,
            )
        elif name == "catboost_physics_joint":
            prediction, details = fit_physics_joint_outcome_model(
                f"{season}/{args.stage}/{name}",
                train_x,
                history,
                valid_x,
                params,
                args.outcome_scheme,
                train_weight,
                outcome_labels_full,
                args.save_outcome_components,
            )
        elif name == "catboost_hier_pitch_joint":
            prediction, details = fit_hierarchical_pitch_joint_outcome_model(
                f"{season}/{args.stage}/{name}",
                train_x,
                history,
                valid_x,
                params,
                args.outcome_scheme,
                train_weight,
                outcome_labels_full,
                args.save_outcome_components,
            )
        elif name == "catboost_auto_pitch_joint":
            prediction, details = fit_auto_pitch_joint_outcome_model(
                f"{season}/{args.stage}/{name}",
                train_x,
                history,
                valid_x,
                params,
                args.outcome_scheme,
                train_weight,
                outcome_labels_full,
                args.save_outcome_components,
            )
        elif name == "catboost_fine_pitch_moe":
            prediction, details = fit_fine_pitch_moe_outcome_model(
                f"{season}/{args.stage}/{name}",
                train_x,
                history,
                valid_x,
                params,
                args.outcome_scheme,
                train_weight,
                outcome_labels_full,
                args.save_outcome_components,
            )
        elif name == "catboost_fine_pitch_binary_moe":
            prediction, details = fit_fine_pitch_binary_moe_model(
                f"{season}/{args.stage}/{name}",
                train_x,
                history,
                valid_x,
                params,
                train_weight,
                args.save_outcome_components,
            )
        elif name == "realmlp_outcome":
            prediction, details = fit_realmlp_model(
                f"{season}/{args.stage}/{name}", train_x, train_y, valid_x,
                params, outcome_labels_full.reindex(history.index), train_weight,
                args.save_outcome_components,
            )
        elif name == "realmlp":
            prediction, details = fit_realmlp_model(
                f"{season}/{args.stage}/{name}", train_x, train_y, valid_x,
                params, None, train_weight, args.save_outcome_components,
            )
        elif name in {"tabr_outcome", "realtabr_outcome"}:
            prediction, details = fit_realmlp_model(
                f"{season}/{args.stage}/{name}", train_x, train_y, valid_x,
                params, outcome_labels_full.reindex(history.index), train_weight,
                args.save_outcome_components, name.removesuffix("_outcome"),
            )
        elif name in {"tabr", "realtabr"}:
            prediction, details = fit_realmlp_model(
                f"{season}/{args.stage}/{name}", train_x, train_y, valid_x,
                params, None, train_weight, args.save_outcome_components, name,
            )
        elif name in {
            "deep_mlp_outcome", "deepfm_outcome", "tabtransformer_outcome",
            "tabm_outcome", "tabm_periodic_outcome",
            "tabm_piecewise_outcome",
        }:
            prediction, details = fit_deep_outcome_model(
                f"{season}/{args.stage}/{name}",
                name.removesuffix("_outcome"),
                train_x, history, valid_x, params,
                args.outcome_scheme, train_weight, outcome_labels_full,
                args.save_outcome_components,
                valid_y,
            )
        elif name == "catboost_leaf_refit":
            prediction, details = fit_leaf_refit_model(
                f"{season}/{args.stage}/{name}",
                train_x, history, valid_x, valid, params,
                args.outcome_scheme, train_weight, outcome_labels_full,
            )
        elif name == "catboost_brier":
            prediction, details = fit_brier_model(
                f"{season}/{args.stage}/{name}", train_x, train_y, valid_x,
                params, train_weight,
            )
        elif name == "catboost_group_soft":
            prediction, details = fit_group_soft_brier_model(
                f"{season}/{args.stage}/{name}",
                train_x,
                history,
                train_y,
                valid_x,
                params,
                train_weight,
            )
        elif name == "catboost_state_residual":
            prediction, details = fit_state_residual_model(
                f"{season}/{args.stage}/{name}",
                train_x,
                history,
                train_y,
                valid_x,
                params,
                train_weight,
            )
        elif name == "catboost_multi_brier":
            prediction, details = fit_multi_brier_model(
                f"{season}/{args.stage}/{name}", train_x, history, valid_x,
                params, args.outcome_scheme, train_weight, outcome_labels_full,
            )
        elif name == "catboost_dense_multitask":
            prediction, details = fit_dense_multitask_brier_model(
                f"{season}/{args.stage}/{name}",
                train_x,
                history,
                valid_x,
                component15_labels_full.reindex(history.index),
                params,
                train_weight,
                args.save_outcome_components,
            )
        elif name == "tabm_dense_multitask":
            prediction, details = fit_neural_dense_multitask_model(
                f"{season}/{args.stage}/{name}",
                train_x,
                history,
                valid_x,
                component15_labels_full.reindex(history.index),
                params,
                train_weight,
                args.save_outcome_components,
            )
        elif name == "tabm_pitch_gated":
            prediction, details = fit_pitch_gated_control_model(
                f"{season}/{args.stage}/{name}",
                train_x,
                history,
                valid_x,
                params,
                train_weight,
            )
        elif name == "catboost_component_pattern_moe":
            prediction, details = fit_component_pattern_moe_model(
                f"{season}/{args.stage}/{name}",
                train_x,
                history,
                valid_x,
                component15_labels_full.reindex(history.index),
                params,
                train_weight,
                args.save_outcome_components,
            )
        elif name == "catboost_failure_decomp":
            prediction, details = fit_failure_decomposition_model(
                f"{season}/{args.stage}/{name}", train_x, valid_x,
                component15_labels_full.reindex(history.index), params, train_weight,
            )
        elif name == "catboost_failure_chain":
            prediction, details = fit_failure_chain_model(
                f"{season}/{args.stage}/{name}", train_x, valid_x,
                outcome_labels_full.reindex(history.index), params, train_weight,
            )
        elif name == "catboost_teacher":
            if teacher_target is None or teacher_mask is None:
                raise RuntimeError("Teacher target was not initialized")
            prediction, details = fit_brier_model(
                f"{season}/{args.stage}/{name}",
                train_x.loc[teacher_mask],
                teacher_target,
                valid_x,
                params,
                train_weight[teacher_mask] if train_weight is not None else None,
            )
            if teacher_anchor_valid is not None:
                prediction = np.clip(
                    teacher_anchor_valid + prediction - 0.5,
                    1e-6,
                    1.0 - 1e-6,
                )
                details["prediction_mode"] = "anchor_plus_learned_residual"
            elif args.teacher_anchor_stage and args.teacher_residual_output:
                details["prediction_mode"] = "centered_learned_residual_around_0.5"
            details["teacher"] = teacher_details
        else:
            prediction, details = fit_model(
                f"{season}/{args.stage}/{name}", model_factory(name, params),
                train_x, train_y, valid_x, history[SEASON],
                history["game_type"], args.inner_validation, train_weight,
            )
        predictions[name] = prediction
        for component_name, component_prediction in details.pop(
            "_component_predictions", {}
        ).items():
            auxiliary_predictions[f"{name}__{component_name}"] = component_prediction
        fit_details[name] = details

    scored: dict[str, np.ndarray] = dict(predictions)
    if args.blend:
        if len(args.blend) != len(args.models):
            raise ValueError("--blend must have one weight per --models entry")
        total = float(sum(args.blend))
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"--blend must sum to 1.0; got {total}")
        scored["blend"] = sum(
            weight * predictions[name] for weight, name in zip(args.blend, args.models)
        )

    if args.save_predictions is not None:
        args.save_predictions.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_predictions / f"{args.stage}_{season}.npz",
            y=valid_y,
            row_index=valid.index.to_numpy(),
            # Store a fixed-width Unicode array.  Pandas otherwise returns an
            # object array, which requires pickle and cannot be opened by the
            # ensemble stage's default-safe np.load(..., allow_pickle=False).
            cluster=np.asarray(valid[PITCHER].astype(str).to_numpy(), dtype=np.str_),
            **{name: value for name, value in scored.items()},
            **auxiliary_predictions,
        )

    baseline_source = "internal"
    if args.baseline_stage:
        path = args.save_predictions / f"{args.baseline_stage}_{season}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Baseline predictions not found: {path}")
        stored = np.load(path)
        if not np.array_equal(stored["row_index"], valid.index.to_numpy()):
            raise ValueError(
                f"Baseline fold {season} covers different rows than this run; "
                "re-run the baseline stage with the same subsampling options."
            )
        if args.baseline_key not in stored:
            raise KeyError(
                f"{path.name} has no '{args.baseline_key}'; available: {sorted(stored.files)}"
            )
        baseline = np.asarray(stored[args.baseline_key], dtype=np.float64)
        baseline_names = [f"{args.baseline_stage}:{args.baseline_key}"]
        baseline_source = "stage"
    else:
        baseline_names = args.baseline_models or [args.models[0]]
        baseline = np.mean([predictions[name] for name in baseline_names], axis=0)

    summaries: dict[str, Any] = {}
    for name, prediction in scored.items():
        summary = metric(valid_y, prediction)
        interval = paired_bootstrap_brier_ci(
            valid_y, baseline, prediction, iterations=args.bootstrap, seed=RANDOM_SEED,
            clusters=valid[PITCHER].astype(str).to_numpy(),
        )
        summaries[name] = {"summary": summary, "vs_baseline": interval}
        print(
            f"[{season}/{args.stage}/{name}] Brier={summary['brier']:.8f} "
            f"score={summary['competition_score']:,.1f} "
            f"delta={interval['point']:+.3e} "
            f"CI=[{interval['ci_low']:+.3e}, {interval['ci_high']:+.3e}]"
            f"{' SIG' if interval['significant'] else ''}",
            flush=True,
        )

    result = {
        "validation_season": season,
        "history_rows": int(len(history)),
        "state_history_rows": int(len(history_all)),
        "fit_history_seasons": sorted(int(value) for value in history[SEASON].unique()),
        "fit_game_types": args.fit_game_types or ["R", "F"],
        "fit_count_states": args.fit_count_states,
        "valid_rows": int(len(valid)),
        "feature_columns": list(train_x.columns),
        "dropped_features": args.drop_features or [],
        "prior": prior,
        "e14_multi": {
            "enabled": use_e14_multi,
            "k_values": list(E14_MULTI_KS) if use_e14_multi else [],
        },
        "e14_k": args.e14_k,
        "e14_training_prior": {
            "mode": args.prior_mode if use_consistent_prior else "legacy_all_history",
            "consistent_with_validation": use_consistent_prior,
        },
        "platoon": platoon_meta,
        "trackman": trackman_meta,
        "partial_trackman_linkage": partial_trackman_meta,
        "expanded_trackman_profiles": expanded_trackman_profile_meta,
        "e22_probs": e22_meta,
        "fine_pitch_latent": fine_pitch_meta,
        "components": component_meta,
        "platoon_centered": centered_meta,
        "pitcher_hand_category": {"enabled": use_pitcher_hand_category},
        "f_regime": {"enabled": use_f_regime, "break_season": 2023},
        "hand_matchup": {"enabled": use_hand_matchup},
        "semantic_row": {
            "enabled": use_semantic_row,
            "added_feature_count": 10 if use_semantic_row else 0,
            "reused_existing_feature": "c36_same_hand" if use_semantic_row else None,
            "current_row_only": True,
            "external_data_used": False,
        },
        "count_state": {"enabled": use_count_state},
        "type_count": {"enabled": use_type_count},
        "type_month": {"enabled": use_type_month},
        "e14_hand_cells": {"enabled": use_e14_hand_cells},
        "e14_count_cells": {
            "enabled": use_e14_count_cells,
            "game_type_split": use_e14_type_count_cells,
        },
        "rate_hand_cells": {
            "reverse": use_reverse_hand_cells,
            "fastball": use_fastball_hand_cells,
        },
        "e14_hand_bins": {
            "rate": use_e14_rate_hand_bin,
            "n": use_e14_n_hand_bin,
        },
        "team_matchup": {"enabled": use_team_matchup},
        "venue": {"enabled": use_venue, "current_row_only": True},
        "pitcher_profile": pitcher_profile_meta,
        "batter_e14": batter_e14_meta,
        "hierarchical_e14": hierarchical_e14_meta,
        "hierarchical_batter_e14": hierarchical_batter_meta,
        "recent_form": {
            "enabled": use_recent_form,
            "count_cells": use_recent_form_count_cells,
            "current_row_only": True,
        },
        "recent_denominators": {
            "enabled": use_recent_denominators,
            "method": "LCM of reduced success/middle denominators per 1/3/5-game window",
            "current_row_only": True,
        },
        "recent_workload_decoder": recent_workload_meta,
        "batter_e14_interactions": {
            "count_cells": use_batter_e14_count_cells,
            "pitcher_batter": use_pitcher_batter_interactions,
        },
        "auxiliary_season_components": aux_component_meta,
        "current_state_full": current_state_meta,
        "historical_group_rates": history_group_meta,
        "temporal_stable_joint": temporal_stable_meta,
        "outcome_context": outcome_context_meta,
        "entity_context_profiles": entity_profile_meta,
        "pitcher_target_encoder": {
            "enabled": use_pitcher_te,
            "implementation": "sklearn TargetEncoder cross-fit inside model pipeline",
        },
        "baseline_models": baseline_names,
        "baseline_source": baseline_source,
        "models": summaries,
        "fit_details": fit_details,
        "sample_weight": {
            "season_decay": args.season_decay,
            "f_pre_regime_weight": args.f_pre_regime_weight,
            "fit_game_types": args.fit_game_types or ["R", "F"],
            "min": float(train_weight.min()) if train_weight is not None else 1.0,
            "max": float(train_weight.max()) if train_weight is not None else 1.0,
            "effective_rows": (
                float(np.square(train_weight.sum()) / np.square(train_weight).sum())
                if train_weight is not None else float(len(history))
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    del history_all, history, valid, train_x, valid_x, predictions, scored
    gc.collect()
    return result


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    params = json.loads(args.params.read_text(encoding="utf-8")) if args.params else None

    columns = sorted(set(BASE_FEATURES) | {TARGET})
    frame = load_train(args.data)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Training data is missing columns: {missing}")

    joined_trackman: pd.DataFrame | None = None
    raw_trackman: pd.DataFrame | None = None
    main_linkage_frame: pd.DataFrame | None = None
    if any(
        name in args.features
        for name in (
            "trackman",
            "trackman_rich",
            "trackman_stability",
            "trackman_group_stability",
            "trackman_game_repeatability",
            "trackman_inning_physics",
            "trackman_trend",
            "trackman_platoon",
            "trackman_count",
            "trackman_workload",
            "trackman_teacher",
            "trackman_lupi",
            "trackman_archetype",
            "trackman_batter_rich",
            "expanded_trackman_profiles",
            "fine_pitch_latent",
            "auto_pitch_latent",
            "auto_pitch_profile_latent",
            "expanded_auto_pitch_latent",
            "partial_expanded_auto_pitch_latent",
            "matchup_hand_auto_pitch_latent",
        )
    ):
        if (
            "expanded_auto_pitch_latent" in args.features
            or "partial_expanded_auto_pitch_latent" in args.features
            or "matchup_hand_auto_pitch_latent" in args.features
            or "expanded_trackman_profiles" in args.features
            or "trackman_batter_rich" in args.features
        ):
            from experiments.run_e20r_rolling import (  # noqa: WPS433
                load_joined_and_raw_trackman,
            )

            joined_trackman, raw_trackman = load_joined_and_raw_trackman()
            if "partial_expanded_auto_pitch_latent" in args.features:
                from experiments.v5_partial_trackman_linkage import (  # noqa: WPS433
                    load_main_linkage_frame,
                )

                main_linkage_frame = load_main_linkage_frame(args.data)
        else:
            from experiments.run_e20r_rolling import (  # noqa: WPS433
                load_joined_trackman,
            )

            joined_trackman = load_joined_trackman()

    needs_row_id = bool(
        "e22_probs" in args.features
        or "e22_cat" in args.features
        or "fine_pitch_latent" in args.features
        or "auto_pitch_latent" in args.features
        or "auto_pitch_profile_latent" in args.features
        or "expanded_auto_pitch_latent" in args.features
        or "partial_expanded_auto_pitch_latent" in args.features
        or "matchup_hand_auto_pitch_latent" in args.features
    )
    if needs_row_id:
        row_ids = pd.read_csv(
            args.data, usecols=["row_id"], dtype="string", encoding="utf-8-sig"
        )["row_id"]
        if len(row_ids) != len(frame):
            raise AssertionError("row_id length does not match optimized train frame")
        frame.insert(0, "row_id", row_ids.to_numpy())

    if "e22_probs" in args.features or "e22_cat" in args.features:
        from experiments.run_e22r_probs_rolling import load_group_labels  # noqa: WPS433

        labels = load_group_labels()
        frame["e22_pitch_type_group"] = frame["row_id"].map(labels)
        del labels

    if (
        "fine_pitch_latent" in args.features
        or "auto_pitch_latent" in args.features
        or "auto_pitch_profile_latent" in args.features
        or "expanded_auto_pitch_latent" in args.features
        or "partial_expanded_auto_pitch_latent" in args.features
        or "matchup_hand_auto_pitch_latent" in args.features
    ):
        if joined_trackman is None:
            raise RuntimeError("pitch latent feature requires joined TrackMan history")
        fine_label_columns = ["row_id"]
        if "fine_pitch_latent" in args.features:
            fine_label_columns.append("tagged_pitch_type")
        if (
            "auto_pitch_latent" in args.features
            or "auto_pitch_profile_latent" in args.features
            or "expanded_auto_pitch_latent" in args.features
            or "partial_expanded_auto_pitch_latent" in args.features
            or "matchup_hand_auto_pitch_latent" in args.features
        ):
            fine_label_columns.append("auto_pitch_type")
        if "catboost_physics_joint" in args.models:
            fine_label_columns.extend(PHYSICS_AUX_COLUMNS)
        fine_labels = joined_trackman[fine_label_columns].drop_duplicates("row_id")
        fine_labels["row_id"] = fine_labels["row_id"].astype(str)
        source_column = (
            "auto_pitch_type"
            if (
                "auto_pitch_latent" in args.features
                or "auto_pitch_profile_latent" in args.features
                or "expanded_auto_pitch_latent" in args.features
                or "partial_expanded_auto_pitch_latent" in args.features
                or "matchup_hand_auto_pitch_latent" in args.features
            )
            else "tagged_pitch_type"
        )
        normalized = (
            fine_labels[source_column]
            .astype("string")
            .replace(
                {
                    "Changeup": "ChangeUp",
                    "Four-Seam": "Fastball",
                    "SInker": "Sinker",
                }
            )
        )
        label_column = (
            "auto_fine_pitch_type"
            if (
                "auto_pitch_latent" in args.features
                or "auto_pitch_profile_latent" in args.features
                or "expanded_auto_pitch_latent" in args.features
                or "partial_expanded_auto_pitch_latent" in args.features
                or "matchup_hand_auto_pitch_latent" in args.features
            )
            else "fine_pitch_type"
        )
        fine_labels[label_column] = normalized.where(
            normalized.isin(FINE_PITCH_TYPES[:-1]), "Other"
        )
        fine_map = fine_labels.set_index("row_id")[label_column]
        frame[label_column] = frame["row_id"].astype(str).map(fine_map)
        if "catboost_physics_joint" in args.models:
            physics_map = fine_labels.set_index("row_id")
            for column in PHYSICS_AUX_COLUMNS:
                frame[f"_aux_tm_{column}"] = frame["row_id"].astype(str).map(
                    physics_map[column]
                )
        del fine_labels, fine_map, normalized

    if needs_row_id:
        del row_ids

    folds = [
        run_fold(
            frame, season, args, params, joined_trackman, raw_trackman,
            main_linkage_frame,
        )
        for season in sorted(args.validation_seasons)
    ]
    del joined_trackman
    del raw_trackman
    del main_linkage_frame
    del frame
    gc.collect()

    scored_names = list(folds[0]["models"])
    gates: dict[str, Any] = {}
    for name in scored_names:
        intervals = {
            fold["validation_season"]: fold["models"][name]["vs_baseline"] for fold in folds
        }
        primary = 2024 if 2024 in intervals else max(intervals)
        secondary = 2022 if 2022 in intervals else min(intervals)
        gates[name] = aggregate_gate(intervals, primary, secondary)
        gates[name]["mean_score"] = float(
            np.mean([fold["models"][name]["summary"]["competition_score"] for fold in folds])
        )
        gates[name]["primary_score"] = float(
            next(
                fold["models"][name]["summary"]["competition_score"]
                for fold in folds
                if fold["validation_season"] == primary
            )
        )

    payload = {
        "metadata": {
            "stage": args.stage,
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "data": str(args.data),
            "models": args.models,
            "features": args.features,
            "feature_view": args.feature_view,
            "blend": args.blend,
            "prior_mode": args.prior_mode,
            "history_window": args.history_window,
            "f_regime_start": args.f_regime_start,
            "season_decay": args.season_decay,
            "f_pre_regime_weight": args.f_pre_regime_weight,
            "inner_validation": args.inner_validation,
            "booster_params": params,
            "outcome_scheme": args.outcome_scheme,
            "booster_device": os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower(),
            "validation_seasons": sorted(args.validation_seasons),
            "protocol": "outer history season < Y; validation season == Y",
            "row_independent_inference": True,
            "encoder_cutoff": "season-wise out-of-fold on history; frozen full-history for the fold",
            "bootstrap_unit": "pitcher_id cluster",
            "inference_note": "2024 is a development fold; post-selection intervals are exploratory",
            "smoke_test": bool(args.max_history_rows or args.max_valid_rows),
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "command": " ".join(sys.argv),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "gates": gates,
        "folds": folds,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.stage}.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    rows = [
        {
            "stage": args.stage,
            "validation_season": fold["validation_season"],
            "model": name,
            "brier": item["summary"]["brier"],
            "competition_score": item["summary"]["competition_score"],
            "delta_vs_baseline": item["vs_baseline"]["point"],
            "ci_low": item["vs_baseline"]["ci_low"],
            "ci_high": item["vs_baseline"]["ci_high"],
            "significant": item["vs_baseline"]["significant"],
        }
        for fold in folds
        for name, item in fold["models"].items()
    ]
    pd.DataFrame(rows).to_csv(args.output_dir / f"{args.stage}.csv", index=False)

    print("\nGate summary:", flush=True)
    for name, gate in gates.items():
        print(
            f"  {name:<12} primary({gate['primary_season']})="
            f"{gate['primary_score']:,.1f} pass={gate['gate_pass']}",
            flush=True,
        )
    print(f"Saved {json_path}.", flush=True)


if __name__ == "__main__":
    main()
