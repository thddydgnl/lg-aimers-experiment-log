#!/usr/bin/env python3
"""Evaluate fixed public residual post-processing on the locked OOT stack.

Only official train rows are read.  The public recipe hyperparameters are
copied without looking at the 2024 labels: three pitcher context contrasts
(hand, two-strike, runners) plus very strongly shrunk pitcher/batter main
effects.  A 2022 -> 2023 selection transfer is reported before the untouched
2024 confirmation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    score,
)


PRED = ROOT / "experiments/results/predictions"
REPORT = ROOT / "experiments/results/v4_public_residual_postprocess.json"
YEARS = (2022, 2023, 2024)
CONTEXTS = {
    "same_hand": 1000.0,
    "two_strike": 1000.0,
    "runner_present": 2000.0,
}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    return score(y, np.clip(pred, 0.0, 1.0))


def main_table(ids: np.ndarray, residual: np.ndarray, k: float) -> pd.Series:
    frame = pd.DataFrame({"id": ids, "residual": residual})
    grouped = frame.groupby("id", observed=True)["residual"].agg(["mean", "size"])
    return grouped["mean"] * grouped["size"] / (grouped["size"] + k)


def contrast_table(
    ids: np.ndarray,
    context: np.ndarray,
    residual: np.ndarray,
    k: float,
) -> pd.Series:
    frame = pd.DataFrame({"id": ids, "context": context, "residual": residual})
    grouped = frame.groupby(["id", "context"], observed=True)["residual"].agg(
        ["mean", "size"]
    )
    mean = grouped["mean"].unstack("context")
    size = grouped["size"].unstack("context").fillna(0.0)
    if 0 not in mean or 1 not in mean:
        return pd.Series(dtype=np.float64)
    valid = mean[0].notna() & mean[1].notna()
    n0 = size[0]
    n1 = size[1]
    effective_n = n0 * n1 / (n0 + n1).replace(0.0, np.nan)
    differential = (mean[1] - mean[0]) * effective_n / (effective_n + k)
    return differential[valid].dropna()


def map_values(ids: np.ndarray, table: pd.Series) -> np.ndarray:
    return pd.Series(ids).map(table).fillna(0.0).to_numpy(np.float64)


def group_direction(
    source: pd.DataFrame,
    residual: np.ndarray,
    target: pd.DataFrame,
    columns: list[str],
    k: float,
) -> np.ndarray:
    work = source.loc[:, columns].copy()
    work["residual"] = residual
    grouped = work.groupby(columns, observed=True)["residual"].agg(["mean", "size"])
    table = grouped["mean"] * grouped["size"] / (grouped["size"] + k)
    target_index = pd.MultiIndex.from_frame(target.loc[:, columns])
    return table.reindex(target_index).fillna(0.0).to_numpy(np.float64)


def fit_without_intercept(x: np.ndarray, residual: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, residual, rcond=None)[0]


def fit_standardized_ridge(
    x: np.ndarray, residual: np.ndarray, penalty: float
) -> np.ndarray:
    scale = np.sqrt(np.mean(np.square(x), axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = x / scale
    gram = standardized.T @ standardized
    regularizer = float(len(x) * penalty) * np.eye(x.shape[1])
    beta_standardized = np.linalg.solve(
        gram + regularizer,
        standardized.T @ residual,
    )
    return beta_standardized / scale


def main() -> None:
    columns = [
        "pitcher_id",
        "batter_id",
        "pitcher_hand",
        "batter_hand",
        "strikes_before",
        "num_runners_on",
        "inning",
        "balls_before",
        "game_type",
    ]
    train = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=columns,
        encoding="utf-8-sig",
        low_memory=False,
    )
    train["same_hand"] = (
        train["pitcher_hand"].astype(str) == train["batter_hand"].astype(str)
    ).astype(np.int8)
    train["two_strike"] = (train["strikes_before"].astype(float) == 2).astype(np.int8)
    train["runner_present"] = (train["num_runners_on"].astype(float) > 0).astype(np.int8)

    accepted: dict[int, dict[str, np.ndarray]] = {}
    frames: dict[int, pd.DataFrame] = {}
    for year in YEARS:
        item = load_npz(PRED / f"v4_routed_tabm_stack_locked_{year}.npz")
        accepted[year] = item
        frames[year] = train.iloc[item["row_index"].astype(np.int64)].reset_index(drop=True)

    deep: dict[int, dict[str, np.ndarray]] = {}
    for year in (2023, 2024):
        item = load_npz(PRED / f"v4_deep_oof_stacker_{year}.npz")
        if not np.array_equal(item["row_index"], accepted[year]["row_index"]):
            raise ValueError(f"Deep artifact alignment mismatch for {year}")
        if not np.allclose(item["accepted"], accepted[year]["routed_tabm_stack"], atol=1e-12):
            raise ValueError(f"Deep accepted baseline mismatch for {year}")
        deep[year] = item

    residuals = {
        year: accepted[year]["y"].astype(np.float64)
        - accepted[year]["routed_tabm_stack"].astype(np.float64)
        for year in YEARS
    }

    def public_components(target_year: int, source_years: tuple[int, ...]) -> dict[str, np.ndarray]:
        source = pd.concat([frames[year] for year in source_years], ignore_index=True)
        residual = np.concatenate([residuals[year] for year in source_years])
        target = frames[target_year]
        pitcher = source["pitcher_id"].to_numpy()
        target_pitcher = target["pitcher_id"].to_numpy()
        out: dict[str, np.ndarray] = {}
        for name, k in CONTEXTS.items():
            table = contrast_table(
                pitcher,
                source[name].to_numpy(np.int8),
                residual,
                k,
            )
            mapped = map_values(target_pitcher, table)
            sign = np.where(target[name].to_numpy(np.int8) == 1, 0.5, -0.5)
            out[name] = 0.65 * mapped * sign
        out["pitcher_main"] = map_values(
            target_pitcher,
            main_table(pitcher, residual, 50000.0),
        )
        out["batter_main"] = 2.5 * map_values(
            target["batter_id"].to_numpy(),
            main_table(source["batter_id"].to_numpy(), residual, 20000.0),
        )
        out["fixed_total"] = sum(out.values(), np.zeros(len(target), dtype=np.float64))
        return out

    public = {
        2023: public_components(2023, (2022,)),
        2024: public_components(2024, (2022, 2023)),
    }

    # Reconstruct the earlier inning/count persistence probe under every
    # meaningful R/F routing choice.  This is diagnostic only: variants are
    # ranked on 2023, while 2024 remains a reported confirmation.
    group_variant_scan: list[dict[str, object]] = []
    locked_residuals = {
        year: accepted[year]["y"].astype(np.float64)
        - accepted[year]["locked"].astype(np.float64)
        for year in YEARS
    }
    group_directions: dict[str, dict[int, np.ndarray]] = {}
    for residual_name, source_residuals in (
        ("accepted", residuals),
        ("locked", locked_residuals),
    ):
        for source_route in ("all", "R"):
            for apply_route in ("all", "R"):
                for fit_scope in ("all", "R"):
                    name = (
                        f"resid={residual_name}|source={source_route}|"
                        f"apply={apply_route}|fit={fit_scope}"
                    )
                    directions: dict[int, np.ndarray] = {}
                    for target_year, source_year in ((2023, 2022), (2024, 2023)):
                        source_frame = frames[source_year]
                        source_residual = source_residuals[source_year]
                        if source_route == "R":
                            source_mask = source_frame["game_type"].astype(str).to_numpy() == "R"
                            source_frame = source_frame.loc[source_mask].reset_index(drop=True)
                            source_residual = source_residual[source_mask]
                        direction = group_direction(
                            source_frame,
                            source_residual,
                            frames[target_year],
                            ["inning", "balls_before", "strikes_before"],
                            20.0,
                        )
                        if apply_route == "R":
                            target_r = (
                                frames[target_year]["game_type"].astype(str).to_numpy() == "R"
                            )
                            direction = np.where(target_r, direction, 0.0)
                        directions[target_year] = direction
                    fit_mask = np.ones(len(directions[2023]), dtype=bool)
                    if fit_scope == "R":
                        fit_mask = frames[2023]["game_type"].astype(str).to_numpy() == "R"
                    d23 = directions[2023][fit_mask]
                    r23 = residuals[2023][fit_mask]
                    denominator = float(np.dot(d23, d23))
                    gamma = float(np.dot(d23, r23) / denominator) if denominator else 0.0
                    gains: dict[str, float] = {}
                    for target_year in (2023, 2024):
                        y_target = accepted[target_year]["y"].astype(np.float64)
                        base_target = accepted[target_year]["routed_tabm_stack"].astype(np.float64)
                        gains[str(target_year)] = float(
                            metric(y_target, base_target + gamma * directions[target_year])[
                                "raw_competition_score"
                            ]
                            - metric(y_target, base_target)["raw_competition_score"]
                        )
                    group_directions[name] = directions
                    group_variant_scan.append(
                        {"name": name, "gamma": gamma, "gains": gains}
                    )
    group_variant_scan.sort(
        key=lambda item: float(item["gains"]["2023"]), reverse=True
    )

    deep_direction: dict[int, np.ndarray] = {}
    inning_direction: dict[int, np.ndarray] = {}
    for year, source_year in ((2023, 2022), (2024, 2023)):
        route = frames[year]["game_type"].astype(str).to_numpy() == "R"
        raw = deep[year]["raw_tabm_bce_brier"].astype(np.float64)
        if len(raw) != int(route.sum()):
            raise ValueError(f"Unexpected raw deep prediction length for {year}")
        direction = np.zeros(len(route), dtype=np.float64)
        direction[route] = raw - deep[year]["accepted"].astype(np.float64)[route]
        deep_direction[year] = direction
        inning_source_frame = frames[source_year]
        inning_source_residual = residuals[source_year]
        if year == 2024:
            inning_source_frame = pd.concat(
                [frames[2022], frames[2023]], ignore_index=True
            )
            inning_source_residual = np.concatenate(
                [residuals[2022], residuals[2023]]
            )
        inning_direction[year] = group_direction(
            inning_source_frame,
            inning_source_residual,
            frames[year],
            ["inning", "balls_before", "strikes_before"],
            20.0,
        )
        # The persistence signal is an R-regime correction.  F rows keep the
        # accepted prediction untouched, exactly like the neural direction.
        inning_direction[year] = np.where(route, inning_direction[year], 0.0)

    x23 = np.column_stack((deep_direction[2023], inning_direction[2023]))
    joint_coef = fit_without_intercept(x23, residuals[2023])
    context_joint = {
        year: accepted[year]["routed_tabm_stack"].astype(np.float64)
        + np.column_stack((deep_direction[year], inning_direction[year])) @ joint_coef
        for year in (2023, 2024)
    }

    public_scalar = float(
        fit_without_intercept(public[2023]["fixed_total"][:, None], residuals[2023])[0]
    )
    three_coef = fit_without_intercept(
        np.column_stack((deep_direction[2023], inning_direction[2023], public[2023]["fixed_total"])),
        residuals[2023],
    )

    component_names = [*CONTEXTS.keys(), "pitcher_main", "batter_main"]

    def meta_design(year: int, kind: str) -> np.ndarray:
        route_r = frames[year]["game_type"].astype(str).to_numpy() == "R"
        fixed = public[year]["fixed_total"]
        leading = [deep_direction[year], inning_direction[year]]
        if kind == "split_total":
            return np.column_stack(
                [*leading, np.where(route_r, fixed, 0.0), np.where(~route_r, fixed, 0.0)]
            )
        components = [public[year][name] for name in component_names]
        if kind == "components_all":
            return np.column_stack([*leading, *components])
        if kind == "components_regime":
            routed = [np.where(route_r, value, 0.0) for value in components]
            routed += [np.where(~route_r, value, 0.0) for value in components]
            return np.column_stack([*leading, *routed])
        raise ValueError(kind)

    meta_coefficients: dict[str, dict[str, object]] = {}
    for kind in ("split_total", "components_all", "components_regime"):
        design23 = meta_design(2023, kind)
        fitters = {"lstsq": fit_without_intercept(design23, residuals[2023])}
        for penalty in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            fitters[f"ridge_{penalty:g}"] = fit_standardized_ridge(
                design23, residuals[2023], penalty
            )
        for fit_name, coefficients in fitters.items():
            meta_coefficients[f"{kind}_{fit_name}"] = {
                "kind": kind,
                "coefficients": coefficients,
            }
    # R coefficients are learned on 2023.  The F coefficient remains the
    # external recipe's fixed value 1.0 because the freely fitted F value is
    # based on a much smaller, season-sensitive subset.
    split_selected = np.asarray(
        meta_coefficients["split_total_lstsq"]["coefficients"], dtype=np.float64
    ).copy()
    split_selected[3] = 1.0
    meta_coefficients["split_r_selected_f_fixed"] = {
        "kind": "split_total",
        "coefficients": split_selected,
    }

    results: dict[str, dict[str, object]] = {}
    predictions: dict[int, dict[str, np.ndarray]] = {}
    for year in (2023, 2024):
        base = accepted[year]["routed_tabm_stack"].astype(np.float64)
        y = accepted[year]["y"].astype(np.float64)
        route_r = frames[year]["game_type"].astype(str).to_numpy() == "R"
        fixed = public[year]["fixed_total"]
        fixed_r = np.where(route_r, fixed, 0.0)
        fixed_f = np.where(~route_r, fixed, 0.0)
        matrix3 = np.column_stack((deep_direction[year], inning_direction[year], fixed))
        candidates = {
            "accepted": base,
            "deep_context_joint": context_joint[year],
            "public_fixed_all": base + fixed,
            "public_fixed_r": base + fixed_r,
            "public_fixed_f": base + fixed_f,
            "public_scaled_all": base + public_scalar * fixed,
            "deep_context_plus_public_fixed": context_joint[year] + fixed,
            "joint_three_direction": base + matrix3 @ three_coef,
        }
        for name, spec in meta_coefficients.items():
            candidates[name] = (
                base
                + meta_design(year, str(spec["kind"]))
                @ np.asarray(spec["coefficients"], dtype=np.float64)
            )
        base_score = metric(y, base)["raw_competition_score"]
        results[str(year)] = {
            name: {
                "metrics": metric(y, pred),
                "gain_over_accepted": float(metric(y, pred)["raw_competition_score"] - base_score),
            }
            for name, pred in candidates.items()
        }
        results[str(year)]["component_std"] = {
            name: float(values.std()) for name, values in public[year].items()
        }
        predictions[year] = candidates
        np.savez_compressed(
            PRED / f"v4_public_residual_postprocess_{year}.npz",
            y=y,
            row_index=accepted[year]["row_index"],
            cluster=accepted[year]["cluster"],
            deep_direction=deep_direction[year],
            inning_direction=inning_direction[year],
            **{f"public_{name}": values for name, values in public[year].items()},
            **candidates,
        )

    best_2024 = max(
        (
            (name, entry)
            for name, entry in results["2024"].items()
            if name != "component_std"
        ),
        key=lambda item: item[1]["metrics"]["raw_competition_score"],
    )
    local = float(best_2024[1]["metrics"]["raw_competition_score"])
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "public_recipe_hyperparameters_fixed_before_2024_confirmation": True,
            "selection_target": 2023,
            "selection_source": [2022],
            "confirmation_target": 2024,
            "confirmation_sources": [2022, 2023],
            "row_independent_inference": True,
        },
        "public_recipe": {
            "context_weight": 0.65,
            "context_k": CONTEXTS,
            "pitcher_main_k": 50000.0,
            "pitcher_main_weight": 1.0,
            "batter_main_k": 20000.0,
            "batter_main_weight": 2.5,
        },
        "selection_coefficients": {
            "deep_context_joint": {
                "deep_raw_tabm_bce_brier": float(joint_coef[0]),
                "inning_balls_strikes_k20": float(joint_coef[1]),
            },
            "public_total_scalar": public_scalar,
            "joint_three_direction": {
                "deep_raw_tabm_bce_brier": float(three_coef[0]),
                "inning_balls_strikes_k20": float(three_coef[1]),
                "public_fixed_total": float(three_coef[2]),
            },
            "expanded_meta_models": {
                name: {
                    "kind": value["kind"],
                    "coefficients": np.asarray(value["coefficients"]).tolist(),
                }
                for name, value in meta_coefficients.items()
            },
        },
        "inning_group_variant_scan": group_variant_scan,
        "folds": results,
        "best_observed_2024_diagnostic": {
            "name": best_2024[0],
            "local_score": local,
            "expected_lb_median": local + MEDIAN_OFFSET,
            "required_local_score": REQUIRED_LOCAL,
            "crosses_required_local_score": local > REQUIRED_LOCAL,
            "warning": "2024 is confirmation-only; best-observed ranking is not a selection rule",
        },
        "prediction_artifacts": {
            str(year): str(
                (PRED / f"v4_public_residual_postprocess_{year}.npz").relative_to(ROOT)
            )
            for year in (2023, 2024)
        },
    }
    REPORT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    compact = {
        "coefficients": report["selection_coefficients"],
        "selection_2023_gains": {
            key: value["gain_over_accepted"]
            for key, value in results["2023"].items()
            if key != "component_std"
        },
        "confirmation_2024_gains": {
            key: value["gain_over_accepted"]
            for key, value in results["2024"].items()
            if key != "component_std"
        },
        "top_inning_group_variants": group_variant_scan[:8],
        "best_observed_2024_diagnostic": report["best_observed_2024_diagnostic"],
    }
    print(json.dumps(json_safe(compact), ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {REPORT}", flush=True)


if __name__ == "__main__":
    main()
