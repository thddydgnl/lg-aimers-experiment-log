#!/usr/bin/env python3
"""Streaming EDA for the LG Aimers 9th competition data.

The script intentionally uses only the Python standard library so that it can
run in a clean environment without pandas/numpy. It scans each large CSV once,
keeps exact aggregates where practical, and uses a deterministic systematic
sample only for quantiles and correlation pairs between input features.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "open" / "data"
RESULT_DIR = ROOT / "eda" / "results"
FIGURE_DIR = ROOT / "eda" / "figures"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
TRACKMAN_PATH = DATA_DIR / "trackman_history.csv"

TRAIN_SAMPLE_STRIDE = 50
TRACKMAN_SAMPLE_STRIDE = 75


class NumericStats:
    __slots__ = (
        "n",
        "missing",
        "total",
        "total_sq",
        "minimum",
        "maximum",
        "sum_y",
        "sum_xy",
        "corr_n",
    )

    def __init__(self) -> None:
        self.n = 0
        self.missing = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.sum_y = 0.0
        self.sum_xy = 0.0
        self.corr_n = 0

    def update(self, value: float | None, target: int | None = None) -> None:
        if value is None:
            self.missing += 1
            return
        self.n += 1
        self.total += value
        self.total_sq += value * value
        if value < self.minimum:
            self.minimum = value
        if value > self.maximum:
            self.maximum = value
        if target is not None:
            self.sum_y += target
            self.sum_xy += value * target
            self.corr_n += 1

    def mean(self) -> float | None:
        return self.total / self.n if self.n else None

    def std(self) -> float | None:
        if self.n < 2:
            return None
        variance = max(0.0, (self.total_sq - self.total * self.total / self.n) / (self.n - 1))
        return math.sqrt(variance)

    def correlation_with_binary_target(self) -> float | None:
        n = self.corr_n
        if n < 2:
            return None
        sum_x = self.total
        sum_xx = self.total_sq
        sum_y = self.sum_y
        sum_yy = self.sum_y  # y is binary
        numerator = n * self.sum_xy - sum_x * sum_y
        denominator = math.sqrt(max(0.0, (n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)))
        return numerator / denominator if denominator else None


def parse_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def parse_int(value: str) -> int | None:
    if value == "":
        return None
    return int(float(value))


def quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize_samples(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    return {
        "sample_n": len(ordered),
        "p001": quantile(ordered, 0.001),
        "p01": quantile(ordered, 0.01),
        "p05": quantile(ordered, 0.05),
        "p25": quantile(ordered, 0.25),
        "p50": quantile(ordered, 0.50),
        "p75": quantile(ordered, 0.75),
        "p95": quantile(ordered, 0.95),
        "p99": quantile(ordered, 0.99),
        "p999": quantile(ordered, 0.999),
    }


def summarize_numeric(stats: NumericStats, samples: list[float], total_rows: int) -> dict[str, Any]:
    result = {
        "count": stats.n,
        "missing": stats.missing,
        "missing_rate": stats.missing / total_rows if total_rows else None,
        "mean": stats.mean(),
        "std": stats.std(),
        "min": stats.minimum if stats.n else None,
        "max": stats.maximum if stats.n else None,
        "target_correlation": stats.correlation_with_binary_target(),
    }
    result.update(summarize_samples(samples))
    return result


def add_target_group(groups: dict[str, dict[str, list[float]]], name: str, key: Any, target: int) -> None:
    label = str(key)
    bucket = groups[name].get(label)
    if bucket is None:
        groups[name][label] = [1.0, float(target)]
    else:
        bucket[0] += 1.0
        bucket[1] += target


def serialize_target_groups(groups: dict[str, dict[str, list[float]]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for name, mapping in groups.items():
        rows = []
        for key, (count, target_sum) in mapping.items():
            rows.append({"key": key, "count": int(count), "target_sum": int(target_sum), "target_rate": target_sum / count})
        output[name] = rows
    return output


def experience_bucket(value: int | None) -> str:
    if value is None:
        return "missing"
    if value == 0:
        return "0"
    if value < 10:
        return "1-9"
    if value < 50:
        return "10-49"
    if value < 200:
        return "50-199"
    if value < 1000:
        return "200-999"
    return "1000+"


def leverage_bucket(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 0.5:
        return "<0.5"
    if value < 1.0:
        return "0.5-<1"
    if value < 2.0:
        return "1-<2"
    if value < 4.0:
        return "2-<4"
    return "4+"


def score_diff_bucket(value: int | None) -> str:
    if value is None:
        return "missing"
    if value <= -5:
        return "<=-5"
    if value <= -2:
        return "-4..-2"
    if value == -1:
        return "-1"
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 4:
        return "2..4"
    return "5+"


def inning_bucket(value: int | None) -> str:
    if value is None:
        return "missing"
    if value <= 3:
        return "1-3"
    if value <= 6:
        return "4-6"
    if value <= 9:
        return "7-9"
    return "10+"


def rate_bucket(value: float | None) -> str:
    if value is None:
        return "missing"
    if value >= 1.0:
        return "0.9-1.0"
    index = max(0, min(9, int(value * 10)))
    return f"{index / 10:.1f}-{(index + 1) / 10:.1f}"


def update_calibration(
    calibration: dict[str, dict[str, list[float]]], feature: str, value: float | None, target: int
) -> None:
    key = rate_bucket(value)
    bucket = calibration[feature].get(key)
    if bucket is None:
        calibration[feature][key] = [1.0, 0.0 if value is None else value, float(target)]
    else:
        bucket[0] += 1.0
        if value is not None:
            bucket[1] += value
        bucket[2] += target


def serialize_calibration(calibration: dict[str, dict[str, list[float]]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for feature, mapping in calibration.items():
        rows = []
        for key, (count, prediction_sum, target_sum) in mapping.items():
            rows.append(
                {
                    "bin": key,
                    "count": int(count),
                    "mean_feature": None if key == "missing" else prediction_sum / count,
                    "target_rate": target_sum / count,
                }
            )
        output[feature] = rows
    return output


def pairwise_correlation(sample_rows: list[list[float | None]], columns: list[str]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for left_index in range(len(columns)):
        for right_index in range(left_index + 1, len(columns)):
            n = 0
            sx = sy = sxx = syy = sxy = 0.0
            for row in sample_rows:
                x = row[left_index]
                y = row[right_index]
                if x is None or y is None:
                    continue
                n += 1
                sx += x
                sy += y
                sxx += x * x
                syy += y * y
                sxy += x * y
            denominator = math.sqrt(max(0.0, (n * sxx - sx * sx) * (n * syy - sy * sy))) if n > 1 else 0.0
            corr = (n * sxy - sx * sy) / denominator if denominator else None
            if corr is not None:
                pairs.append({"left": columns[left_index], "right": columns[right_index], "n": n, "correlation": corr})
    pairs.sort(key=lambda item: abs(item["correlation"]), reverse=True)
    return pairs


def frequency_summary(counter: Counter[str]) -> dict[str, Any]:
    frequencies = sorted(counter.values())
    total = sum(frequencies)
    top = counter.most_common(10)
    return {
        "unique": len(counter),
        "frequency_min": frequencies[0] if frequencies else None,
        "frequency_p25": quantile([float(v) for v in frequencies], 0.25) if frequencies else None,
        "frequency_p50": quantile([float(v) for v in frequencies], 0.50) if frequencies else None,
        "frequency_p75": quantile([float(v) for v in frequencies], 0.75) if frequencies else None,
        "frequency_p95": quantile([float(v) for v in frequencies], 0.95) if frequencies else None,
        "frequency_max": frequencies[-1] if frequencies else None,
        "top10_share": sum(count for _, count in top) / total if total else None,
        "top10": [{"id": key, "count": count} for key, count in top],
    }


def entity_target_summary(mapping: dict[str, list[int]], minimum_count: int = 1000) -> dict[str, Any]:
    counts = Counter({key: values[0] for key, values in mapping.items()})
    eligible = [
        {"id": key, "count": values[0], "target_rate": values[1] / values[0]}
        for key, values in mapping.items()
        if values[0] >= minimum_count
    ]
    eligible.sort(key=lambda item: item["target_rate"])
    return {
        **frequency_summary(counts),
        "minimum_count_for_extremes": minimum_count,
        "eligible_entities": len(eligible),
        "lowest_target_rate": eligible[:10],
        "highest_target_rate": list(reversed(eligible[-10:])),
    }


def category_effects(serialized_groups: dict[str, list[dict[str, Any]]], overall_rate: float) -> list[dict[str, Any]]:
    effects = []
    excluded = {"row_chunk", "pitcher_id", "batter_id", "season_game_type"}
    for name, rows in serialized_groups.items():
        if name in excluded or len(rows) < 2:
            continue
        total = sum(row["count"] for row in rows)
        variance = sum(row["count"] * (row["target_rate"] - overall_rate) ** 2 for row in rows) / total
        effects.append(
            {
                "feature": name,
                "groups": len(rows),
                "weighted_target_rate_std": math.sqrt(variance),
                "min_target_rate": min(row["target_rate"] for row in rows),
                "max_target_rate": max(row["target_rate"] for row in rows),
            }
        )
    effects.sort(key=lambda item: item["weighted_target_rate_std"], reverse=True)
    return effects


def scan_train() -> tuple[dict[str, Any], Counter[tuple[Any, ...]], Counter[str], Counter[str]]:
    numeric_columns = [
        "season",
        "game_month",
        "game_dayofweek",
        "inning",
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
        "home_win_expectancy",
        "away_win_expectancy",
        "li",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
        "asof_batter_middle_rate",
        "asof_pitcher_pitchmix_n",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]
    correlation_columns = [
        "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
        "asof_batter_success_rate",
        "asof_batter_middle_rate",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]
    calibration_columns = [
        "asof_pitcher_success_rate",
        "asof_batter_success_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ]

    stats = {column: NumericStats() for column in numeric_columns}
    samples = {column: [] for column in numeric_columns}
    sampled_correlations: list[list[float | None]] = []
    missing_counts: Counter[str] = Counter()
    missing_target_sum: Counter[str] = Counter()
    groups: dict[str, dict[str, list[float]]] = defaultdict(dict)
    calibration: dict[str, dict[str, list[float]]] = defaultdict(dict)
    pitcher_targets: dict[str, list[int]] = {}
    batter_targets: dict[str, list[int]] = {}
    pitcher_counts: Counter[str] = Counter()
    batter_counts: Counter[str] = Counter()
    categorical_counts: dict[str, Counter[str]] = {
        name: Counter()
        for name in [
            "top_bottom",
            "game_type",
            "base_state",
            "pitcher_hand",
            "batter_hand",
            "pitcher_team_id",
            "batter_team_id",
        ]
    }
    feature_hashes: dict[int, int] = {}
    context_keys: Counter[tuple[Any, ...]] = Counter()
    invariant_violations: Counter[str] = Counter()
    data_quality_notes: Counter[str] = Counter()
    row_count = 0
    target_sum = 0
    first_row_by_season: dict[str, str] = {}
    last_row_by_season: dict[str, str] = {}
    season_transitions = 0
    previous_season: str | None = None

    with TRAIN_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        index = {name: pos for pos, name in enumerate(header)}
        expected_columns = set(numeric_columns + correlation_columns + calibration_columns + ["row_id", "control_success"])
        missing_expected_columns = sorted(expected_columns - set(header))
        if missing_expected_columns:
            raise RuntimeError(f"Missing train columns: {missing_expected_columns}")

        num_positions = [(column, index[column]) for column in numeric_columns]
        corr_positions = [index[column] for column in correlation_columns]
        categorical_positions = [(column, index[column]) for column in categorical_counts]
        target_pos = index["control_success"]
        row_id_pos = index["row_id"]

        for row_count, row in enumerate(reader, start=1):
            if len(row) != len(header):
                invariant_violations["row_length"] += 1
                continue
            target_raw = row[target_pos]
            if target_raw not in {"0", "1"}:
                invariant_violations["target_not_binary"] += 1
                continue
            target = int(target_raw)
            target_sum += target
            row_id = row[row_id_pos]
            expected_row_id = f"TRAIN_{row_count:07d}"
            if row_id != expected_row_id:
                invariant_violations["row_id_not_sequential"] += 1

            season = row[index["season"]]
            first_row_by_season.setdefault(season, row_id)
            last_row_by_season[season] = row_id
            if previous_season is not None and previous_season != season:
                season_transitions += 1
            previous_season = season

            values: dict[str, float | None] = {}
            is_sample = row_count % TRAIN_SAMPLE_STRIDE == 0
            for column, position in num_positions:
                value = parse_float(row[position])
                values[column] = value
                stats[column].update(value, target)
                if value is None:
                    missing_counts[column] += 1
                    missing_target_sum[column] += target
                elif is_sample:
                    samples[column].append(value)

            for column, position in categorical_positions:
                value = row[position]
                if value == "":
                    missing_counts[column] += 1
                    missing_target_sum[column] += target
                else:
                    categorical_counts[column][value] += 1

            for column in calibration_columns:
                update_calibration(calibration, column, values[column], target)
            if is_sample:
                sampled_correlations.append([parse_float(row[position]) for position in corr_positions])

            pitcher_id = row[index["pitcher_id"]]
            batter_id = row[index["batter_id"]]
            pitcher_counts[pitcher_id] += 1
            batter_counts[batter_id] += 1
            pitcher_bucket = pitcher_targets.get(pitcher_id)
            if pitcher_bucket is None:
                pitcher_targets[pitcher_id] = [1, target]
            else:
                pitcher_bucket[0] += 1
                pitcher_bucket[1] += target
            batter_bucket = batter_targets.get(batter_id)
            if batter_bucket is None:
                batter_targets[batter_id] = [1, target]
            else:
                batter_bucket[0] += 1
                batter_bucket[1] += target

            balls = int(values["balls_before"] or 0)
            strikes = int(values["strikes_before"] or 0)
            outs = int(values["outs_before"] or 0)
            inning = int(values["inning"] or 0)
            num_runners = int(values["num_runners_on"] or 0)
            pitcher_n = parse_int(row[index["asof_pitcher_n"]])
            batter_n = parse_int(row[index["asof_batter_n"]])

            add_target_group(groups, "season", season, target)
            add_target_group(groups, "game_month", int(values["game_month"] or 0), target)
            add_target_group(groups, "game_dayofweek", int(values["game_dayofweek"] or 0), target)
            add_target_group(groups, "inning", inning, target)
            add_target_group(groups, "inning_bucket", inning_bucket(inning), target)
            add_target_group(groups, "top_bottom", row[index["top_bottom"]], target)
            game_type = row[index["game_type"]]
            add_target_group(groups, "game_type", game_type, target)
            add_target_group(groups, "season_game_type", f"{season}-{game_type}", target)
            add_target_group(groups, "count", f"{balls}-{strikes}", target)
            add_target_group(groups, "outs", outs, target)
            add_target_group(groups, "base_state", row[index["base_state"]], target)
            add_target_group(groups, "num_runners", num_runners, target)
            add_target_group(groups, "pitcher_hand", row[index["pitcher_hand"]], target)
            add_target_group(groups, "batter_hand", row[index["batter_hand"]], target)
            add_target_group(groups, "pitcher_team_id", row[index["pitcher_team_id"]], target)
            add_target_group(groups, "batter_team_id", row[index["batter_team_id"]], target)
            add_target_group(
                groups,
                "hand_matchup",
                f"P{row[index['pitcher_hand']]}-B{row[index['batter_hand']]}",
                target,
            )
            add_target_group(groups, "pitcher_experience", experience_bucket(pitcher_n), target)
            add_target_group(groups, "batter_experience", experience_bucket(batter_n), target)
            add_target_group(groups, "leverage", leverage_bucket(values["li"]), target)
            add_target_group(groups, "score_diff_pitcher_team", score_diff_bucket(parse_int(row[index["score_diff_pitcher_team"]])), target)
            add_target_group(groups, "row_chunk", (row_count - 1) // 150000 + 1, target)

            r1 = int(values["runner_on_1b"] or 0)
            r2 = int(values["runner_on_2b"] or 0)
            r3 = int(values["runner_on_3b"] or 0)
            if balls not in {0, 1, 2, 3}:
                invariant_violations["balls_out_of_range"] += 1
            if strikes not in {0, 1, 2}:
                invariant_violations["strikes_out_of_range"] += 1
            if outs not in {0, 1, 2}:
                invariant_violations["outs_out_of_range"] += 1
            if any(value not in {0, 1} for value in (r1, r2, r3)):
                invariant_violations["runner_flag_not_binary"] += 1
            if r1 + r2 + r3 != num_runners:
                invariant_violations["runner_count_mismatch"] += 1
            expected_base = ("1" if r1 else "_") + ("2" if r2 else "_") + ("3" if r3 else "_")
            if row[index["base_state"]] != expected_base:
                invariant_violations["base_state_mismatch"] += 1
            if abs((values["run_top_before"] or 0) + (values["run_bot_before"] or 0) - (values["run_total_before"] or 0)) > 1e-9:
                invariant_violations["run_total_mismatch"] += 1
            expected_home_diff = (values["run_bot_before"] or 0) - (values["run_top_before"] or 0)
            if abs(expected_home_diff - (values["score_diff_home"] or 0)) > 1e-9:
                invariant_violations["home_score_diff_mismatch"] += 1
            expected_pitcher_diff = expected_home_diff if row[index["top_bottom"]] == "T" else -expected_home_diff
            if abs(expected_pitcher_diff - (values["score_diff_pitcher_team"] or 0)) > 1e-9:
                invariant_violations["pitcher_score_diff_mismatch"] += 1
            home_we = values["home_win_expectancy"]
            away_we = values["away_win_expectancy"]
            if home_we is None or away_we is None or abs(home_we + away_we - 100.0) > 0.100001:
                invariant_violations["win_expectancy_not_100"] += 1
            elif abs(home_we + away_we - 100.0) > 1e-6:
                rounded_delta = round(home_we + away_we - 100.0, 1)
                data_quality_notes[f"win_expectancy_sum_rounding_{rounded_delta:+.1f}"] += 1

            for column in numeric_columns:
                if column.endswith("_rate"):
                    value = values[column]
                    if value is not None and not (0.0 <= value <= 1.0):
                        invariant_violations[f"{column}_out_of_range"] += 1
            mix_values = [
                values["asof_pitcher_fastball_rate"],
                values["asof_pitcher_breaking_rate"],
                values["asof_pitcher_offspeed_rate"],
            ]
            if all(value is not None for value in mix_values) and abs(sum(value for value in mix_values if value is not None) - 1.0) > 2e-6:
                invariant_violations["pitchmix_rates_not_one"] += 1
            if parse_int(row[index["asof_pitcher_pitchmix_n"]]) != pitcher_n:
                invariant_violations["pitchmix_n_mismatch"] += 1

            feature_text = "\x1f".join(row[1:target_pos])
            feature_hash = int.from_bytes(hashlib.blake2b(feature_text.encode("utf-8"), digest_size=8).digest(), "big")
            target_bit = 1 if target == 0 else 2
            previous = feature_hashes.get(feature_hash)
            if previous is None:
                feature_hashes[feature_hash] = (1 << 2) | target_bit
            else:
                feature_hashes[feature_hash] = (((previous >> 2) + 1) << 2) | ((previous & 3) | target_bit)

            context_key = (
                season,
                int(values["game_month"] or 0),
                int(values["game_dayofweek"] or 0),
                inning,
                row[index["top_bottom"]],
                balls,
                strikes,
                outs,
                row[index["pitcher_hand"]],
                row[index["batter_hand"]],
            )
            context_keys[context_key] += 1

            if row_count % 250000 == 0:
                print(f"train: {row_count:,} rows", flush=True)

    duplicate_groups = duplicate_rows = duplicate_excess = conflicting_groups = conflicting_rows = 0
    for state in feature_hashes.values():
        count = state >> 2
        mask = state & 3
        if count > 1:
            duplicate_groups += 1
            duplicate_rows += count
            duplicate_excess += count - 1
            if mask == 3:
                conflicting_groups += 1
                conflicting_rows += count

    overall_rate = target_sum / row_count
    serialized_groups = serialize_target_groups(groups)
    numeric_summary = {
        column: summarize_numeric(stats[column], samples[column], row_count) for column in numeric_columns
    }
    missing_summary = []
    for column in header:
        if column in {"row_id", "control_success"}:
            continue
        missing = missing_counts[column]
        missing_summary.append(
            {
                "column": column,
                "missing": missing,
                "missing_rate": missing / row_count,
                "target_rate_when_missing": missing_target_sum[column] / missing if missing else None,
                "target_rate_when_present": (target_sum - missing_target_sum[column]) / (row_count - missing) if row_count > missing else None,
            }
        )
    missing_summary.sort(key=lambda item: item["missing_rate"], reverse=True)

    numeric_correlations = [
        {"feature": column, "correlation": numeric_summary[column]["target_correlation"], "count": numeric_summary[column]["count"]}
        for column in numeric_columns
        if numeric_summary[column]["target_correlation"] is not None
    ]
    numeric_correlations.sort(key=lambda item: abs(item["correlation"]), reverse=True)

    result = {
        "file": str(TRAIN_PATH.relative_to(ROOT)),
        "file_size_bytes": TRAIN_PATH.stat().st_size,
        "rows": row_count,
        "columns": len(header),
        "header": header,
        "target_sum": target_sum,
        "target_rate": overall_rate,
        "first_row_by_season": first_row_by_season,
        "last_row_by_season": last_row_by_season,
        "season_transitions": season_transitions,
        "numeric_summary": numeric_summary,
        "missingness": missing_summary,
        "categorical_cardinality": {column: len(counter) for column, counter in categorical_counts.items()},
        "categorical_values": {
            column: [{"value": key, "count": count} for key, count in counter.most_common()]
            for column, counter in categorical_counts.items()
        },
        "groups": serialized_groups,
        "category_effects": category_effects(serialized_groups, overall_rate),
        "calibration": serialize_calibration(calibration),
        "numeric_target_correlations": numeric_correlations,
        "asof_pairwise_correlations": pairwise_correlation(sampled_correlations, correlation_columns),
        "pitchers": entity_target_summary(pitcher_targets),
        "batters": entity_target_summary(batter_targets),
        "invariant_violations": dict(invariant_violations),
        "data_quality_notes": dict(data_quality_notes),
        "deterministic_relationships": {
            "row_id_is_strictly_sequential": invariant_violations["row_id_not_sequential"] == 0,
            "seasons_form_contiguous_blocks": season_transitions == len(first_row_by_season) - 1,
            "runner_count_equals_runner_flags": invariant_violations["runner_count_mismatch"] == 0,
            "base_state_matches_runner_flags": invariant_violations["base_state_mismatch"] == 0,
            "run_total_equals_top_plus_bottom": invariant_violations["run_total_mismatch"] == 0,
            "home_score_diff_is_derived": invariant_violations["home_score_diff_mismatch"] == 0,
            "pitcher_team_score_diff_is_derived": invariant_violations["pitcher_score_diff_mismatch"] == 0,
            "pitchmix_rates_sum_to_one_when_present": invariant_violations["pitchmix_rates_not_one"] == 0,
            "pitchmix_n_equals_pitcher_n": invariant_violations["pitchmix_n_mismatch"] == 0,
            "home_and_away_expectancy_sum_within_rounding_tolerance": invariant_violations["win_expectancy_not_100"] == 0,
        },
        "duplicates_excluding_row_id_and_target": {
            "hash_bits": 64,
            "unique_feature_hashes": len(feature_hashes),
            "duplicate_groups": duplicate_groups,
            "rows_in_duplicate_groups": duplicate_rows,
            "duplicate_excess_rows": duplicate_excess,
            "conflicting_target_groups": conflicting_groups,
            "rows_in_conflicting_groups": conflicting_rows,
        },
        "sampling": {
            "method": "every Nth row",
            "stride": TRAIN_SAMPLE_STRIDE,
            "sample_rows": len(sampled_correlations),
            "used_for": "quantiles and pairwise correlations only",
        },
    }
    return result, context_keys, pitcher_counts, batter_counts


def scan_test(train_header: list[str], train_pitchers: Counter[str], train_batters: Counter[str]) -> dict[str, Any]:
    with TEST_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        rows = list(reader)
    expected_header = [column for column in train_header if column != "control_success"]
    row_ids = [row["row_id"] for row in rows]
    pitcher_ids = [row["pitcher_id"] for row in rows]
    batter_ids = [row["batter_id"] for row in rows]
    missing = Counter()
    for row in rows:
        for column, value in row.items():
            if value == "":
                missing[column] += 1
    with SUBMISSION_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        submission_rows = list(csv.DictReader(handle))
    return {
        "file": str(TEST_PATH.relative_to(ROOT)),
        "rows": len(rows),
        "columns": len(header),
        "header": header,
        "schema_matches_train_features": header == expected_header,
        "row_ids_unique": len(set(row_ids)) == len(row_ids),
        "seasons": sorted(Counter(row["season"] for row in rows).items()),
        "missingness": dict(missing),
        "pitchers_seen_in_train": sum(pitcher_id in train_pitchers for pitcher_id in pitcher_ids),
        "batters_seen_in_train": sum(batter_id in train_batters for batter_id in batter_ids),
        "sample_submission_rows": len(submission_rows),
        "sample_submission_columns": list(submission_rows[0].keys()) if submission_rows else [],
        "sample_submission_ids_match_test_order": [row["row_id"] for row in submission_rows] == row_ids,
        "note": "Only five format-check rows are distributed; no population drift inference is valid from this sample.",
    }


def scan_trackman() -> tuple[dict[str, Any], Counter[tuple[Any, ...]]]:
    numeric_columns = [
        "season",
        "game_month",
        "game_dayofweek",
        "pitch_no",
        "inning",
        "balls_before",
        "strikes_before",
        "outs_before",
        "pitch_of_pa",
        "rel_speed",
        "spin_rate",
        "induced_vert_break",
        "horz_break",
        "extension",
        "rel_height",
        "rel_side",
        "zone_speed",
    ]
    physical_columns = [
        "rel_speed",
        "spin_rate",
        "induced_vert_break",
        "horz_break",
        "extension",
        "rel_height",
        "rel_side",
        "zone_speed",
    ]
    categorical_columns = [
        "top_bottom",
        "pitcher_hand",
        "batter_hand",
        "pitcher_team",
        "batter_team",
        "tagged_pitch_type",
        "auto_pitch_type",
        "pitch_type_group",
    ]
    stats = {column: NumericStats() for column in numeric_columns}
    samples = {column: [] for column in numeric_columns}
    group_metric_samples: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    physical_by_group: dict[str, dict[str, NumericStats]] = defaultdict(
        lambda: {column: NumericStats() for column in physical_columns}
    )
    pitch_group_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    categorical_counts = {column: Counter() for column in categorical_columns}
    pitcher_counts: Counter[str] = Counter()
    batter_counts: Counter[str] = Counter()
    game_counts: Counter[str] = Counter()
    season_pitch_group: dict[str, Counter[str]] = defaultdict(Counter)
    invariant_violations: Counter[str] = Counter()
    invariant_examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    event_hashes: set[int] = set()
    duplicate_events = 0
    context_keys: Counter[tuple[Any, ...]] = Counter()
    min_date: str | None = None
    max_date: str | None = None
    row_count = 0
    exact_tag_auto_match = 0
    tag_auto_comparable = 0

    with TRACKMAN_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        index = {name: pos for pos, name in enumerate(header)}
        num_positions = [(column, index[column]) for column in numeric_columns]
        cat_positions = [(column, index[column]) for column in categorical_columns]

        for row_count, row in enumerate(reader, start=1):
            if len(row) != len(header):
                invariant_violations["row_length"] += 1
                continue
            if row[index["trackman_id"]] != str(row_count):
                invariant_violations["trackman_id_not_sequential"] += 1
            is_sample = row_count % TRACKMAN_SAMPLE_STRIDE == 0
            values: dict[str, float | None] = {}
            for column, position in num_positions:
                value = parse_float(row[position])
                values[column] = value
                stats[column].update(value)
                if value is None:
                    missing_counts[column] += 1
                elif is_sample:
                    samples[column].append(value)
            for column, position in cat_positions:
                value = row[position]
                if value == "":
                    missing_counts[column] += 1
                else:
                    categorical_counts[column][value] += 1

            game_id = row[index["trackman_game_id"]]
            pitcher_id = row[index["pitcher_trackman_id"]]
            batter_id = row[index["batter_trackman_id"]]
            game_counts[game_id] += 1
            pitcher_counts[pitcher_id] += 1
            batter_counts[batter_id] += 1
            season = row[index["season"]]
            pitch_group = row[index["pitch_type_group"]] or "missing"
            pitch_group_counts[pitch_group] += 1
            season_pitch_group[season][pitch_group] += 1
            for column in physical_columns:
                value = values[column]
                physical_by_group[pitch_group][column].update(value)
                if is_sample and value is not None:
                    group_metric_samples[pitch_group][column].append(value)

            tagged = row[index["tagged_pitch_type"]]
            auto = row[index["auto_pitch_type"]]
            if tagged and auto:
                tag_auto_comparable += 1
                if tagged == auto:
                    exact_tag_auto_match += 1

            date_raw = row[index["game_date"]]
            date_value = None
            for date_format in ("%m/%d/%Y", "%Y-%m-%d"):
                try:
                    date_value = datetime.strptime(date_raw, date_format)
                    break
                except ValueError:
                    continue
            if date_value is not None:
                iso_date = date_value.strftime("%Y-%m-%d")
                min_date = iso_date if min_date is None or iso_date < min_date else min_date
                max_date = iso_date if max_date is None or iso_date > max_date else max_date
                if date_value.year != int(values["season"] or 0):
                    invariant_violations["date_season_mismatch"] += 1
                if date_value.month != int(values["game_month"] or 0):
                    invariant_violations["date_month_mismatch"] += 1
                if date_value.weekday() != int(values["game_dayofweek"] or 0):
                    invariant_violations["date_dayofweek_mismatch"] += 1
            else:
                invariant_violations["invalid_game_date"] += 1
                if len(invariant_examples["invalid_game_date"]) < 5:
                    invariant_examples["invalid_game_date"].append(
                        {"trackman_id": row[index["trackman_id"]], "game_date": date_raw}
                    )

            balls = int(values["balls_before"] or 0)
            strikes = int(values["strikes_before"] or 0)
            outs = int(values["outs_before"] or 0)
            if balls not in {0, 1, 2, 3}:
                invariant_violations["balls_out_of_range"] += 1
                if len(invariant_examples["balls_out_of_range"]) < 5:
                    invariant_examples["balls_out_of_range"].append(
                        {"trackman_id": row[index["trackman_id"]], "value": str(balls)}
                    )
            if strikes not in {0, 1, 2}:
                invariant_violations["strikes_out_of_range"] += 1
                if len(invariant_examples["strikes_out_of_range"]) < 5:
                    invariant_examples["strikes_out_of_range"].append(
                        {"trackman_id": row[index["trackman_id"]], "value": str(strikes)}
                    )
            if outs not in {0, 1, 2}:
                invariant_violations["outs_out_of_range"] += 1
                if len(invariant_examples["outs_out_of_range"]) < 5:
                    invariant_examples["outs_out_of_range"].append(
                        {"trackman_id": row[index["trackman_id"]], "value": str(outs)}
                    )
            for column in ["rel_speed", "spin_rate", "extension", "zone_speed"]:
                value = values[column]
                if value is not None and value < 0:
                    invariant_violations[f"{column}_negative"] += 1
                    if len(invariant_examples[f"{column}_negative"]) < 5:
                        invariant_examples[f"{column}_negative"].append(
                            {
                                "trackman_id": row[index["trackman_id"]],
                                "game_date": date_raw,
                                "pitcher_trackman_id": pitcher_id,
                                "value": str(value),
                            }
                        )

            event_text = game_id + "\x1f" + row[index["pitch_no"]]
            event_hash = int.from_bytes(hashlib.blake2b(event_text.encode("utf-8"), digest_size=8).digest(), "big")
            if event_hash in event_hashes:
                duplicate_events += 1
                if len(invariant_examples["duplicate_game_pitch_event"]) < 5:
                    invariant_examples["duplicate_game_pitch_event"].append(
                        {"trackman_game_id": game_id, "pitch_no": row[index["pitch_no"]]}
                    )
            else:
                event_hashes.add(event_hash)

            top_bottom = row[index["top_bottom"]]
            canonical_top_bottom = "T" if top_bottom == "Top" else "B" if top_bottom == "Bottom" else top_bottom
            context_key = (
                season,
                int(values["game_month"] or 0),
                int(values["game_dayofweek"] or 0),
                int(values["inning"] or 0),
                canonical_top_bottom,
                balls,
                strikes,
                outs,
                row[index["pitcher_hand"]],
                row[index["batter_hand"]],
            )
            context_keys[context_key] += 1

            if row_count % 250000 == 0:
                print(f"trackman: {row_count:,} rows", flush=True)

    numeric_summary = {
        column: summarize_numeric(stats[column], samples[column], row_count) for column in numeric_columns
    }
    physical_group_summary: dict[str, dict[str, Any]] = {}
    for group, metric_stats in physical_by_group.items():
        physical_group_summary[group] = {
            column: summarize_numeric(
                metric_stats[column], group_metric_samples[group][column], pitch_group_counts[group]
            )
            for column in physical_columns
        }
    missing_summary = [
        {"column": column, "missing": missing_counts[column], "missing_rate": missing_counts[column] / row_count}
        for column in header
        if column != "trackman_id"
    ]
    missing_summary.sort(key=lambda item: item["missing_rate"], reverse=True)

    result = {
        "file": str(TRACKMAN_PATH.relative_to(ROOT)),
        "file_size_bytes": TRACKMAN_PATH.stat().st_size,
        "rows": row_count,
        "columns": len(header),
        "header": header,
        "date_min": min_date,
        "date_max": max_date,
        "numeric_summary": numeric_summary,
        "missingness": missing_summary,
        "categorical_cardinality": {column: len(counter) for column, counter in categorical_counts.items()},
        "categorical_values": {
            column: [{"value": key, "count": count} for key, count in counter.most_common()]
            for column, counter in categorical_counts.items()
        },
        "games": frequency_summary(game_counts),
        "pitchers": frequency_summary(pitcher_counts),
        "batters": frequency_summary(batter_counts),
        "season_pitch_type_group": {
            season: dict(counter) for season, counter in sorted(season_pitch_group.items())
        },
        "physical_by_pitch_group": physical_group_summary,
        "tagged_auto_exact_match": {
            "comparable_rows": tag_auto_comparable,
            "exact_matches": exact_tag_auto_match,
            "exact_match_rate": exact_tag_auto_match / tag_auto_comparable if tag_auto_comparable else None,
        },
        "duplicate_game_pitch_events": duplicate_events,
        "invariant_violations": dict(invariant_violations),
        "invariant_examples": dict(invariant_examples),
        "sampling": {
            "method": "every Nth row",
            "stride": TRACKMAN_SAMPLE_STRIDE,
            "sample_rows": len(samples["season"]),
            "used_for": "quantiles only",
        },
    }
    return result, context_keys


def candidate_quantile(candidate_counts: Counter[int], q: float) -> float | None:
    total = sum(candidate_counts.values())
    if total == 0:
        return None
    threshold = total * q
    cumulative = 0
    for candidates, weight in sorted(candidate_counts.items()):
        cumulative += weight
        if cumulative >= threshold:
            return float(candidates)
    return float(max(candidate_counts))


def context_overlap(
    train_context: Counter[tuple[Any, ...]], track_context: Counter[tuple[Any, ...]]
) -> dict[str, Any]:
    mapping_options = {
        "1=Right,2=Left": {"1": "Right", "2": "Left"},
        "1=Left,2=Right": {"1": "Left", "2": "Right"},
    }
    option_results = []
    for option, mapping in mapping_options.items():
        total_rows = 0
        matched_rows = 0
        unique_candidate_rows = 0
        candidate_weight: Counter[int] = Counter()
        matched_keys = 0
        transformed_keys = 0
        for key, train_count in train_context.items():
            transformed = (*key[:-2], mapping.get(str(key[-2]), str(key[-2])), mapping.get(str(key[-1]), str(key[-1])))
            candidates = track_context.get(transformed, 0)
            transformed_keys += 1
            total_rows += train_count
            candidate_weight[candidates] += train_count
            if candidates:
                matched_rows += train_count
                matched_keys += 1
            if candidates == 1:
                unique_candidate_rows += train_count
        option_results.append(
            {
                "hand_mapping": option,
                "train_context_keys": transformed_keys,
                "matched_context_keys": matched_keys,
                "train_rows": total_rows,
                "rows_with_at_least_one_candidate": matched_rows,
                "row_coverage": matched_rows / total_rows if total_rows else None,
                "rows_with_exactly_one_candidate": unique_candidate_rows,
                "unique_candidate_rate": unique_candidate_rows / total_rows if total_rows else None,
                "candidate_count_p50": candidate_quantile(candidate_weight, 0.50),
                "candidate_count_p90": candidate_quantile(candidate_weight, 0.90),
                "candidate_count_p99": candidate_quantile(candidate_weight, 0.99),
                "candidate_count_max": max(candidate_weight) if candidate_weight else None,
            }
        )
    option_results.sort(key=lambda item: item["row_coverage"], reverse=True)
    return {
        "key_definition": [
            "season",
            "game_month",
            "game_dayofweek",
            "inning",
            "top_bottom",
            "balls_before",
            "strikes_before",
            "outs_before",
            "pitcher_hand",
            "batter_hand",
        ],
        "options": option_results,
        "interpretation": "This coarse shared key measures feasibility only. It is not safe for a direct row-level join when candidate counts exceed one.",
    }


def svg_document(width: int, height: int, content: str, title: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#FFFFFF"/>
<style>
text {{ font-family: "Apple SD Gothic Neo", "Noto Sans KR", Arial, sans-serif; fill: #172033; }}
.title {{ font-size: 22px; font-weight: 700; }}
.subtitle {{ font-size: 12px; fill: #667085; }}
.axis {{ font-size: 11px; fill: #667085; }}
.label {{ font-size: 12px; }}
.value {{ font-size: 11px; font-weight: 600; }}
</style>
<text x="32" y="36" class="title">{html.escape(title)}</text>
{content}
</svg>'''


def save_svg(name: str, svg: str) -> None:
    (FIGURE_DIR / name).write_text(svg, encoding="utf-8")


def horizontal_bar_svg(
    rows: list[tuple[str, float]], title: str, value_format: str = "number", color: str = "#2563EB"
) -> str:
    width = 920
    row_height = 30
    height = 90 + row_height * len(rows)
    left = 260
    right = 80
    chart_width = width - left - right
    maximum = max((abs(value) for _, value in rows), default=1.0) or 1.0
    parts = []
    for index, (label, value) in enumerate(rows):
        y = 66 + index * row_height
        bar_width = abs(value) / maximum * chart_width
        parts.append(f'<text x="{left - 10}" y="{y + 16}" text-anchor="end" class="label">{html.escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="20" rx="3" fill="{color}" opacity="0.88"/>')
        if value_format == "percent":
            rendered = f"{value * 100:.2f}%"
        elif value_format == "correlation":
            rendered = f"{value:+.4f}"
        else:
            rendered = f"{value:,.0f}"
        parts.append(f'<text x="{left + bar_width + 8:.2f}" y="{y + 15}" class="value">{rendered}</text>')
    return svg_document(width, height, "\n".join(parts), title)


def line_svg(rows: list[tuple[str, float]], title: str, percent: bool = True) -> str:
    width, height = 920, 460
    left, right, top, bottom = 90, 40, 70, 70
    chart_w, chart_h = width - left - right, height - top - bottom
    values = [value for _, value in rows]
    min_value, max_value = min(values), max(values)
    padding = max((max_value - min_value) * 0.2, 0.002 if percent else 1.0)
    y_min = min_value - padding
    y_max = max_value + padding
    parts = []
    for step in range(6):
        y = top + chart_h * step / 5
        value = y_max - (y_max - y_min) * step / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#E6EAF0"/>')
        label = f"{value * 100:.2f}%" if percent else f"{value:,.0f}"
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="axis">{label}</text>')
    points = []
    for index, (label, value) in enumerate(rows):
        x = left + chart_w * index / max(1, len(rows) - 1)
        y = top + chart_h * (y_max - value) / (y_max - y_min)
        points.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#2563EB"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + chart_h + 28}" text-anchor="middle" class="axis">{html.escape(label)}</text>')
        value_label = f"{value * 100:.2f}%" if percent else f"{value:,.0f}"
        parts.append(f'<text x="{x:.1f}" y="{y - 12:.1f}" text-anchor="middle" class="value">{value_label}</text>')
    parts.insert(6, f'<polyline points="{" ".join(points)}" fill="none" stroke="#2563EB" stroke-width="3"/>')
    return svg_document(width, height, "\n".join(parts), title)


def count_heatmap_svg(rows: list[dict[str, Any]], title: str) -> str:
    mapping = {row["key"]: row for row in rows}
    width, height = 740, 500
    left, top = 130, 85
    cell_w, cell_h = 135, 95
    rates = [row["target_rate"] for row in rows]
    low, high = min(rates), max(rates)
    parts = []
    for strike in range(3):
        parts.append(f'<text x="{left - 18}" y="{top + strike * cell_h + 54}" text-anchor="end" class="label">{strike} 스트라이크</text>')
        for ball in range(4):
            key = f"{ball}-{strike}"
            row = mapping.get(key)
            x, y = left + ball * cell_w, top + strike * cell_h
            if row is None:
                color = "#F2F4F7"
                text = "N/A"
                count = ""
            else:
                ratio = (row["target_rate"] - low) / (high - low) if high > low else 0.5
                red = int(232 - 120 * ratio)
                green = int(244 - 40 * ratio)
                blue = int(252 - 5 * ratio)
                color = f"#{red:02X}{green:02X}{blue:02X}"
                text = f"{row['target_rate'] * 100:.2f}%"
                count = f"n={row['count']:,}"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 8}" height="{cell_h - 8}" rx="8" fill="{color}"/>')
            parts.append(f'<text x="{x + (cell_w - 8) / 2:.1f}" y="{y + 39}" text-anchor="middle" class="value">{text}</text>')
            parts.append(f'<text x="{x + (cell_w - 8) / 2:.1f}" y="{y + 61}" text-anchor="middle" class="axis">{count}</text>')
    for ball in range(4):
        parts.append(f'<text x="{left + ball * cell_w + (cell_w - 8) / 2:.1f}" y="{top - 16}" text-anchor="middle" class="label">{ball} 볼</text>')
    return svg_document(width, height, "\n".join(parts), title)


def calibration_svg(rows: list[dict[str, Any]], title: str) -> str:
    usable = [row for row in rows if row["mean_feature"] is not None]
    usable.sort(key=lambda row: row["mean_feature"])
    width, height = 700, 590
    left, top, size = 90, 75, 430
    plotted_values = [0.5]
    plotted_values.extend(row["mean_feature"] for row in usable)
    plotted_values.extend(row["target_rate"] for row in usable)
    raw_min, raw_max = min(plotted_values), max(plotted_values)
    padding = max(0.03, (raw_max - raw_min) * 0.12)
    axis_min = max(0.0, raw_min - padding)
    axis_max = min(1.0, raw_max + padding)
    if axis_max - axis_min < 0.2:
        midpoint = (axis_min + axis_max) / 2
        axis_min = max(0.0, midpoint - 0.1)
        axis_max = min(1.0, midpoint + 0.1)
    parts = []
    for step in range(6):
        value = axis_min + step * (axis_max - axis_min) / 5
        x = left + (value - axis_min) / (axis_max - axis_min) * size
        y = top + size - (value - axis_min) / (axis_max - axis_min) * size
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + size}" stroke="#EEF1F5"/>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + size}" y2="{y:.1f}" stroke="#EEF1F5"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + size + 24}" text-anchor="middle" class="axis">{value:.2f}</text>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="axis">{value:.2f}</text>')
    parts.append(f'<line x1="{left}" y1="{top + size}" x2="{left + size}" y2="{top}" stroke="#98A2B3" stroke-dasharray="6 5"/>')
    points = []
    for row in usable:
        x_value = row["mean_feature"]
        y_value = row["target_rate"]
        x = left + (x_value - axis_min) / (axis_max - axis_min) * size
        y = top + size - (y_value - axis_min) / (axis_max - axis_min) * size
        points.append(f"{x:.1f},{y:.1f}")
        radius = min(13, max(4, math.sqrt(row["count"]) / 80))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="#2563EB" opacity="0.85"/>')
    parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#2563EB" stroke-width="2"/>')
    parts.append(f'<text x="{left + size / 2}" y="{top + size + 55}" text-anchor="middle" class="label">과거 투수 성공률 평균</text>')
    parts.append(f'<text x="26" y="{top + size / 2}" text-anchor="middle" class="label" transform="rotate(-90 26 {top + size / 2})">현재 투구 성공률</text>')
    parts.append(f'<text x="{left + size + 30}" y="{top + 20}" class="subtitle">점 크기 = 표본 수</text>')
    return svg_document(width, height, "\n".join(parts), title)


def stacked_pitch_mix_svg(data: dict[str, dict[str, int]], title: str) -> str:
    years = sorted(data)
    categories = ["fastball", "breaking", "offspeed", "other", "missing"]
    colors = {"fastball": "#2563EB", "breaking": "#F79009", "offspeed": "#12B76A", "other": "#98A2B3", "missing": "#D0D5DD"}
    width, height = 920, 500
    left, top, chart_w, chart_h = 90, 85, 750, 310
    bar_w = 78
    gap = (chart_w - bar_w * len(years)) / max(1, len(years) - 1)
    parts = []
    for index, year in enumerate(years):
        total = sum(data[year].values())
        x = left + index * (bar_w + gap)
        y_cursor = top + chart_h
        for category in categories:
            count = data[year].get(category, 0)
            height_value = chart_h * count / total if total else 0
            y_cursor -= height_value
            parts.append(f'<rect x="{x:.1f}" y="{y_cursor:.1f}" width="{bar_w}" height="{height_value:.1f}" fill="{colors[category]}"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{top + chart_h + 26}" text-anchor="middle" class="label">{year}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{top + chart_h + 44}" text-anchor="middle" class="axis">n={total:,}</text>')
    for step in range(6):
        y = top + chart_h - chart_h * step / 5
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="axis">{step * 20}%</text>')
    legend_x = left
    for category in categories:
        parts.append(f'<rect x="{legend_x}" y="{height - 52}" width="14" height="14" fill="{colors[category]}"/>')
        parts.append(f'<text x="{legend_x + 20}" y="{height - 40}" class="axis">{category}</text>')
        legend_x += 135
    return svg_document(width, height, "\n".join(parts), title)


def interval_svg(data: dict[str, dict[str, Any]], metric: str, title: str, unit: str) -> str:
    groups = [group for group in ["fastball", "breaking", "offspeed", "other"] if group in data]
    rows = []
    for group in groups:
        summary = data[group][metric]
        rows.append((group, summary.get("p05"), summary.get("p50"), summary.get("p95"), summary.get("count")))
    values = [value for _, low, median, high, _ in rows for value in (low, median, high) if value is not None]
    low_bound, high_bound = min(values), max(values)
    width, height = 860, 120 + len(rows) * 76
    left, right, top = 170, 70, 80
    chart_w = width - left - right
    parts = []
    for step in range(6):
        value = low_bound + (high_bound - low_bound) * step / 5
        x = left + chart_w * step / 5
        parts.append(f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{height - 55}" stroke="#EEF1F5"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 32}" text-anchor="middle" class="axis">{value:,.1f}</text>')
    for index, (group, low, median, high, count) in enumerate(rows):
        if low is None or median is None or high is None:
            continue
        y = top + index * 76
        x1 = left + (low - low_bound) / (high_bound - low_bound) * chart_w
        xm = left + (median - low_bound) / (high_bound - low_bound) * chart_w
        x2 = left + (high - low_bound) / (high_bound - low_bound) * chart_w
        parts.append(f'<text x="{left - 16}" y="{y + 5}" text-anchor="end" class="label">{group}</text>')
        parts.append(f'<line x1="{x1:.1f}" y1="{y}" x2="{x2:.1f}" y2="{y}" stroke="#98A2B3" stroke-width="8" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{xm:.1f}" cy="{y}" r="8" fill="#2563EB"/>')
        parts.append(f'<text x="{xm:.1f}" y="{y - 15}" text-anchor="middle" class="value">{median:,.1f}</text>')
        parts.append(f'<text x="{left - 16}" y="{y + 23}" text-anchor="end" class="axis">n={count:,}</text>')
    parts.append(f'<text x="{left + chart_w / 2}" y="{height - 6}" text-anchor="middle" class="label">{html.escape(unit)} (5–95 percentile)</text>')
    return svg_document(width, height, "\n".join(parts), title)


def make_figures(summary: dict[str, Any]) -> list[str]:
    figures: list[str] = []
    train = summary["train"]
    trackman = summary["trackman"]

    season_rows = sorted(train["groups"]["season"], key=lambda row: int(row["key"]))
    save_svg(
        "train_rows_by_season.svg",
        horizontal_bar_svg([(row["key"], float(row["count"])) for row in season_rows], "시즌별 학습 행 수"),
    )
    figures.append("train_rows_by_season.svg")
    save_svg(
        "target_rate_by_season.svg",
        line_svg([(row["key"], row["target_rate"]) for row in season_rows], "시즌별 제구 성공률"),
    )
    figures.append("target_rate_by_season.svg")
    season_game_type_rows = sorted(train["groups"]["season_game_type"], key=lambda row: row["key"])
    save_svg(
        "target_rate_by_season_game_type.svg",
        horizontal_bar_svg(
            [(row["key"], row["target_rate"]) for row in season_game_type_rows],
            "시즌·경기 유형별 제구 성공률",
            value_format="percent",
            color="#0BA5EC",
        ),
    )
    figures.append("target_rate_by_season_game_type.svg")
    save_svg("target_rate_by_count.svg", count_heatmap_svg(train["groups"]["count"], "볼-스트라이크 카운트별 제구 성공률"))
    figures.append("target_rate_by_count.svg")

    missing_rows = [row for row in train["missingness"] if row["missing_rate"] > 0][:15]
    save_svg(
        "train_missingness.svg",
        horizontal_bar_svg(
            [(row["column"], row["missing_rate"]) for row in missing_rows],
            "학습 데이터 결측률 상위 컬럼",
            value_format="percent",
            color="#F79009",
        ),
    )
    figures.append("train_missingness.svg")

    correlations = train["numeric_target_correlations"][:15]
    save_svg(
        "numeric_target_correlations.svg",
        horizontal_bar_svg(
            [(row["feature"], row["correlation"]) for row in correlations],
            "타깃과의 Pearson 상관계수 상위 수치 피처 (막대 길이=절대값)",
            value_format="correlation",
            color="#12B76A",
        ),
    )
    figures.append("numeric_target_correlations.svg")

    save_svg(
        "pitcher_prior_calibration.svg",
        calibration_svg(train["calibration"]["asof_pitcher_success_rate"], "투수 과거 성공률과 현재 투구 성공률"),
    )
    figures.append("pitcher_prior_calibration.svg")

    base_order = ["___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"]
    base_map = {row["key"]: row for row in train["groups"]["base_state"]}
    save_svg(
        "target_rate_by_base_state.svg",
        horizontal_bar_svg(
            [(state, base_map[state]["target_rate"]) for state in base_order if state in base_map],
            "주자 상황별 제구 성공률",
            value_format="percent",
            color="#7F56D9",
        ),
    )
    figures.append("target_rate_by_base_state.svg")

    save_svg(
        "trackman_pitch_mix_by_season.svg",
        stacked_pitch_mix_svg(trackman["season_pitch_type_group"], "시즌별 Trackman 구종군 구성"),
    )
    figures.append("trackman_pitch_mix_by_season.svg")

    track_missing = [row for row in trackman["missingness"] if row["missing_rate"] > 0][:15]
    save_svg(
        "trackman_missingness.svg",
        horizontal_bar_svg(
            [(row["column"], row["missing_rate"]) for row in track_missing],
            "Trackman 결측률 상위 컬럼",
            value_format="percent",
            color="#F04438",
        ),
    )
    figures.append("trackman_missingness.svg")

    save_svg(
        "trackman_speed_by_group.svg",
        interval_svg(trackman["physical_by_pitch_group"], "rel_speed", "구종군별 릴리스 구속 분포", "rel_speed"),
    )
    figures.append("trackman_speed_by_group.svg")
    return figures


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Scanning train.csv...", flush=True)
    train, train_context, train_pitchers, train_batters = scan_train()
    print("Scanning test.csv and sample_submission.csv...", flush=True)
    test = scan_test(train["header"], train_pitchers, train_batters)
    print("Scanning trackman_history.csv...", flush=True)
    trackman, track_context = scan_trackman()
    print("Calculating cross-dataset context overlap...", flush=True)
    overlap = context_overlap(train_context, track_context)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": os.sys.version,
        "method": {
            "exact": "row counts, missingness, group counts/rates, numeric moments, target correlations, integrity checks",
            "approximate": "quantiles and pairwise input-feature correlations use deterministic systematic samples",
            "external_dependencies": "none",
        },
        "train": train,
        "test_sample": test,
        "trackman": trackman,
        "cross_dataset_context_overlap": overlap,
    }
    summary_path = RESULT_DIR / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    figures = make_figures(summary)
    print(f"Wrote {summary_path.relative_to(ROOT)}", flush=True)
    print(f"Wrote {len(figures)} figures to {FIGURE_DIR.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
