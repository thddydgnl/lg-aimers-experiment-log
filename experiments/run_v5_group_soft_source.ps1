$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:V2_BOOSTER_DEVICE = "gpu"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$runner = Join-Path $PSScriptRoot "run_v2_rolling.py"
$common = @(
    "--models", "catboost_group_soft",
    "--features", "base", "e14", "platoon", "hand_matchup",
        "e14_hand_cells", "e14_count_cells", "e14_type_count_cells",
        "trackman_rich", "batter_e14", "batter_middle_e14",
    "--validation-seasons", "2020", "2021",
    "--fit-game-types", "R",
    "--inner-validation", "none",
    "--e14-k", "50",
    "--k-pitcher", "50",
    "--k-platoon", "50",
    "--batter-e14-k", "80",
    "--batter-middle-k", "100",
    "--drop-features", "pitcher_id",
        "e58_fastball_rate", "e58_fastball_rel_speed_mean",
        "e58_fastball_spin_rate_mean", "e58_fastball_induced_vert_break_mean",
        "e58_fastball_horz_break_mean", "e58_fastball_extension_mean",
        "e58_fastball_rel_height_mean", "e58_fastball_rel_side_mean",
        "e58_fastball_zone_speed_mean", "e58_breaking_rate",
        "e58_breaking_rel_speed_mean", "e58_breaking_spin_rate_mean",
        "e58_breaking_induced_vert_break_mean", "e58_breaking_horz_break_mean",
        "e58_breaking_extension_mean", "e58_breaking_rel_height_mean",
        "e58_breaking_rel_side_mean", "e58_breaking_zone_speed_mean",
        "e58_offspeed_rate", "e58_offspeed_rel_speed_mean",
        "e58_offspeed_spin_rate_mean", "e58_offspeed_induced_vert_break_mean",
        "e58_offspeed_horz_break_mean", "e58_offspeed_extension_mean",
        "e58_offspeed_rel_height_mean", "e58_offspeed_rel_side_mean",
        "e58_offspeed_zone_speed_mean", "e58_other_rate",
        "e58_other_rel_speed_mean", "e58_other_spin_rate_mean",
        "e58_other_induced_vert_break_mean", "e58_other_horz_break_mean",
        "e58_other_extension_mean", "e58_other_rel_height_mean",
        "e58_other_rel_side_mean", "e58_other_zone_speed_mean",
    "--bootstrap", "50"
)

$configs = @(
    @{ Name = "alpha0"; Params = "v5_group_soft_alpha0.json" },
    @{ Name = "alpha025"; Params = "v5_group_soft_alpha025.json" },
    @{ Name = "alpha05"; Params = "v5_group_soft_alpha05.json" }
)

foreach ($config in $configs) {
    $stage = "v5_group_soft_$($config.Name)_source"
    $params = Join-Path $PSScriptRoot "params\$($config.Params)"
    & $python $runner "--stage" $stage @common "--params" $params
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
