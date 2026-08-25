import pandas as pd
import numpy as np
import pytest

from enr_risk_premium.benchmark import (
    build_annual_scenario_metrics,
    build_hourly_index,
    build_hourly_index,
    catalog_joint_blocks,
    compute_annual_metrics,
    compute_scenario_margin,
    draw_joint_block_positions,
    lower_tail_cvar,
    margin_risk_criterion,
    normalize_prm,
    recenter_spot_monthly,
    risk_neutral_price,
    solve_adjusted_price,
    build_group_capacity_profiles,
    price_scenario_metrics,
)


def test_2027_has_8760_unique_utc_hours():
    """2027 doit contenir exactement 8 760 heures UTC."""

    hours = build_hourly_index(2027)

    assert len(hours) == 8760
    assert hours.is_unique
    assert hours.is_monotonic_increasing
    assert str(hours.tz) == "UTC"

    assert hours[0] == pd.Timestamp(
        "2027-01-01 00:00:00",
        tz="UTC",
    )

    assert hours[-1] == pd.Timestamp(
        "2027-12-31 23:00:00",
        tz="UTC",
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1496092570620, "01496092570620"),
        (1496092570620.0, "01496092570620"),
        (" 01496092570620 ", "01496092570620"),
        ("50019681610016", "50019681610016"),
        ("249619EP2", "249619EP2"),
    ],
)
def test_normalize_prm(value, expected):
    """Les PRM valides doivent être normalisés sans information perdue."""

    assert normalize_prm(value) == expected


@pytest.mark.parametrize(
    "value",
    [None, pd.NA, float("nan"), "", "?", "nan"],
)
def test_normalize_missing_prm(value):
    """Les marqueurs d'absence doivent devenir des valeurs manquantes."""

    assert pd.isna(normalize_prm(value))

def test_lower_tail_cvar_uses_worst_values():
    """La CVaR basse doit moyenner les plus petites valeurs."""

    values = [-4.0, -3.0, -2.0, -1.0]

    result = lower_tail_cvar(
        values,
        alpha=0.50,
    )

    assert result == pytest.approx(-3.50)


def test_lower_tail_cvar_handles_fractional_boundary():
    """La frontière de la queue peut contenir une fraction d'observation."""

    values = [0.0, 10.0, 20.0]

    result = lower_tail_cvar(
        values,
        alpha=0.50,
    )

    assert result == pytest.approx(10 / 3)


def test_risk_neutral_price_cancels_mean_margin():
    """Le prix neutre doit annuler la marge moyenne."""

    production = [10.0, 20.0, 30.0, 40.0]
    revenue = [400.0, 1000.0, 1800.0, 2800.0]

    result = risk_neutral_price(
        production_mwh=production,
        revenue_eur=revenue,
    )

    assert result == pytest.approx(60.0)


def test_zero_risk_aversion_recovers_neutral_price():
    """Sans aversion au risque, le prix ajusté est le prix neutre."""

    production = [10.0, 20.0, 30.0, 40.0]
    revenue = [400.0, 1000.0, 1800.0, 2800.0]

    adjusted = solve_adjusted_price(
        production_mwh=production,
        revenue_eur=revenue,
        alpha=0.50,
        risk_aversion=0.0,
    )

    assert adjusted == pytest.approx(
        60.0,
        abs=1e-8,
    )


def test_adjusted_price_has_zero_risk_criterion():
    """Le prix ajusté doit annuler exactement le critère."""

    production = [10.0, 20.0, 30.0, 40.0]
    revenue = [400.0, 1000.0, 1800.0, 2800.0]

    adjusted = solve_adjusted_price(
        production_mwh=production,
        revenue_eur=revenue,
        alpha=0.50,
        risk_aversion=0.50,
    )

    residual = margin_risk_criterion(
        price_eur_mwh=adjusted,
        production_mwh=production,
        revenue_eur=revenue,
        alpha=0.50,
        risk_aversion=0.50,
    )

    assert adjusted == pytest.approx(55.0)
    assert adjusted < 60.0
    assert abs(residual) < 1e-8

def test_collective_recentering_preserves_dispersion():
    """Le recalage collectif déplace le niveau sans réduire la dispersion."""

    hours = pd.date_range(
        "2027-01-01",
        periods=4,
        freq="h",
        tz="UTC",
    )

    spot = np.array(
        [
            [10.0, 20.0, 30.0, 40.0],
            [20.0, 30.0, 50.0, 60.0],
        ]
    )

    forwards = pd.Series(
        {1: 50.0}
    )

    centered, audit = recenter_spot_monthly(
        spot_scenarios=spot,
        hours=hours,
        monthly_forwards=forwards,
        method="collective",
    )

    assert centered.mean() == pytest.approx(50.0)

    assert centered.mean(axis=1).std() == pytest.approx(
        spot.mean(axis=1).std()
    )

    assert audit.loc[
        0,
        "collective_mean_after",
    ] == pytest.approx(50.0)


def test_individual_recentering_removes_monthly_dispersion():
    """Le recalage individuel impose la même moyenne à chaque scénario."""

    hours = pd.date_range(
        "2027-01-01",
        periods=4,
        freq="h",
        tz="UTC",
    )

    spot = np.array(
        [
            [10.0, 20.0, 30.0, 40.0],
            [20.0, 30.0, 50.0, 60.0],
        ]
    )

    centered, audit = recenter_spot_monthly(
        spot_scenarios=spot,
        hours=hours,
        monthly_forwards=pd.Series({1: 50.0}),
        method="individual",
    )

    assert centered.mean(axis=1) == pytest.approx(
        [50.0, 50.0]
    )

    assert audit.loc[
        0,
        "dispersion_after",
    ] == pytest.approx(0.0)


def test_compute_annual_metrics():
    """Les volumes et revenus doivent être calculés heure par heure."""

    load_factors = np.array(
        [
            [0.5, 1.0],
            [0.0, 0.5],
        ]
    )

    spot = np.array(
        [
            [50.0, 100.0],
            [20.0, 40.0],
        ]
    )

    production, revenue = compute_annual_metrics(
        load_factor_scenarios=load_factors,
        spot_scenarios=spot,
        capacity_mw=2.0,
    )

    assert production == pytest.approx(
        [3.0, 1.0]
    )

    assert revenue == pytest.approx(
        [250.0, 40.0]
    )


def test_compute_scenario_margin():
    """La marge doit être égale au revenu moins le paiement fixe."""

    production = [3.0, 1.0]
    revenue = [250.0, 40.0]

    margin = compute_scenario_margin(
        production_mwh=production,
        revenue_eur=revenue,
        fixed_price_eur_mwh=50.0,
    )

    assert margin == pytest.approx(
        [100.0, -10.0]
    )

def test_catalog_excludes_incomplete_blocks():
    """Un bloc contenant une valeur manquante doit être rejeté."""

    hours = pd.date_range(
        "2024-01-01",
        periods=48,
        freq="h",
        tz="UTC",
    )

    panel = pd.DataFrame(
        {
            "spot": np.arange(48, dtype=float),
            "load_factor": np.arange(
                48,
                dtype=float,
            ),
        },
        index=hours,
    )

    panel.loc[
        pd.Timestamp(
            "2024-01-02 01:00:00",
            tz="UTC",
        ),
        "spot",
    ] = np.nan

    catalog = catalog_joint_blocks(
        panel,
        block_hours=4,
    )

    assert len(catalog) == 1

    assert catalog.iloc[
        0
    ]["source_start"] == pd.Timestamp(
        "2024-01-01 00:00:00",
        tz="UTC",
    )


def test_joint_draw_is_reproducible_and_synchronised():
    """Le même tirage doit être utilisé pour toutes les colonnes."""

    source_hours = pd.date_range(
        "2024-01-01",
        periods=48,
        freq="h",
        tz="UTC",
    )

    spot = np.arange(
        48,
        dtype=float,
    )

    panel = pd.DataFrame(
        {
            "spot": spot,
            "load_factor": spot + 1000.0,
        },
        index=source_hours,
    )

    catalog = catalog_joint_blocks(
        panel,
        block_hours=4,
    )

    target_hours = pd.date_range(
        "2027-01-01",
        periods=8,
        freq="h",
        tz="UTC",
    )

    positions_1, draws_1 = (
        draw_joint_block_positions(
            candidate_blocks=catalog,
            target_hours=target_hours,
            n_scenarios=2,
            block_hours=4,
            random_seed=123,
        )
    )

    positions_2, draws_2 = (
        draw_joint_block_positions(
            candidate_blocks=catalog,
            target_hours=target_hours,
            n_scenarios=2,
            block_hours=4,
            random_seed=123,
        )
    )

    assert np.array_equal(
        positions_1,
        positions_2,
    )

    pd.testing.assert_frame_equal(
        draws_1,
        draws_2,
    )

    simulated_spot = (
        panel["spot"]
        .to_numpy()[positions_1]
    )

    simulated_load_factor = (
        panel["load_factor"]
        .to_numpy()[positions_1]
    )

    assert simulated_load_factor == pytest.approx(
        simulated_spot + 1000.0
    )

    assert positions_1.shape == (
        2,
        8,
    )

    assert len(draws_1) == 4


def test_seasonal_fallback_when_month_is_unavailable():
    """Un mois absent doit utiliser les autres mois de sa saison."""

    march_hours = pd.date_range(
        "2024-03-01",
        periods=8,
        freq="h",
        tz="UTC",
    )

    may_hours = pd.date_range(
        "2024-05-01",
        periods=8,
        freq="h",
        tz="UTC",
    )

    source_hours = march_hours.append(
        may_hours
    )

    panel = pd.DataFrame(
        {
            "spot": np.arange(
                len(source_hours),
                dtype=float,
            ),
            "load_factor": np.full(
                len(source_hours),
                0.5,
            ),
        },
        index=source_hours,
    )

    catalog = catalog_joint_blocks(
        panel,
        block_hours=4,
    )

    target_hours = pd.date_range(
        "2027-04-01",
        periods=8,
        freq="h",
        tz="UTC",
    )

    positions, draws = draw_joint_block_positions(
        candidate_blocks=catalog,
        target_hours=target_hours,
        n_scenarios=1,
        block_hours=4,
        random_seed=123,
    )

    assert positions.shape == (
        1,
        8,
    )

    assert set(
        draws["source_month"]
    ).issubset({3, 5})

    assert draws[
        "selection_rule"
    ].eq("season_fallback").all()

def test_build_group_capacity_profiles():
    """Les univers figé et dynamique doivent suivre les contrats."""

    target_hours = pd.date_range(
        "2027-01-01",
        periods=6,
        freq="h",
        tz="UTC",
    )

    contracts = pd.DataFrame(
        {
            "plant_id": [
                "1496092570620",
                "hydro-B",
                "biogas-C",
            ],
            "capacity_mw": [
                2.0,
                1.0,
                0.5,
            ],
            "contract_start": [
                "2026-01-01",
                "2027-01-01",
                "2026-01-01",
            ],
            "contract_end_exclusive": [
                "2028-01-01",
                "2027-01-01 03:00:00",
                "2028-01-01",
            ],
            "date_issue": [
                False,
                False,
                False,
            ],
            "capacity_issue": [
                False,
                False,
                False,
            ],
        }
    )

    membership = pd.DataFrame(
        {
            "plant_id": [
                "01496092570620",
                "hydro-B",
                "biogas-C",
            ],
            "group_id": [
                "solar__76",
                "hydro__84",
                "biogas__76",
            ],
            "in_all": [
                True,
                True,
                True,
            ],
            "in_weather_mvp": [
                True,
                False,
                False,
            ],
        }
    )

    group_ids = [
        "solar__76",
        "hydro__84",
        "biogas__76",
    ]

    fixed, dynamic = (
        build_group_capacity_profiles(
            contracts=contracts,
            universe_membership=membership,
            target_hours=target_hours,
            universe_column="in_all",
            group_ids=group_ids,
        )
    )

    assert fixed.sum(axis=1).to_numpy() == pytest.approx(
        [3.5] * 6
    )

    assert dynamic.sum(axis=1).to_numpy() == pytest.approx(
        [
            3.5,
            3.5,
            3.5,
            2.5,
            2.5,
            2.5,
        ]
    )

    assert fixed[
        "solar__76"
    ].to_numpy() == pytest.approx(
        [2.0] * 6
    )

    mvp_fixed, mvp_dynamic = (
        build_group_capacity_profiles(
            contracts=contracts,
            universe_membership=membership,
            target_hours=target_hours,
            universe_column="in_weather_mvp",
            group_ids=group_ids,
        )
    )

    assert mvp_fixed.sum(axis=1).to_numpy() == pytest.approx(
        [2.0] * 6
    )

    assert mvp_dynamic.sum(axis=1).to_numpy() == pytest.approx(
        [2.0] * 6
    )

def test_price_scenario_metrics():
    """Le pricing doit produire une ligne par aversion au risque."""

    annual_metrics = pd.DataFrame(
        {
            "scenario_id": [0, 1, 2, 3],
            "universe": ["all"] * 4,
            "portfolio": ["fixed"] * 4,
            "level": ["portfolio_global"] * 4,
            "entity_id": ["portfolio_total"] * 4,
            "technology": [np.nan] * 4,
            "region_code": [np.nan] * 4,
            "region_name": [np.nan] * 4,
            "production_mwh": [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
            "spot_revenue_eur": [
                400.0,
                1000.0,
                1800.0,
                2800.0,
            ],
        }
    )

    result = price_scenario_metrics(
        annual_metrics=annual_metrics,
        alpha=0.50,
        risk_aversions=(0.0, 0.50),
    )

    assert result.shape == (2, 17)

    assert result[
        "risk_neutral_price_eur_mwh"
    ].to_numpy() == pytest.approx(
        [60.0, 60.0]
    )

    no_risk = result.loc[
        result["risk_aversion"].eq(0.0)
    ].iloc[0]

    assert no_risk[
        "risk_adjusted_price_eur_mwh"
    ] == pytest.approx(60.0)

    assert no_risk[
        "risk_premium_eur_mwh"
    ] == pytest.approx(0.0)

    assert result[
        "criterion_residual_eur"
    ].abs().max() < 1e-8

def test_build_annual_scenario_metrics():
    """Les résultats locaux doivent être agrégés."""

    panel = pd.DataFrame(
        {
            "solar__53": [0.2, 0.4],
            "solar__76": [0.5, 1.0],
        }
    )

    membership = pd.DataFrame(
        {
            "group_id": ["solar__53", "solar__76"],
            "technology": ["solar", "solar"],
            "region_code": ["53", "76"],
            "region_name": ["Bretagne", "Occitanie"],
            "in_all": [True, True],
        }
    )

    fixed = pd.DataFrame(
        {
            "solar__53": [1.0, 1.0],
            "solar__76": [2.0, 2.0],
        }
    )

    dynamic = pd.DataFrame(
        {
            "solar__53": [1.0, 1.0],
            "solar__76": [2.0, 0.0],
        }
    )

    result = build_annual_scenario_metrics(
        joint_panel=panel,
        source_positions=np.array(
            [[0, 1]],
            dtype=np.int32,
        ),
        spot_scenarios=np.array(
            [[10.0, 20.0]]
        ),
        capacity_profiles={
            ("all", "fixed"): fixed,
            ("all", "dynamic"): dynamic,
        },
        universe_membership=membership,
        universe_columns={
            "all": "in_all",
        },
    )

    global_results = (
        result.loc[
            result["level"].eq(
                "portfolio_global"
            )
        ]
        .set_index("portfolio")
    )

    assert result.shape == (8, 10)

    assert global_results.loc[
        "fixed",
        "production_mwh",
    ] == pytest.approx(3.6)

    assert global_results.loc[
        "fixed",
        "spot_revenue_eur",
    ] == pytest.approx(60.0)

    assert global_results.loc[
        "dynamic",
        "production_mwh",
    ] == pytest.approx(1.6)