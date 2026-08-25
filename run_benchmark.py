"""Exécuter le benchmark conjoint production-spot."""

from pathlib import Path

import numpy as np
import pandas as pd

from enr_risk_premium.benchmark import (
    build_annual_scenario_metrics,
    build_group_capacity_profiles,
    build_hourly_index,
    catalog_joint_blocks,
    draw_joint_block_positions,
    recenter_spot_monthly,
    price_scenario_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


CANONICAL_INPUT_FILES = {
    "joint_panel": (
        "joint_panel_ready_for_bootstrap.parquet"
    ),
    "universe_membership": (
        "portfolio_universe_membership_2027.parquet"
    ),
    "monthly_forwards": (
        "eex_monthly_base_2027_snapshot_2026-08-24.parquet"
    ),
    "group_quality": (
        "load_factor_group_quality.parquet"
    ),
    "contracts": "contracts_canonical.parquet",
    "imputation_summary": (
        "load_factor_imputation_summary.parquet"
    ),
}

UNIVERSE_COLUMNS = {
    "all": "in_all",
    "no_biomass_biogas": (
        "in_no_biomass_biogas"
    ),
    "weather_mvp": "in_weather_mvp",
}


def load_canonical_inputs() -> dict[str, pd.DataFrame]:
    """Charger les six entrées canoniques du benchmark."""

    tables = {}

    for table_name, file_name in (
        CANONICAL_INPUT_FILES.items()
    ):
        file_path = (
            PROCESSED_DIR
            / file_name
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Fichier canonique absent : {file_path}"
            )

        tables[table_name] = pd.read_parquet(
            file_path
        )

    return tables


def print_table_contract(
    table_name: str,
    table: pd.DataFrame,
) -> None:
    """Afficher le contrat technique d'une table."""

    print()
    print("=" * 72)
    print(table_name)
    print("=" * 72)

    print("Forme :", table.shape)
    print("Index :", type(table.index).__name__)
    print("Nom de l'index :", table.index.name)

    print(
        "Fuseau :",
        getattr(
            table.index,
            "tz",
            None,
        ),
    )

    print(
        "Début :",
        table.index.min(),
    )

    print(
        "Fin :",
        table.index.max(),
    )

    print(
        "Colonnes :",
        list(table.columns),
    )

    print("Types :")
    print(
        table.dtypes.to_string()
    )

    if table.shape[1] <= 10:
        print("Premières lignes :")
        print(
            table.head(3).to_string()
        )


def main() -> None:
    """Valider les capacités de chaque univers."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tables = load_canonical_inputs()

    hours_2027 = build_hourly_index(
        2027
    )

    group_ids = [
        column
        for column in tables[
            "joint_panel"
        ].columns
        if column != "spot_eur_mwh"
    ]

    capacity_profiles = {}
    summary_rows = []

    for universe, universe_column in (
        UNIVERSE_COLUMNS.items()
    ):
        fixed, dynamic = (
            build_group_capacity_profiles(
                contracts=tables[
                    "contracts"
                ],
                universe_membership=tables[
                    "universe_membership"
                ],
                target_hours=hours_2027,
                universe_column=universe_column,
                group_ids=group_ids,
            )
        )

        capacity_profiles[
            (universe, "fixed")
        ] = fixed

        capacity_profiles[
            (universe, "dynamic")
        ] = dynamic

        n_plants = int(
            tables[
                "universe_membership"
            ][universe_column].sum()
        )

        n_active_groups = int(
            dynamic.max(axis=0)
            .gt(0)
            .sum()
        )

        summary_rows.append(
            {
                "universe": universe,
                "n_plants": n_plants,
                "n_active_groups": (
                    n_active_groups
                ),
                "fixed_mw": float(
                    fixed.iloc[0].sum()
                ),
                "dynamic_january_1_mw": float(
                    dynamic.iloc[0].sum()
                ),
                "dynamic_december_31_mw": float(
                    dynamic.iloc[-1].sum()
                ),
            }
        )

    capacity_summary = pd.DataFrame(
        summary_rows
    )

    capacity_summary.to_csv(
        OUTPUT_DIR
        / "capacity_summary_2027.csv",
        index=False,
    )

    print()
    print("CAPACITÉS 2027 CONSTRUITES")

    print(
        capacity_summary
        .round(6)
        .to_string(index=False)
    )

    print()
    print(
        "Construction des capacités terminée."
    )
    print()
    print("CONSTRUCTION DES BLOCS CONJOINTS")

    joint_panel = tables[
        "joint_panel"
    ]

    block_catalog = catalog_joint_blocks(
        joint_panel=joint_panel,
        block_hours=168,
    )

    block_counts = (
        block_catalog
        .groupby("source_month")
        .size()
        .reindex(
            range(1, 13),
            fill_value=0,
        )
        .rename("n_candidate_blocks")
    )

    print()
    print("Blocs candidats par mois :")
    print(
        block_counts.to_string()
    )

    source_positions, bootstrap_draws = (
        draw_joint_block_positions(
            candidate_blocks=block_catalog,
            target_hours=hours_2027,
            n_scenarios=1500,
            block_hours=168,
            random_seed=20270824,
        )
    )

    print()
    print(
        "Forme des positions :",
        source_positions.shape,
    )

    print(
        "Nombre de blocs tirés :",
        len(bootstrap_draws),
    )

    print()
    print("Premiers blocs :")
    print(
        bootstrap_draws
        .head()
        .to_string(index=False)
    )

    # Fichiers d’audit légers permettant de
    # reproduire chaque tirage.
    block_catalog.to_parquet(
        OUTPUT_DIR
        / "joint_block_catalog_168h.parquet",
        index=False,
    )

    bootstrap_draws.to_parquet(
        OUTPUT_DIR
        / "bootstrap_draws_2027.parquet",
        index=False,
    )

    # Reconstruction du spot avec exactement
    # les mêmes positions historiques.
    spot_history = (
        joint_panel[
            "spot_eur_mwh"
        ]
        .to_numpy()
    )

    raw_spot_scenarios = (
        spot_history[
            source_positions
        ]
    )

    monthly_forwards = (
        tables[
            "monthly_forwards"
        ]
        .set_index(
            "target_month"
        )[
            "forward_base_eur_mwh"
        ]
    )

    centered_spot_scenarios, spot_audit = (
        recenter_spot_monthly(
            spot_scenarios=raw_spot_scenarios,
            hours=hours_2027,
            monthly_forwards=monthly_forwards,
            method="collective",
        )
    )

    spot_audit.to_csv(
        OUTPUT_DIR
        / "spot_forward_calibration_2027.csv",
        index=False,
    )

    print()
    print("RECALAGE COLLECTIF DU SPOT")
    print(
        spot_audit[
            [
                "target_month",
                "forward_eur_mwh",
                "collective_mean_before",
                "collective_mean_after",
                "dispersion_before",
                "dispersion_after",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print()
    print(
        "Construction des blocs et recalage du spot terminés."
    )

    print("\nCALCUL DES MÉTRIQUES ANNUELLES")

    annual_metrics = build_annual_scenario_metrics(
        joint_panel=joint_panel,
        source_positions=source_positions,
        spot_scenarios=centered_spot_scenarios,
        capacity_profiles=capacity_profiles,
        universe_membership=tables[
            "universe_membership"
        ],
        universe_columns=UNIVERSE_COLUMNS,
    )

    annual_metrics_path = (
        OUTPUT_DIR
        / "annual_scenario_metrics_2027.parquet"
    )

    annual_metrics.to_parquet(
        annual_metrics_path,
        index=False,
    )

    print(
        "Forme des métriques :",
        annual_metrics.shape,
    )

    entity_summary = (
        annual_metrics[
            [
                "universe",
                "portfolio",
                "level",
                "entity_id",
            ]
        ]
        .drop_duplicates()
        .groupby(
            [
                "universe",
                "portfolio",
                "level",
            ]
        )
        .size()
        .rename("n_entities")
        .reset_index()
    )

    print("\nEntités reconstruites :")
    print(
        entity_summary.to_string(
            index=False
        )
    )

    print(
        "\nCalcul des métriques annuelles terminé."
    )

    print("\nCALCUL DES PRIMES DE RISQUE")

    pricing_results = price_scenario_metrics(
        annual_metrics=annual_metrics,
        alpha=0.05,
        risk_aversions=(
            0.25,
            0.50,
            0.75,
            1.00,
        ),
    )

    pricing_results_path = (
        OUTPUT_DIR
        / "pricing_results_2027.parquet"
    )

    pricing_results.to_parquet(
        pricing_results_path,
        index=False,
    )

    pricing_results.to_csv(
        OUTPUT_DIR
        / "pricing_results_2027.csv",
        index=False,
    )

    print(
        "Forme du pricing :",
        pricing_results.shape,
    )

    # Résultats principaux pour lambda = 0,50.
    primary_results = (
        pricing_results.loc[
            np.isclose(
                pricing_results[
                    "risk_aversion"
                ],
                0.50,
            )
        ]
        .copy()
    )

    # Qualité des facteurs de charge régionaux.
    quality_table = (
        tables["group_quality"][
            [
                "group_id",
                "n_prms",
                "nominal_capacity_mw",
                "median_coverage",
                "share_hours_ge_80",
                "n_lf_above_1",
            ]
        ]
        .merge(
            tables["imputation_summary"][
                [
                    "group_id",
                    "n_imputed",
                    "imputed_share",
                ]
            ],
            on="group_id",
            how="left",
            validate="one_to_one",
        )
        .rename(
            columns={
                "group_id": "entity_id",
            }
        )
    )

    quality_table["quality_flag"] = (
        np.select(
            [
                quality_table[
                    "imputed_share"
                ].isna(),
                quality_table[
                    "imputed_share"
                ].le(0.10),
                quality_table[
                    "imputed_share"
                ].le(0.30),
            ],
            [
                "unknown",
                "robust",
                "moderate",
            ],
            default="fragile",
        )
    )

    primary_results = (
        primary_results.merge(
            quality_table,
            on="entity_id",
            how="left",
            validate="many_to_one",
        )
    )

    aggregated_rows = primary_results[
        "level"
    ].ne("technology_region")

    primary_results.loc[
        aggregated_rows,
        "quality_flag",
    ] = "aggregated"

    primary_results.to_csv(
        OUTPUT_DIR
        / "benchmark_results_all_levels_lambda_050.csv",
        index=False,
    )

    level_output_files = {
        "technology_region": (
            "benchmark_results_local_lambda_050.csv"
        ),
        "technology_national": (
            "benchmark_results_technology_lambda_050.csv"
        ),
        "portfolio_global": (
            "benchmark_results_global_lambda_050.csv"
        ),
    }

    for level, file_name in (
        level_output_files.items()
    ):
        primary_results.loc[
            primary_results[
                "level"
            ].eq(level)
        ].to_csv(
            OUTPUT_DIR / file_name,
            index=False,
        )
    main_global_results = (
        pricing_results.loc[
            pricing_results[
                "level"
            ].eq("portfolio_global")
            & np.isclose(
                pricing_results[
                    "risk_aversion"
                ],
                0.50,
            ),
            [
                "universe",
                "portfolio",
                "mean_production_mwh",
                "risk_neutral_price_eur_mwh",
                "risk_adjusted_price_eur_mwh",
                "risk_premium_eur_mwh",
            ],
        ]
        .sort_values(
            [
                "universe",
                "portfolio",
            ]
        )
    )

    print(
        "\nRÉSULTATS GLOBAUX — LAMBDA 0,50"
    )

    print(
        main_global_results
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nCalcul du pricing terminé."
    )

if __name__ == "__main__":
    main()