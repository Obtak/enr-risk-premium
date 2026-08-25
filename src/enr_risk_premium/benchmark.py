"""Benchmark par bootstrap conjoint production-spot."""

import numpy as np
import pandas as pd

from numpy.typing import ArrayLike
from scipy.optimize import brentq
from collections.abc import Mapping
DEFAULT_SEASON_MONTHS = {
    1: (12, 1, 2),
    2: (12, 1, 2),
    3: (3, 4, 5),
    4: (3, 4, 5),
    5: (3, 4, 5),
    6: (6, 7, 8),
    7: (6, 7, 8),
    8: (6, 7, 8),
    9: (9, 10, 11),
    10: (9, 10, 11),
    11: (9, 10, 11),
    12: (12, 1, 2),
}


def build_hourly_index(year: int) -> pd.DatetimeIndex:
    """Construire l'index horaire UTC complet d'une année."""

    return pd.date_range(
        start=f"{year}-01-01",
        end=f"{year + 1}-01-01",
        freq="h",
        inclusive="left",
        tz="UTC",
    )


def normalize_prm(value: object) -> object:
    """Normaliser un PRM tout en conservant ses zéros initiaux."""

    if pd.isna(value):
        return pd.NA

    text = str(value).strip()

    if text.lower() in {"", "?", "nan", "none", "<na>"}:
        return pd.NA

    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]

    if text.isdigit():
        text = text.zfill(14)

    return text

def _validate_scenario_arrays(
    production_mwh: ArrayLike,
    revenue_eur: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Valider les productions et revenus annuels par scénario."""

    production = np.asarray(
        production_mwh,
        dtype=float,
    )

    revenue = np.asarray(
        revenue_eur,
        dtype=float,
    )

    if production.ndim != 1 or revenue.ndim != 1:
        raise ValueError(
            "Les productions et revenus doivent être unidimensionnels."
        )

    if production.size == 0:
        raise ValueError("Aucun scénario fourni.")

    if production.shape != revenue.shape:
        raise ValueError(
            "Les productions et revenus doivent avoir la même taille."
        )

    if not np.isfinite(production).all():
        raise ValueError("Les productions contiennent une valeur invalide.")

    if not np.isfinite(revenue).all():
        raise ValueError("Les revenus contiennent une valeur invalide.")

    if (production < 0).any():
        raise ValueError("Une production annuelle ne peut pas être négative.")

    if production.mean() <= 0:
        raise ValueError("La production annuelle moyenne doit être positive.")

    return production, revenue


def lower_tail_cvar(
    values: ArrayLike,
    alpha: float = 0.05,
) -> float:
    """Calculer la moyenne exacte de la queue gauche."""

    observations = np.asarray(
        values,
        dtype=float,
    )

    if observations.ndim != 1 or observations.size == 0:
        raise ValueError(
            "La CVaR nécessite un vecteur non vide."
        )

    if not np.isfinite(observations).all():
        raise ValueError(
            "La CVaR ne peut pas contenir de valeur invalide."
        )

    if not 0 < alpha <= 1:
        raise ValueError(
            "Alpha doit appartenir à l'intervalle ]0, 1]."
        )

    observations = np.sort(observations)

    tail_size = alpha * observations.size
    n_complete = int(np.floor(tail_size))
    fraction = tail_size - n_complete

    tail_sum = observations[:n_complete].sum()

    if fraction > 0:
        tail_sum += (
            fraction
            * observations[n_complete]
        )

    return float(tail_sum / tail_size)


def risk_neutral_price(
    production_mwh: ArrayLike,
    revenue_eur: ArrayLike,
) -> float:
    """Calculer le prix qui annule la marge moyenne."""

    production, revenue = _validate_scenario_arrays(
        production_mwh,
        revenue_eur,
    )

    return float(
        revenue.mean()
        / production.mean()
    )


def margin_risk_criterion(
    price_eur_mwh: float,
    production_mwh: ArrayLike,
    revenue_eur: ArrayLike,
    alpha: float = 0.05,
    risk_aversion: float = 0.50,
) -> float:
    """Calculer le critère de marge ajustée par la CVaR."""

    if not 0 <= risk_aversion <= 1:
        raise ValueError(
            "L'aversion au risque doit appartenir à [0, 1]."
        )

    production, revenue = _validate_scenario_arrays(
        production_mwh,
        revenue_eur,
    )

    margin = (
        revenue
        - price_eur_mwh * production
    )

    mean_margin = margin.mean()

    cvar_margin = lower_tail_cvar(
        margin,
        alpha=alpha,
    )

    criterion = (
        mean_margin
        - risk_aversion
        * (mean_margin - cvar_margin)
    )

    return float(criterion)


def solve_adjusted_price(
    production_mwh: ArrayLike,
    revenue_eur: ArrayLike,
    alpha: float = 0.05,
    risk_aversion: float = 0.50,
    price_bounds: tuple[float, float] = (
        -1000.0,
        1000.0,
    ),
) -> float:
    """Résoudre exactement le prix annulant le critère de risque."""

    lower_price, upper_price = price_bounds

    if lower_price >= upper_price:
        raise ValueError(
            "La borne basse doit être inférieure à la borne haute."
        )

    def objective(price: float) -> float:
        return margin_risk_criterion(
            price_eur_mwh=price,
            production_mwh=production_mwh,
            revenue_eur=revenue_eur,
            alpha=alpha,
            risk_aversion=risk_aversion,
        )

    lower_value = objective(lower_price)
    upper_value = objective(upper_price)

    if lower_value * upper_value > 0:
        raise ValueError(
            "Les bornes choisies n'encadrent pas la racine."
        )

    adjusted_price = brentq(
        objective,
        lower_price,
        upper_price,
        xtol=1e-10,
        rtol=1e-12,
    )

    return float(adjusted_price)

def recenter_spot_monthly(
    spot_scenarios: ArrayLike,
    hours: pd.DatetimeIndex,
    monthly_forwards: pd.Series,
    method: str = "collective",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Recaler les scénarios spot sur les forwards mensuels."""

    spot = np.asarray(
        spot_scenarios,
        dtype=float,
    )

    if spot.ndim != 2:
        raise ValueError(
            "Les scénarios spot doivent former une matrice."
        )

    if spot.shape[1] != len(hours):
        raise ValueError(
            "Le nombre d'heures du spot et de l'index diffère."
        )

    if not np.isfinite(spot).all():
        raise ValueError(
            "Les scénarios spot contiennent une valeur invalide."
        )

    if not isinstance(hours, pd.DatetimeIndex):
        raise TypeError(
            "hours doit être un DatetimeIndex."
        )

    if hours.tz is None:
        raise ValueError(
            "L'index horaire doit posséder un fuseau."
        )

    if method not in {"collective", "individual"}:
        raise ValueError(
            "method doit valoir 'collective' ou 'individual'."
        )

    forwards = pd.Series(
        monthly_forwards,
        dtype=float,
    )

    if not forwards.index.is_unique:
        raise ValueError(
            "Chaque mois doit posséder un seul forward."
        )

    centered = spot.copy()
    audit_rows = []

    for month in sorted(hours.month.unique()):
        if month not in forwards.index:
            raise ValueError(
                f"Forward absent pour le mois {month}."
            )

        hour_mask = np.asarray(
            hours.month == month
        )

        forward = float(
            forwards.loc[month]
        )

        means_before = centered[
            :,
            hour_mask,
        ].mean(axis=1)

        if method == "collective":
            shifts = np.full(
                centered.shape[0],
                forward - means_before.mean(),
            )
        else:
            shifts = (
                forward
                - means_before
            )

        centered[
            :,
            hour_mask,
        ] += shifts[:, None]

        means_after = centered[
            :,
            hour_mask,
        ].mean(axis=1)

        audit_rows.append(
            {
                "target_month": int(month),
                "method": method,
                "forward_eur_mwh": forward,
                "collective_mean_before": (
                    float(means_before.mean())
                ),
                "collective_mean_after": (
                    float(means_after.mean())
                ),
                "dispersion_before": (
                    float(means_before.std())
                ),
                "dispersion_after": (
                    float(means_after.std())
                ),
                "minimum_shift": float(shifts.min()),
                "maximum_shift": float(shifts.max()),
            }
        )

    audit = pd.DataFrame(audit_rows)

    return centered, audit


def compute_annual_metrics(
    load_factor_scenarios: ArrayLike,
    spot_scenarios: ArrayLike,
    capacity_mw: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculer la production et le revenu annuels par scénario."""

    load_factors = np.asarray(
        load_factor_scenarios,
        dtype=float,
    )

    spot = np.asarray(
        spot_scenarios,
        dtype=float,
    )

    if load_factors.ndim != 2:
        raise ValueError(
            "Les facteurs de charge doivent former une matrice."
        )

    if load_factors.shape != spot.shape:
        raise ValueError(
            "Le spot et les facteurs de charge doivent être alignés."
        )

    if not np.isfinite(load_factors).all():
        raise ValueError(
            "Les facteurs de charge contiennent une valeur invalide."
        )

    if not np.isfinite(spot).all():
        raise ValueError(
            "Le spot contient une valeur invalide."
        )

    if (
        (load_factors < -1e-12).any()
        or (load_factors > 1 + 1e-12).any()
    ):
        raise ValueError(
            "Les facteurs de charge doivent appartenir à [0, 1]."
        )

    capacity = np.asarray(
        capacity_mw,
        dtype=float,
    )

    if capacity.ndim == 0:
        capacity = np.full(
            load_factors.shape[1],
            float(capacity),
        )

    if (
        capacity.ndim != 1
        or capacity.size != load_factors.shape[1]
    ):
        raise ValueError(
            "La capacité doit avoir une valeur par heure."
        )

    if (
        not np.isfinite(capacity).all()
        or (capacity < 0).any()
    ):
        raise ValueError(
            "Les capacités doivent être finies et positives ou nulles."
        )

    hourly_production = (
        load_factors
        * capacity[None, :]
    )

    annual_production = hourly_production.sum(
        axis=1
    )

    annual_revenue = (
        hourly_production
        * spot
    ).sum(axis=1)

    return annual_production, annual_revenue


def compute_scenario_margin(
    production_mwh: ArrayLike,
    revenue_eur: ArrayLike,
    fixed_price_eur_mwh: float,
) -> np.ndarray:
    """Calculer la marge annuelle de chaque scénario."""

    production, revenue = _validate_scenario_arrays(
        production_mwh,
        revenue_eur,
    )

    if not np.isfinite(fixed_price_eur_mwh):
        raise ValueError(
            "Le prix fixe doit être fini."
        )

    return (
        revenue
        - fixed_price_eur_mwh * production
    )
def build_annual_scenario_metrics(
    joint_panel: pd.DataFrame,
    source_positions: ArrayLike,
    spot_scenarios: ArrayLike,
    capacity_profiles: Mapping[
        tuple[str, str],
        pd.DataFrame,
    ],
    universe_membership: pd.DataFrame,
    universe_columns: Mapping[str, str],
) -> pd.DataFrame:
    """Calculer Q et R aux niveaux local, national et global."""

    positions = np.asarray(source_positions)
    spot = np.asarray(
        spot_scenarios,
        dtype=float,
    )

    if positions.ndim != 2 or positions.size == 0:
        raise ValueError(
            "Les positions sources doivent former une matrice."
        )

    if (
        not np.issubdtype(
            positions.dtype,
            np.integer,
        )
        or positions.min() < 0
        or positions.max() >= len(joint_panel)
    ):
        raise ValueError(
            "Les positions sources sont invalides."
        )

    if spot.shape != positions.shape:
        raise ValueError(
            "Le spot et les positions doivent être alignés."
        )

    metadata = (
        universe_membership[
            [
                "group_id",
                "technology",
                "region_code",
                "region_name",
            ]
        ]
        .drop_duplicates()
    )

    if metadata["group_id"].duplicated().any():
        raise ValueError(
            "Métadonnées contradictoires pour un groupe."
        )

    metadata = metadata.set_index(
        "group_id"
    )

    scenario_ids = np.arange(
        positions.shape[0],
        dtype=np.int32,
    )

    local_frames = []

    for universe, flag_column in (
        universe_columns.items()
    ):
        selected = universe_membership.loc[
            universe_membership[
                flag_column
            ].fillna(False)
        ]

        active_groups = sorted(
            selected["group_id"]
            .dropna()
            .unique()
        )

        for group_id in active_groups:
            if group_id not in joint_panel.columns:
                raise ValueError(
                    f"LF absent du panel : {group_id}"
                )

            load_factors = (
                joint_panel[group_id]
                .to_numpy(dtype=float)[positions]
            )

            group_info = metadata.loc[
                group_id
            ]

            for portfolio in (
                "fixed",
                "dynamic",
            ):
                capacity = (
                    capacity_profiles[
                        (
                            universe,
                            portfolio,
                        )
                    ][group_id]
                    .to_numpy(dtype=float)
                )

                production, revenue = (
                    compute_annual_metrics(
                        load_factor_scenarios=(
                            load_factors
                        ),
                        spot_scenarios=spot,
                        capacity_mw=capacity,
                    )
                )

                local_frames.append(
                    pd.DataFrame(
                        {
                            "scenario_id": scenario_ids,
                            "universe": universe,
                            "portfolio": portfolio,
                            "level": (
                                "technology_region"
                            ),
                            "entity_id": group_id,
                            "technology": group_info[
                                "technology"
                            ],
                            "region_code": group_info[
                                "region_code"
                            ],
                            "region_name": group_info[
                                "region_name"
                            ],
                            "production_mwh": (
                                production
                            ),
                            "spot_revenue_eur": (
                                revenue
                            ),
                        }
                    )
                )

    local = pd.concat(
        local_frames,
        ignore_index=True,
    )

    values = [
        "production_mwh",
        "spot_revenue_eur",
    ]

    national = (
        local.groupby(
            [
                "scenario_id",
                "universe",
                "portfolio",
                "technology",
            ],
            as_index=False,
            sort=True,
        )[values]
        .sum()
        .assign(
            level="technology_national",
            entity_id=lambda table: table[
                "technology"
            ],
            region_code=np.nan,
            region_name=np.nan,
        )
    )

    global_portfolio = (
        local.groupby(
            [
                "scenario_id",
                "universe",
                "portfolio",
            ],
            as_index=False,
            sort=True,
        )[values]
        .sum()
        .assign(
            level="portfolio_global",
            entity_id="portfolio_total",
            technology=np.nan,
            region_code=np.nan,
            region_name=np.nan,
        )
    )

    columns = [
        "scenario_id",
        "universe",
        "portfolio",
        "level",
        "entity_id",
        "technology",
        "region_code",
        "region_name",
        "production_mwh",
        "spot_revenue_eur",
    ]

    return pd.concat(
        [
            local,
            national,
            global_portfolio,
        ],
        ignore_index=True,
    )[columns]

def price_scenario_metrics(
    annual_metrics: pd.DataFrame,
    alpha: float = 0.05,
    risk_aversions: tuple[float, ...] = (
        0.25,
        0.50,
        0.75,
        1.00,
    ),
) -> pd.DataFrame:
    """Calculer les prix neutres, ajustés et les primes."""

    entity_columns = [
        "universe",
        "portfolio",
        "level",
        "entity_id",
        "technology",
        "region_code",
        "region_name",
    ]

    required_columns = {
        "scenario_id",
        "production_mwh",
        "spot_revenue_eur",
        *entity_columns,
    }

    missing_columns = required_columns - set(
        annual_metrics.columns
    )

    if missing_columns:
        raise ValueError(
            "Colonnes de métriques absentes : "
            f"{sorted(missing_columns)}"
        )

    if annual_metrics.empty:
        raise ValueError(
            "La table des métriques annuelles est vide."
        )

    scenario_ids = np.sort(
        annual_metrics["scenario_id"].unique()
    )

    pricing_rows = []

    for entity_key, entity_metrics in annual_metrics.groupby(
        entity_columns,
        sort=True,
        dropna=False,
    ):
        entity_metrics = entity_metrics.sort_values(
            "scenario_id"
        )

        if not np.array_equal(
            entity_metrics["scenario_id"].to_numpy(),
            scenario_ids,
        ):
            raise ValueError(
                "Chaque entité doit posséder exactement "
                "les mêmes scénarios."
            )

        production = entity_metrics[
            "production_mwh"
        ].to_numpy(dtype=float)

        revenue = entity_metrics[
            "spot_revenue_eur"
        ].to_numpy(dtype=float)

        neutral_price = risk_neutral_price(
            production,
            revenue,
        )

        metadata = dict(
            zip(entity_columns, entity_key)
        )

        for risk_aversion in risk_aversions:
            adjusted_price = solve_adjusted_price(
                production_mwh=production,
                revenue_eur=revenue,
                alpha=alpha,
                risk_aversion=risk_aversion,
            )

            margin = compute_scenario_margin(
                production_mwh=production,
                revenue_eur=revenue,
                fixed_price_eur_mwh=adjusted_price,
            )

            expected_margin = float(margin.mean())

            cvar_margin = lower_tail_cvar(
                margin,
                alpha=alpha,
            )

            residual = (
                expected_margin
                - risk_aversion
                * (expected_margin - cvar_margin)
            )

            pricing_rows.append(
                {
                    **metadata,
                    "n_scenarios": len(production),
                    "cvar_alpha": alpha,
                    "risk_aversion": risk_aversion,
                    "mean_production_mwh": float(
                        production.mean()
                    ),
                    "risk_neutral_price_eur_mwh": (
                        neutral_price
                    ),
                    "risk_adjusted_price_eur_mwh": (
                        adjusted_price
                    ),
                    "risk_premium_eur_mwh": (
                        neutral_price - adjusted_price
                    ),
                    "expected_margin_at_adjusted_eur": (
                        expected_margin
                    ),
                    "cvar_margin_at_adjusted_eur": (
                        cvar_margin
                    ),
                    "criterion_residual_eur": residual,
                }
            )

    return pd.DataFrame(pricing_rows)

def catalog_joint_blocks(
    joint_panel: pd.DataFrame,
    block_hours: int = 168,
) -> pd.DataFrame:
    """Répertorier les blocs conjoints historiques admissibles."""

    if not isinstance(joint_panel.index, pd.DatetimeIndex):
        raise TypeError(
            "Le panel doit posséder un DatetimeIndex."
        )

    if joint_panel.index.tz is None:
        raise ValueError(
            "L'index du panel doit posséder un fuseau."
        )

    if joint_panel.index.has_duplicates:
        raise ValueError(
            "L'index du panel contient des doublons."
        )

    if not joint_panel.index.is_monotonic_increasing:
        raise ValueError(
            "L'index du panel doit être trié."
        )

    if block_hours <= 0:
        raise ValueError(
            "La longueur du bloc doit être positive."
        )

    panel_values = joint_panel.to_numpy(
        dtype=float,
    )

    candidate_rows = []

    for start_position, source_start in enumerate(
        joint_panel.index
    ):
        # Chaque bloc historique commence à minuit.
        if source_start.hour != 0:
            continue

        source_hours = pd.date_range(
            start=source_start,
            periods=block_hours,
            freq="h",
        )

        source_end = source_hours[-1]

        # Le bloc doit rester dans son mois source.
        if source_end.month != source_start.month:
            continue

        source_positions = (
            joint_panel.index.get_indexer(
                source_hours
            )
        )

        # Une position égale à -1 indique une heure absente.
        if (source_positions < 0).any():
            continue

        # Les productions et le spot doivent tous être disponibles.
        if not np.isfinite(
            panel_values[source_positions]
        ).all():
            continue

        candidate_rows.append(
            {
                "source_start": source_start,
                "source_end": source_end,
                "source_year": source_start.year,
                "source_month": source_start.month,
                "start_position": int(
                    source_positions[0]
                ),
                "block_hours": block_hours,
            }
        )

    return pd.DataFrame(
        candidate_rows,
        columns=[
            "source_start",
            "source_end",
            "source_year",
            "source_month",
            "start_position",
            "block_hours",
        ],
    )

def draw_joint_block_positions(
    candidate_blocks: pd.DataFrame,
    target_hours: pd.DatetimeIndex,
    n_scenarios: int = 1500,
    block_hours: int = 168,
    random_seed: int = 20270824,
    season_months: Mapping[
        int,
        tuple[int, ...],
    ] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Tirer les positions sources communes à toutes les variables."""

    if not isinstance(target_hours, pd.DatetimeIndex):
        raise TypeError(
            "target_hours doit être un DatetimeIndex."
        )

    if target_hours.tz is None:
        raise ValueError(
            "L'index cible doit posséder un fuseau."
        )

    if target_hours.has_duplicates:
        raise ValueError(
            "L'index cible contient des doublons."
        )

    if not target_hours.is_monotonic_increasing:
        raise ValueError(
            "L'index cible doit être trié."
        )

    target_differences = (
        target_hours
        .to_series()
        .diff()
        .dropna()
    )

    if not target_differences.eq(
        pd.Timedelta(hours=1)
    ).all():
        raise ValueError(
            "L'index cible doit être strictement horaire."
        )

    if n_scenarios <= 0:
        raise ValueError(
            "Le nombre de scénarios doit être positif."
        )

    if candidate_blocks.empty:
        raise ValueError(
            "Aucun bloc historique admissible."
        )

    required_columns = {
        "source_start",
        "source_year",
        "source_month",
        "start_position",
        "block_hours",
    }

    missing_columns = (
        required_columns
        - set(candidate_blocks.columns)
    )

    if missing_columns:
        raise ValueError(
            "Colonnes absentes du catalogue : "
            f"{sorted(missing_columns)}"
        )

    if not candidate_blocks[
        "block_hours"
    ].eq(block_hours).all():
        raise ValueError(
            "La longueur demandée diffère de celle du catalogue."
        )

    if season_months is None:
        season_months = DEFAULT_SEASON_MONTHS

    rng = np.random.default_rng(
        random_seed
    )

    source_positions = np.full(
        shape=(
            n_scenarios,
            len(target_hours),
        ),
        fill_value=-1,
        dtype=np.int32,
    )

    draw_rows = []

    target_months = (
        pd.Index(target_hours.month)
        .unique()
        .tolist()
    )

    for scenario_id in range(n_scenarios):
        for target_month in target_months:
            target_positions = np.flatnonzero(
                target_hours.month
                == target_month
            )

            same_month_pool = candidate_blocks.loc[
                candidate_blocks[
                    "source_month"
                ].eq(target_month)
            ]

            if not same_month_pool.empty:
                candidate_pool = same_month_pool
                selection_rule = "same_month"

            else:
                fallback_months = season_months[
                    target_month
                ]

                candidate_pool = candidate_blocks.loc[
                    candidate_blocks[
                        "source_month"
                    ].isin(fallback_months)
                ]

                selection_rule = "season_fallback"

            if candidate_pool.empty:
                raise ValueError(
                    "Aucun bloc disponible pour "
                    f"le mois cible {target_month}."
                )

            n_target_hours = len(
                target_positions
            )

            n_blocks = int(
                np.ceil(
                    n_target_hours
                    / block_hours
                )
            )

            for block_number in range(n_blocks):
                offset = (
                    block_number
                    * block_hours
                )

                hours_used = min(
                    block_hours,
                    n_target_hours - offset,
                )

                selected_row = candidate_pool.iloc[
                    rng.integers(
                        0,
                        len(candidate_pool),
                    )
                ]

                source_start_position = int(
                    selected_row[
                        "start_position"
                    ]
                )

                selected_target_positions = (
                    target_positions[
                        offset:
                        offset + hours_used
                    ]
                )

                selected_source_positions = np.arange(
                    source_start_position,
                    source_start_position
                    + hours_used,
                    dtype=np.int32,
                )

                source_positions[
                    scenario_id,
                    selected_target_positions,
                ] = selected_source_positions

                draw_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "target_month": target_month,
                        "block_number": block_number,
                        "target_start": target_hours[
                            selected_target_positions[0]
                        ],
                        "source_start": selected_row[
                            "source_start"
                        ],
                        "source_year": int(
                            selected_row[
                                "source_year"
                            ]
                        ),
                        "source_month": int(
                            selected_row[
                                "source_month"
                            ]
                        ),
                        "hours_used": hours_used,
                        "selection_rule": selection_rule,
                    }
                )

    if (source_positions < 0).any():
        raise RuntimeError(
            "Certaines heures cibles n'ont pas reçu de source."
        )

    draws = pd.DataFrame(
        draw_rows
    )

    return source_positions, draws

def _as_utc_datetime_series(
    values: pd.Series,
) -> pd.Series:
    """Convertir une série de dates en UTC."""

    dates = pd.to_datetime(
        values,
        errors="coerce",
        format="mixed",
    )

    if dates.dt.tz is None:
        return dates.dt.tz_localize(
            "UTC"
        )

    return dates.dt.tz_convert(
        "UTC"
    )


def build_group_capacity_profiles(
    contracts: pd.DataFrame,
    universe_membership: pd.DataFrame,
    target_hours: pd.DatetimeIndex,
    universe_column: str,
    group_ids: list[str] | None = None,
    reference_time: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construire les capacités figées et dynamiques d'un univers."""

    required_contract_columns = {
        "plant_id",
        "capacity_mw",
        "contract_start",
        "contract_end_exclusive",
        "date_issue",
        "capacity_issue",
    }

    missing_contract_columns = (
        required_contract_columns
        - set(contracts.columns)
    )

    if missing_contract_columns:
        raise ValueError(
            "Colonnes contractuelles absentes : "
            f"{sorted(missing_contract_columns)}"
        )

    required_membership_columns = {
        "plant_id",
        "group_id",
        universe_column,
    }

    missing_membership_columns = (
        required_membership_columns
        - set(universe_membership.columns)
    )

    if missing_membership_columns:
        raise ValueError(
            "Colonnes d'univers absentes : "
            f"{sorted(missing_membership_columns)}"
        )

    if not isinstance(target_hours, pd.DatetimeIndex):
        raise TypeError(
            "target_hours doit être un DatetimeIndex."
        )

    if target_hours.tz is None:
        raise ValueError(
            "Les heures cibles doivent posséder un fuseau."
        )

    if (
        target_hours.has_duplicates
        or not target_hours.is_monotonic_increasing
    ):
        raise ValueError(
            "Les heures cibles doivent être uniques et triées."
        )

    membership = universe_membership[
        [
            "plant_id",
            "group_id",
            universe_column,
        ]
    ].copy()

    membership["plant_id"] = (
        membership["plant_id"]
        .map(normalize_prm)
        .astype("string")
    )

    if membership["plant_id"].duplicated().any():
        raise ValueError(
            "Une centrale apparaît plusieurs fois dans l'univers."
        )

    selected_membership = membership.loc[
        membership[
            universe_column
        ].fillna(False)
    ].copy()

    if selected_membership.empty:
        raise ValueError(
            f"L'univers {universe_column} est vide."
        )

    contract_periods = contracts[
        list(required_contract_columns)
    ].copy()

    contract_periods["plant_id"] = (
        contract_periods["plant_id"]
        .map(normalize_prm)
        .astype("string")
    )

    contract_periods["contract_start"] = (
        _as_utc_datetime_series(
            contract_periods[
                "contract_start"
            ]
        )
    )

    contract_periods[
        "contract_end_exclusive"
    ] = _as_utc_datetime_series(
        contract_periods[
            "contract_end_exclusive"
        ]
    )

    contract_periods = contract_periods.merge(
        selected_membership[
            [
                "plant_id",
                "group_id",
            ]
        ],
        on="plant_id",
        how="inner",
        validate="many_to_one",
    )

    valid_period = (
        contract_periods[
            "capacity_mw"
        ].gt(0)
        & contract_periods[
            "contract_start"
        ].notna()
        & contract_periods[
            "contract_end_exclusive"
        ].notna()
        & ~contract_periods[
            "date_issue"
        ].fillna(True)
        & ~contract_periods[
            "capacity_issue"
        ].fillna(True)
    )

    contract_periods = contract_periods.loc[
        valid_period
    ].copy()

    target_start = target_hours[0]

    target_end_exclusive = (
        target_hours[-1]
        + pd.Timedelta(hours=1)
    )

    contract_periods = contract_periods.loc[
        contract_periods[
            "contract_start"
        ].lt(target_end_exclusive)
        & contract_periods[
            "contract_end_exclusive"
        ].gt(target_start)
    ].copy()

    plants_with_period = set(
        contract_periods["plant_id"]
    )

    selected_plants = set(
        selected_membership["plant_id"]
    )

    missing_plants = sorted(
        selected_plants
        - plants_with_period
    )

    if missing_plants:
        raise ValueError(
            "Centrales sans période contractuelle "
            f"valide en 2027 : {missing_plants}"
        )

    if group_ids is None:
        group_ids = sorted(
            selected_membership[
                "group_id"
            ].unique()
        )
    else:
        group_ids = list(group_ids)

    missing_groups = (
        set(
            selected_membership[
                "group_id"
            ]
        )
        - set(group_ids)
    )

    if missing_groups:
        raise ValueError(
            "Groupes absents de la grille cible : "
            f"{sorted(missing_groups)}"
        )

    plant_ids = (
        selected_membership[
            "plant_id"
        ]
        .sort_values()
        .tolist()
    )

    plant_position = {
        plant_id: position
        for position, plant_id in enumerate(
            plant_ids
        )
    }

    plant_capacity = np.zeros(
        (
            len(target_hours),
            len(plant_ids),
        ),
        dtype=float,
    )

    for period in contract_periods.itertuples(
        index=False
    ):
        active_hours = (
            (target_hours >= period.contract_start)
            & (
                target_hours < period.contract_end_exclusive
            )
        )

        column_position = plant_position[
            period.plant_id
        ]

        plant_capacity[
            active_hours,
            column_position,
        ] = np.maximum(
            plant_capacity[
                active_hours,
                column_position,
            ],
            period.capacity_mw,
        )

    dynamic_capacity = pd.DataFrame(
        0.0,
        index=target_hours,
        columns=group_ids,
    )

    plant_group = (
        selected_membership
        .set_index("plant_id")[
            "group_id"
        ]
    )

    for group_id in group_ids:
        group_plants = [
            plant_position[plant_id]
            for plant_id in plant_ids
            if plant_group.loc[
                plant_id
            ] == group_id
        ]

        if group_plants:
            dynamic_capacity[
                group_id
            ] = plant_capacity[
                :,
                group_plants,
            ].sum(axis=1)

    if reference_time is None:
        reference_time = target_hours[0]
    else:
        reference_time = pd.Timestamp(
            reference_time
        )

        if reference_time.tzinfo is None:
            reference_time = (
                reference_time.tz_localize(
                    "UTC"
                )
            )
        else:
            reference_time = (
                reference_time.tz_convert(
                    "UTC"
                )
            )

    reference_position = (
        target_hours.get_indexer(
            [reference_time]
        )[0]
    )

    if reference_position < 0:
        raise ValueError(
            "La date de référence n'appartient pas à l'index cible."
        )

    fixed_values = dynamic_capacity.iloc[
        reference_position
    ].to_numpy()

    fixed_capacity = pd.DataFrame(
        np.broadcast_to(
            fixed_values,
            (
                len(target_hours),
                len(group_ids),
            ),
        ).copy(),
        index=target_hours,
        columns=group_ids,
    )

    fixed_capacity.columns.name = "group_id"
    dynamic_capacity.columns.name = "group_id"

    return fixed_capacity, dynamic_capacity