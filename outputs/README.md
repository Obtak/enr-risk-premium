# Sorties du benchmark

Ce dossier reçoit les résultats générés par :

```powershell
python run_benchmark.py
```

Les fichiers calculés ne sont pas versionnés. Ils peuvent être reconstruits à partir des entrées canoniques, des paramètres et de la graine aléatoire documentés.

## Audits de simulation

### `capacity_summary_2027.csv`

Résumé des capacités de chaque univers :

- nombre de centrales ;
- nombre de groupes actifs ;
- capacité du portefeuille figé ;
- capacité dynamique au 1er janvier ;
- capacité dynamique au 31 décembre.

### `joint_block_catalog_168h.parquet`

Catalogue des blocs historiques admissibles.

Principales colonnes :

```text
source_start
source_end
source_year
source_month
start_position
block_hours
```

### `bootstrap_draws_2027.parquet`

Traçabilité complète des blocs effectivement tirés.

Principales colonnes :

```text
scenario_id
target_month
block_number
target_start
source_start
source_year
source_month
hours_used
selection_rule
```

`selection_rule` indique si le bloc provient du même mois ou d’un repli saisonnier.

### `spot_forward_calibration_2027.csv`

Audit du recalage mensuel collectif du spot.

Il compare notamment :

- le forward mensuel ;
- la moyenne collective avant recalage ;
- la moyenne collective après recalage ;
- la dispersion entre scénarios avant et après recalage.

La moyenne après recalage doit être égale au forward, tandis que la dispersion collective doit être conservée.

## Métriques par scénario

### `annual_scenario_metrics_2027.parquet`

Cette table contient une ligne par scénario, univers, portefeuille et entité.

| Colonne | Description |
|---|---|
| `scenario_id` | Identifiant du scénario |
| `universe` | Univers contractuel |
| `portfolio` | `fixed` ou `dynamic` |
| `level` | Niveau d’agrégation |
| `entity_id` | Groupe, technologie ou portefeuille |
| `technology` | Technologie, si applicable |
| `region_code` | Code régional, si applicable |
| `region_name` | Région, si applicable |
| `production_mwh` | Production annuelle |
| `spot_revenue_eur` | Revenu spot annuel |

Les valeurs possibles de `level` sont :

```text
technology_region
technology_national
portfolio_global
```

## Résultats de pricing

### `pricing_results_2027.parquet`

### `pricing_results_2027.csv`

Ces deux fichiers contiennent les mêmes résultats dans deux formats.

| Colonne | Description |
|---|---|
| `n_scenarios` | Nombre de scénarios |
| `cvar_alpha` | Seuil de queue gauche |
| `risk_aversion` | Paramètre d’aversion au risque |
| `mean_production_mwh` | Production annuelle moyenne |
| `risk_neutral_price_eur_mwh` | Prix annulant la marge moyenne |
| `risk_adjusted_price_eur_mwh` | Prix annulant le critère marge-CVaR |
| `risk_premium_eur_mwh` | Prix neutre moins prix ajusté |
| `expected_margin_at_adjusted_eur` | Marge moyenne au prix ajusté |
| `cvar_margin_at_adjusted_eur` | CVaR basse de la marge |
| `criterion_residual_eur` | Résidu numérique du critère |

Un résidu proche de zéro confirme que le prix ajusté a été résolu correctement.

## Résultats principaux

Pour l’aversion au risque principale de 0,50 :

```text
benchmark_results_all_levels_lambda_050.csv
benchmark_results_local_lambda_050.csv
benchmark_results_technology_lambda_050.csv
benchmark_results_global_lambda_050.csv
```

Le fichier local contient également les indicateurs de qualité des facteurs de charge :

```text
n_prms
nominal_capacity_mw
median_coverage
share_hours_ge_80
n_lf_above_1
n_imputed
imputed_share
quality_flag
```

## Confidentialité

Les résultats peuvent révéler des caractéristiques économiques ou contractuelles du portefeuille. Ils restent donc exclus du suivi Git et doivent être examinés avant toute diffusion externe.