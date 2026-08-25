# ENR Risk Premium

Benchmark de tarification du risque d’un portefeuille de production d’électricité renouvelable, fondé sur un bootstrap conjoint des productions historiques et du prix spot.

Le projet construit des scénarios horaires 2027, puis calcule des prix neutres au risque et des prix ajustés par un critère de marge-CVaR.

## Objectif

Le benchmark vise à mesurer simplement et explicitement le risque conjoint :

- de production renouvelable ;
- de prix spot ;
- de dépendance entre le spot et les productions ;
- de variation du portefeuille contractuel.

Il constitue une méthode de référence destinée à être comparée à un moteur plus sophistiqué fondé sur des scénarios météorologiques.

## Méthode

La chaîne de calcul est la suivante :

1. construire les facteurs de charge historiques par technologie et région ;
2. pondérer les productions par les puissances réellement observables ;
3. distinguer production nulle, donnée manquante et centrale inactive ;
4. mesurer la couverture horaire de chaque groupe ;
5. construire un panel horaire conjoint production-spot ;
6. tirer des blocs temporels conjoints de 168 heures ;
7. appliquer les mêmes tirages à toutes les productions et au spot ;
8. reconstruire les productions horaires 2027 ;
9. comparer un portefeuille figé à un portefeuille dynamique ;
10. recaler collectivement les scénarios spot sur les forwards mensuels ;
11. calculer les volumes, revenus et marges annuels ;
12. résoudre les prix ajustés par marge-CVaR.

Les blocs historiques restent consécutifs. Ils ne sont jamais tirés heure par heure.

## Grandeurs économiques

Pour un scénario \(s\) :

\[
Q_s = \sum_t q_{s,t}
\]

\[
R_s = \sum_t P_{s,t}q_{s,t}
\]

La marge associée à un prix fixe \(k\) est :

\[
M_s(k)=R_s-kQ_s
\]

Le prix neutre au risque est :

\[
k_{\mathrm{neutre}}
=
\frac{\mathbb{E}[R_s]}
{\mathbb{E}[Q_s]}
\]

Le prix ajusté résout exactement :

\[
\mathbb{E}[M_s(k)]
-
\lambda
\left(
\mathbb{E}[M_s(k)]
-
\operatorname{CVaR}_{\alpha}(M_s(k))
\right)
=0
\]

La prime de risque est définie par :

\[
\text{prime}
=
k_{\mathrm{neutre}}
-
k_{\mathrm{ajusté}}
\]

## Portefeuilles

Deux représentations contractuelles sont calculées avec exactement les mêmes scénarios :

- `fixed` : capacités actives au 1er janvier 2027, conservées fictivement toute l’année ;
- `dynamic` : capacités variant selon les dates contractuelles réelles.

## Univers analysés

Trois univers sont disponibles :

- `all` : toutes les technologies exploitables ;
- `no_biomass_biogas` : exclusion de la biomasse et du biogaz ;
- `weather_mvp` : périmètre solaire-éolien comparable au moteur météorologique.

Le moteur prend en charge :

- solaire ;
- éolien ;
- hydroélectricité ;
- biomasse ;
- biogaz.

Une technologie prise en charge peut être absente des résultats d’une année si aucune centrale correspondante n’est contractuellement active.

## Niveaux de restitution

Les résultats sont produits aux niveaux suivants :

- technologie × région ;
- technologie nationale ;
- portefeuille global.

Chaque niveau est calculé pour les variantes `fixed` et `dynamic`.

## Paramètres de référence

| Paramètre | Valeur |
|---|---:|
| Année cible | 2027 |
| Nombre de scénarios | 1 500 |
| Longueur des blocs | 168 heures |
| Graine aléatoire | 20270824 |
| Seuil CVaR | 5 % |
| Aversions au risque | 0,25 ; 0,50 ; 0,75 ; 1,00 |
| Aversion principale | 0,50 |
| Convention temporelle | UTC |
| Snapshot forward | 24 août 2026 |

Le recalage principal impose que la moyenne mensuelle collective des scénarios spot soit égale au forward, tout en conservant leur dispersion entre scénarios.

## Structure du projet

```text
enr_risk_premium/
├── src/
│   └── enr_risk_premium/
│       ├── __init__.py
│       └── benchmark.py
├── tests/
│   └── test_benchmark.py
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
├── run_benchmark.py
├── pyproject.toml
└── README.md
```

`benchmark.py` contient les fonctions métier testées.

`run_benchmark.py` orchestre les entrées, les simulations et les sorties.

## Installation sous Windows

Depuis PowerShell :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Le projet nécessite Python 3.13 ou une version compatible avec les contraintes définies dans `pyproject.toml`.

## Entrées canoniques

Les fichiers suivants doivent être placés dans `data/processed/` :

```text
contracts_canonical.parquet
joint_panel_ready_for_bootstrap.parquet
portfolio_universe_membership_2027.parquet
eex_monthly_base_2027_snapshot_2026-08-24.parquet
load_factor_group_quality.parquet
load_factor_imputation_summary.parquet
```

Les données contractuelles, historiques et commerciales ne sont pas publiées dans le dépôt.

## Exécution

```powershell
python run_benchmark.py
```

Le calcul utilise une graine fixe afin de reproduire exactement les tirages.

## Tests

```powershell
python -m pytest -q
```

La version actuellement validée comporte 27 tests unitaires.

## Principales sorties

Le dossier `outputs/` reçoit notamment :

```text
capacity_summary_2027.csv
joint_block_catalog_168h.parquet
bootstrap_draws_2027.parquet
spot_forward_calibration_2027.csv
annual_scenario_metrics_2027.parquet
pricing_results_2027.parquet
pricing_results_2027.csv
benchmark_results_local_lambda_050.csv
benchmark_results_technology_lambda_050.csv
benchmark_results_global_lambda_050.csv
```

Les sorties calculées localement ne sont pas versionnées.

## Contrôles méthodologiques

Le pipeline vérifie notamment :

- l’unicité et l’ordre des heures ;
- la cohérence des capacités contractuelles ;
- l’utilisation des mêmes scénarios pour les portefeuilles figé et dynamique ;
- l’intégrité temporelle des blocs historiques ;
- l’égalité entre les moyennes collectives mensuelles du spot et les forwards ;
- la présence des mêmes scénarios pour chaque entité tarifée ;
- l’annulation numérique du critère de marge-CVaR.

## Limites

Ce benchmark :

- reproduit les dépendances observées historiquement, sans modèle causal météorologique ;
- dépend de la profondeur et de la qualité des historiques disponibles ;
- utilise des imputations documentées pour certains groupes incomplets ;
- suppose que les blocs historiques représentent des régimes plausibles pour 2027 ;
- ne modélise ni les frais d’équilibrage, ni les autres coûts, ni la marge commerciale.

Il doit être interprété comme une baseline transparente et non comme un modèle exhaustif du marché électrique.

## Confidentialité

Les données brutes, les tables contractuelles, les fichiers intermédiaires et les résultats économiques sont exclus du suivi Git.

Le dépôt public contient uniquement le code, les tests et la documentation méthodologique.