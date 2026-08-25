# Données canoniques traitées

Ce dossier reçoit les six entrées nécessaires à `run_benchmark.py`.

Les véritables fichiers Parquet ne sont pas versionnés, car ils contiennent des données contractuelles ou commerciales. Seul ce document décrivant leurs schémas est publié.

## Conventions communes

- les dates horaires sont exprimées en UTC ;
- les index temporels doivent être triés et sans doublon ;
- les puissances sont exprimées en MW ;
- les productions sont exprimées en MWh ;
- les prix sont exprimés en EUR/MWh ;
- les identifiants PRM sont conservés comme chaînes de caractères ;
- une fin contractuelle exclusive désigne la première heure inactive ;
- un `group_id` suit la convention `<technology>__<region_code>`.

Exemples de groupes :

```text
solar__76
wind__53
hydro__84
biogas__76
```

## 1. `contracts_canonical.parquet`

Table des périodes contractuelles canoniques.

Une centrale peut posséder plusieurs lignes lorsqu’elle comporte plusieurs périodes contractuelles.

| Colonne | Type attendu | Description |
|---|---|---|
| `plant_id` | chaîne | Identifiant interne stable de la centrale |
| `prm` | chaîne nullable | Identifiant PRM normalisé |
| `plant_name` | chaîne | Nom de la centrale |
| `contract_name` | chaîne | Nom du contrat |
| `technology` | chaîne | `solar`, `wind`, `hydro`, `biomass` ou `biogas` |
| `capacity_mw` | réel | Puissance contractuelle en MW |
| `contract_start` | date | Première date contractuellement active |
| `contract_end` | date | Dernière date contractuelle incluse |
| `contract_end_exclusive` | date | Première date contractuellement inactive |
| `code_postal` | chaîne | Code postal conservé comme texte |
| `municipality` | chaîne | Commune |
| `date_issue` | booléen | Indique une période contractuelle invalide |
| `capacity_issue` | booléen | Indique une puissance inexploitable |
| `latitude` | réel nullable | Latitude |
| `longitude` | réel nullable | Longitude |
| `region_code` | chaîne | Code de la région |
| `region_name` | chaîne | Nom de la région |

Le runner utilise directement :

```text
plant_id
capacity_mw
contract_start
contract_end_exclusive
date_issue
capacity_issue
```

## 2. `joint_panel_ready_for_bootstrap.parquet`

Panel historique conjoint utilisé pour le bootstrap.

### Index

| Élément | Règle |
|---|---|
| Nom | `date` |
| Type | `DatetimeIndex` |
| Fuseau | UTC |
| Fréquence | horaire |
| Ordre | croissant |
| Doublons | interdits |

### Colonnes

- `spot_eur_mwh` : prix spot historique ;
- une colonne par groupe technologie-région ;
- chaque colonne de groupe contient un facteur de charge compris entre 0 et 1.

Toutes les variables d’une même ligne correspondent à la même heure historique. Le tirage d’un bloc conserve donc leur dépendance commune.

Le panel transmis au bootstrap ne doit plus contenir de valeur manquante.

## 3. `portfolio_universe_membership_2027.parquet`

Table d’appartenance des centrales aux univers 2027.

| Colonne | Type attendu | Description |
|---|---|---|
| `plant_id` | chaîne | Identifiant interne de la centrale |
| `technology` | chaîne | Technologie canonique |
| `region_code` | chaîne | Code régional |
| `region_name` | chaîne | Nom régional |
| `group_id` | chaîne | Groupe technologie-région |
| `in_all` | booléen | Appartenance à l’univers complet |
| `in_no_biomass_biogas` | booléen | Appartenance à l’univers sans biomasse ni biogaz |
| `in_weather_mvp` | booléen | Appartenance au périmètre météo comparable |

Une centrale doit apparaître une seule fois dans cette table.

## 4. `eex_monthly_base_2027_snapshot_2026-08-24.parquet`

Forwards mensuels utilisés pour le recalage du spot.

| Colonne | Type attendu | Description |
|---|---|---|
| `delivery_start` | date | Début de livraison |
| `delivery_end` | date | Fin de livraison |
| `forward_base_eur_mwh` | réel | Forward mensuel base |
| `base_status` | chaîne | Statut ou provenance de la valeur |
| `target_month` | entier | Mois de livraison, de 1 à 12 |
| `snapshot_date` | date | Date du snapshot de marché |

Le runner exige exactement une valeur `forward_base_eur_mwh` pour chacun des douze mois.

## 5. `load_factor_group_quality.parquet`

Audit de qualité des facteurs de charge historiques.

| Colonne | Description |
|---|---|
| `group_id` | Groupe technologie-région |
| `technology` | Technologie |
| `region_code` | Code régional |
| `region_name` | Nom régional |
| `n_prms` | Nombre de PRM dans le groupe |
| `nominal_capacity_mw` | Puissance nominale du groupe |
| `hours_expected` | Nombre d’heures attendues |
| `median_coverage` | Couverture médiane |
| `share_hours_ge_80` | Part des heures avec au moins 80 % de couverture |
| `n_lf_above_1` | Nombre de facteurs de charge bruts supérieurs à 1 |

Cette table accompagne les résultats régionaux afin d’éviter de présenter une estimation fragile comme une estimation robuste.

## 6. `load_factor_imputation_summary.parquet`

Résumé des imputations appliquées aux groupes incomplets.

| Colonne | Description |
|---|---|
| `group_id` | Groupe technologie-région |
| `technology` | Technologie |
| `region_name` | Région |
| `n_prms` | Nombre de PRM |
| `nominal_capacity_mw` | Puissance nominale |
| `n_imputed` | Nombre d’heures imputées |
| `imputed_share` | Part des heures imputées |

La classification principale est :

| `imputed_share` | `quality_flag` |
|---:|---|
| ≤ 10 % | `robust` |
| > 10 % et ≤ 30 % | `moderate` |
| > 30 % | `fragile` |
| indisponible | `unknown` |

Les résultats nationaux et globaux reçoivent le statut `aggregated`.

## Contrôle avant exécution

Les six fichiers doivent être présents :

```powershell
Get-ChildItem .\data\processed -File
```

L’exécution complète se lance ensuite depuis la racine du projet :

```powershell
python run_benchmark.py
```