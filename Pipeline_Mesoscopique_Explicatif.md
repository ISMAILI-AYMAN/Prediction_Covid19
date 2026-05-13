
# Pipeline de Modélisation Mésoscopique Épidémique – Explication Complète

## Étape 1 – Construction du graphe municipal
Chaque municipalité est représentée comme un nœud. Une arête est ajoutée entre deux nœuds si les municipalités partagent une frontière. 
Chaque arête est pondérée par la longueur de frontière partagée et la distance géographique (chemin le plus court via Dijkstra). 
Ce graphe spatial structure l’espace de diffusion épidémique.

## Étape 2 – Estimation de la matrice de transition P (Markov)
À partir des séries de cas lissées, une matrice de transition P est estimée via une régression des moindres carrés régularisée géographiquement. 
Chaque coefficient P_ij représente la proportion du signal en j transféré vers i à t+1.

## Étape 3 – Apportionnement spatial de la source GBM
La source S_i(t+1) apprise par le Gradient Boosting Machine (GBM) est redistribuée sur les voisins j via un poids w_ij,t proportionnel à P_ij * x_j(t).

### Exemple numérique :
Supposons : `x_j(t) = {10, 5, 15}`, `P_ij = {0.2, 0.3, 0.5}`, `S_i(t+1) = 4.0`

- Flux = `{2.0, 1.5, 7.5}`, Somme = `11.0`
- Poids w = `{0.182, 0.136, 0.682}`
- C_ij(t+1) = `w * 4.0 = {0.727, 0.545, 2.727}`

## Étape 4 – Calcul des paramètres de bifurcation
À partir de la matrice `C_ij(t)`, on calcule :
- µ_i = somme ligne de C
- β = maximum des sommes colonnes
- R_i = µ_i * β

### Exemple :
```
C = [[0.2, 0.4, 0.0],
     [0.1, 0.3, 0.6],
     [0.0, 0.1, 0.2]]
```

- µ = `[0.6, 1.0, 0.3]`
- β = `0.8`
- R = `[0.48, 0.80, 0.24]`

## Étape 5 – Identification des corridors épidémiques
Pour chaque instant t, on calcule :
- `F_ij(t) = P_ij * x_j(t)` (flux stationnaire)
- `C_ij(t)` : perturbation redistribuée
- `Corridors_ij(t) = F_ij(t) ∘ C_ij(t)` (produit Hadamard)

Les corridors identifiés par de fortes valeurs de Corridors_ij(t) révèlent les axes de propagation dominants.

## Étape 6 – Analyse temporelle et spatiale
On agrège les erreurs, flux et résidus pour identifier :
- Les municipalités à haute complexité épidémique
- Les périodes où le GBM est le plus actif
- La persistance des corridors dans le temps

---

## Étape 7 – Implémentation : distribution GBM selon la proximité (`11_gbm_proximity_distribution.py`)

À la **racine du dépôt**, ce script matérialise l’**apportionnement spatial** (étape 3) : il charge le modèle GBM final, les scalers, les cas lissés et la matrice Markov retenue, combine celle-ci avec les **poids géographiques**, puis **redistribue** jour après jour la prédiction GBM de chaque commune sur l’ensemble des communes pondéré par la proximité.

**Principales sorties (dossier `Data/` sauf mention contraire) :**
- `gbm_proximity_distribution.json` — dictionnaire *date → matrice* de distribution.
- Dossier `visualizations_gbm_proximity/` — figures d’exploration (Plotly / Matplotlib / Seaborn).

*Prérequis :* dépendances listées dans `requirements.txt` (notamment **plotly** ; **kaleido** pour l’export d’images).

## Étape 8 – Implémentation : analyse des corridors épidémiques (`12_analyse_corridors_epidemiques.py`)

Toujours à la **racine**, ce script consomme `gbm_proximity_distribution.json` et prolonge les étapes **5** et **6** du pipeline théorique : **persistance** des fortes contributions sur plusieurs pas de temps consécutifs, calcul des **β** et **μ** (bifurcation), synthèses et **visualisations Plotly** (vues bipartites, séries temporelles).

**Principales sorties :**
- `corridors_epidemiques_analyse.json`, `corridors_epidemiques_resume.json`
- `corridors_persistance.json`, `corridors_identifies.json`, `corridors_metriques.json`
- Dossier `visualizations_corridors/`

**Version documentée dans le code :** 1.1.0 (corrections persistance sur indices temporels, μ comme moyenne temporelle, visualisations bipartites).

---

*État du dépôt :* ces étapes 7–8 sont intégrées en **scripts Python à la racine** ; elles complètent les heatmaps et graphes du dossier `geography/` sans les remplacer.
