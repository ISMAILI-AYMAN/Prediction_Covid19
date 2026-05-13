# 📈 Traitement des données et modélisation géographique des cas COVID-19

Ce projet a pour objectif de modéliser la **propagation des cas de COVID-19** dans la région de Bruxelles-Capitale :
- en traitant les **données brutes**,
- en les **lissant** et en construisant des **features** pertinentes,
- en appliquant un **modèle Markov** et un **modèle supervisé**,
- en visualisant les résultats sous forme de **heatmaps** et **graphes**,
- en prolongeant l’approche par une **lecture mésoscopique** : redistribution **GBM × proximité spatiale** et détection de **corridors épidémiques** (scripts `11_*` et `12_*` à la racine du dépôt).

## ✉️ Auteurs
Projet réalisé par :
Benjamin, Nouredinne, Harry, Gasparino, Dimitry

## **Pipeline complet du projet**

1. **Téléchargement des données**
    - Téléchargement via **`requests`**
    - Conversion au **format {date : {commune : nb cas}}**

2. **Nettoyage et organisation des données**
    - **Filtrage** des communes de **Bruxelles**
    - Conversion des valeurs **"<5" → 1**

3. **Lissage**
    - Application du filtre **Savitzky-Golay** pour lisser les séries temporelles

4. **Modélisation**
    - Calcul des **prédictions Markov** par **multiplication matricielle**
    - Création des **features** (prédiction Markov scalée, total scalé, moyenne scalée, jour semaine, mois, weekend)
    - Passage dans le **modèle Gradient Boosting**

5. **Prédiction et évaluation des performances**
    - Inversion du log1p avec expm1
    - Comparaison avec les **cas réels** bruts

6. **Visualisation**
    - **Heatmaps** (prédictions, cas réels, différence)
    - Graphe **NetworkX** des erreurs absolues moyennes

7. **Distribution mésoscopique du GBM (proximité)**
    - Redistribution des prédictions **Gradient Boosting** sur les communes voisines via une matrice de **proximité** (Markov × poids géographiques).
    - Export des **matrices de distribution** par date (`gbm_proximity_distribution.json`) et figures associées (`visualizations_gbm_proximity/`).

8. **Corridors épidémiques**
    - Analyse de la **persistance temporelle** des flux, paramètres de **bifurcation** (β, μ) et identification des axes de propagation.
    - Sorties JSON d’analyse / résumés / persistance / métriques et visualisations **Plotly** (`visualizations_corridors/`).


## **Visualisations produites**

  - **Heatmap des prédictions**
    - nombre prédit de cas par commune et par date
  - **Heatmap des cas réels** : 
    - nombre réel de cas observés
  - **Heatmap des différences** : 
    - (prédiction - réel) pour identifier les erreurs
  - **Graphe des communes (NetworkX)** : 
    - nœuds : communes, taille et couleur selon l'erreur absolue moyenne
    - arêtes : poids de la matrice Markov (seuilés)
  - **Distribution GBM–proximité** : graphiques (Matplotlib / Seaborn / Plotly) générés par le script racine `11_gbm_proximity_distribution.py` dans `visualizations_gbm_proximity/`.
  - **Corridors épidémiques** : figures Plotly (flux, persistance, vues bipartites) dans `visualizations_corridors/` via `12_analyse_corridors_epidemiques.py`.


---

## **Données et fichiers utilisés**

| Fichier | Description |
|----------|-------------|
| `C0VID19BE_CASES_MUNI.json` | **Données brutes** téléchargées depuis l'API OpenDataSoft (Sciensano). |
| `C0VID19BE_CASES_MUNI_CLEAN.json` | **Données nettoyées** : uniquement les communes de Bruxelles + **conversion** des valeurs anonymisées. |
| `communes_lissage_savgol.json` | Données après **lissage Savitzky-Golay**. |
| `matrix_markov_models.json` | **Matrice** initiale **de transition Markov** entre communes. |
| `model_combinations/*` | Résultats des différents modèles **combinant Markov + Gradient Boosting**, avec différents poids. |
| `final_gb_model.joblib` | Modèle **Gradient Boosting** sélectionné comme final. |
| `scaler_pred.joblib`, `scaler_total.joblib`, `scaler_mean.joblib` | **Scalers pour normaliser** les features du modèle (entraînés sur les données d'entraînement). |
| `geographic_weights.json` | **Poids spatiaux** utilisés dans certains graphes ou expérimentations (non critique). |
| `gbm_proximity_distribution.json` | **Matrices de distribution** GBM redistribuées par proximité (Markov × géographie), par date — produit par `11_gbm_proximity_distribution.py`. |
| `corridors_epidemiques_analyse.json` | **Analyse détaillée** des corridors (séries, métriques) — `12_analyse_corridors_epidemiques.py`. |
| `corridors_epidemiques_resume.json` | **Résumé textuel / structuré** de l’analyse des corridors. |
| `corridors_identifies.json` | **Corridors identifiés** (agrégation des liens saillants). |
| `corridors_metriques.json` | **Métriques** associées aux corridors (scores, statistiques). |
| `corridors_persistance.json` | **Persistance temporelle** des éléments élevés dans les matrices de distribution. |

---


## **Structure détaillée des scripts**

### **Data_Processing/**

#### `01_data_downloader.py`
- Télécharge les données COVID-19 brutes depuis l'API OpenDataSoft (Sciensano).
- Filtre les données pour ne garder que les communes de Bruxelles.
- Convertit les valeurs **“<5” en 1**.
- Sauvegarde un fichier `C0VID19BE_CASES_MUNI.json` brut et un fichier nettoyé `C0VID19BE_CASES_MUNI_CLEAN.json`.

#### `02_data_loader.py`
- Charge les données **nettoyées au format JSON**.
- Organise les données sous forme de **dictionnaire** `{date: {commune: nb_cas}}`.
- Gère les erreurs ou **données manquantes**.
- Prépare les données pour les étapes suivantes (lissage / modélisation).

#### `03_savitzky_golay.py`
- Applique un filtre **Savitzky-Golay** (scipy.signal) pour lisser les séries temporelles.
- Paramètres ajustables : **taille de fenêtre, degré du polynôme**.
- Génère un fichier `communes_lissage_savgol.json` contenant les données lissées.
- Permet de réduire le bruit dans les séries de cas quotidiens.

#### `04_covid_visualizer.py`
- Génère des graphiques de séries temporelles :
  - **données brutes**
  - **données lissées**
- Produit des images (PNG) des **courbes par commune** (plusieurs communes par page).
- **Sauvegarde les graphiques** dans un dossier `visualizations/`.
- Utile pour inspecter visuellement les données avant modélisation.

# Module `geography/`
Ce dossier regroupe les scripts liés à :
- la modélisation géographique des cas COVID-19 à Bruxelles,
- la prédiction des cas avec un **modèle Markov** et un **modèle intégré Markov + Gradient Boosting**,
- la visualisation des résultats sous forme de **heatmaps et graphes**.

### `05_dijkstra.py`

- Implémenter l’**algorithme de Dijkstra** via `networkx` pour modéliser les distances / poids géographiques entre communes.
- Générer une **matrice des plus courts chemins entre communes**. Ce qui est particulièrement utile pour des **poids géographiques** et pour enrichir la **matrice Markov**.

**Entrée**
- **JSON des poids géographiques** (ou construit dynamiquement via un graphe des communes).

**Sortie**
- Matrice des **distances minimales entre toutes les paires de communes** (ex. fichier JSON ou impression console).

### `06_previsions_selon_le_modele_markov.py`

- Appliquer la **matrice de transition Markov** aux cas actuels pour générer des prédictions brutes pour la date suivante.
- Calculer la **répartition** attendue **des cas** dans chaque commune en fonction des interactions spatiales.

**Entrée**
- `communes_lissage_savgol.json` : **cas lissés** par date et commune
- `matrix_markov_models.json` : **matrice de transition** entre communes

**Sortie**
- dict: **Dictionnaire** contenant pour chaque date prédite :
    - **prediction** : liste des valeurs prédites arrondies.
    - **valeurs_reelles** : liste des valeurs réelles si disponibles, sinon None


### `07_prediction.py`

- Combiner les prédictions Markov avec des features temporelles :
  - **total des cas** prévus
  - **moyenne des cas** prévus
  - **jour de la semaine**
  - indicateur **weekend**
  - **mois**

- Appliquer les **scalers pour normaliser**
    - **prédicteur**
    - **total**
    - **moyenne**

- Passer les **features au modèle Gradient Boosting**
    **→** produire les **prédictions finales**

**Entrée :**
- **Prédictions Markov** (prédicteur, total, moyenne)
- Données temporelles extraites des **dates**
- **Scalers** (joblib)
- Modèle **Gradient Boosting** (joblib)

**Sortie :**
- DataFrame avec :
  - colonnes des **features normalisées**
  - **prédiction finale** (après inversion du log1p)
- **Sauvegarde** des prédictions dans un fichier **JSON**


### `08_markov_sklearn_integration.py`

- **Intégrer les prédictions Markov** dans un **pipeline scikit-learn** pour :
  - permettre un **entraînement supervisé** combinant **Markov + Gradient Boosting**.
  - rendre les prédictions Markov **utilisables** comme input dans une cross-validation ou **en recherche d’hyperparamètres**.

**Entrée**
- Matrice Markov.
- Données lissées ou brutes pour construire les features.

**Sortie**
- **Objet scikit-learn compatible** avec un pipeline ou matrice de features **prête à être utilisée dans** un GridSearchCV / **BayesSearchCV**.


### `09_interroger_modele_de_prediction_integre.py`

- Charger un **modèle final déjà entraîné** (Gradient Boosting **intégré aux features Markov**).
- Charger les **scalers** utilisés à l’entraînement.
- Préparer un jeu de features pour une date et une commune cible.
- Interroger le modèle et produire une prédiction du nombre de cas.

**Entrée**
- **Date et commune** cible
- Données **lissées**
- **Matrice Markov**
- **Scalers** (joblib)
- **Modèle final** (joblib)

**Sortie**
- **Prédiction du nombre de cas** pour la commune à la date donnée (float)


##  Visualisation et analyse géographique

### `10_heatmap_markov_vs_reel.py`

- Générer des **heatmaps** comparant :
  - les **prédictions** issues **uniquement** du modèle **Markov**,
  - les **cas réels** bruts (non lissés),
  - la **différence** entre les **prédictions Markov et les cas réels**.
- Offrir une **visualisation** pour évaluer la capacité du modèle Markov à capturer la dynamique des cas sans l’ajout du Gradient Boosting.

**Comment ça fonctionne ?**
- Pour chaque commune et chaque date sélectionnée (ex : mars 2023), le script :
  - calcule la **prédiction Markov par multiplication matricielle** entre la matrice de transition et les cas actuels,
  - construit des **matrices pour les prédictions**, les cas réels et leurs différences,
  - produit des **heatmaps** côte à côte :
    - **prédiction Markov**
    - **cas réel**
    - **différence**

**Entrées**
- Matrice Markov : fichier JSON ou NumPy array.
- Cas réels bruts : fichier JSON nettoyé.
- Données lissées pour les cas actuels (facultatif, selon prédiction Markov).

**Sorties**
- Figures sous forme de **heatmaps (Seaborn / Matplotlib)**.
- Fichiers images **PNG sauvegardés** dans un dossier type `visualizations/`.
- Affichage des figures pour inspection directe.


### `11_heatmap_modele_de_prediction_integre_vs_reel.py`

- Générer des **heatmaps** comparant :
  - les **prédictions finales** : **Markov + Gradient Boosting**,
  - les **cas réels bruts**,
  - la **différence** entre **prédiction finale et cas réels**.
- Visualiser la précision des **prédictions grâce au modèle supervisé**.

**Comment ça fonctionne ?**
- Le script :
  - applique les **scalers pré-entraînés** sur les features issus du **Markov**,
  - interroge le modèle **Gradient Boosting pré-entraîné**,
  - transforme les prédictions (inverse du log1p via expm1),
  - génère des **matrices de valeurs** pivotées (commune x date),
  - produit des **heatmaps** :
    - prédiction finale,
    - cas réel,
    - différence (prédiction - réel).

**Entrées**
- Prédictions **Markov et features** dérivés.
- **Scalers** (`joblib`).
- **Modèle final** (`joblib`).
- Cas réels bruts (JSON nettoyé).

**Sorties**
- **Heatmaps Matplotlib/Seaborn sauvegardées** (PNG).
- **Visualisation** des différences entre **modèle et réalité**.



### `12_graphe_de_relation_de_l_erreur_entre_prediction_et_reel.py`

- Produire un graphe orienté (via `NetworkX`) représentant :
  - les communes (nœuds),
  - les **erreurs absolues moyennes** (MAE) des prédictions (taille et couleur des nœuds),
  - les **relations spatiales entre communes** (arêtes, pondérées par la matrice Markov).

**Comment ça fonctionne ?**
- Calcule l’erreur absolue moyenne pour chaque commune sur une période donnée.
- Crée un graphe :
  - **nœuds** : **communes**, avec taille proportionnelle à l'erreur,
  - **arêtes** : **transitions Markov significatives** (poids > seuil, pas de boucles sur soi-même).
- Génère une visualisation :
  - disposition via `spring_layout` pour une lisibilité optimale,
  - **coloration des nœuds selon erreur** (colormap viridis ou équivalent),
  - légende / barre de couleur pour l'erreur.

**Entrées**
- Prédictions finales.
- Cas réels bruts.
- Matrice Markov.

**Sorties**
- **Graphe NetworkX** affiché et/ou **sauvegardé en PNG**.
- Permet une analyse spatiale des erreurs du modèle.

> **Note de numérotation :** le script **`12_analyse_corridors_epidemiques.py`** se trouve à la **racine** du dépôt (analyse mésoscopique / corridors). Le fichier homonyme de visualisation dans `geography/` est **`geography/12_graphe_de_relation_de_l_erreur_entre_prediction_et_reel.py`** (graphe d’erreurs) : rôles différents.

### `11_gbm_proximity_distribution.py` *(racine du dépôt)*

- Charge le **modèle GBM final**, les **scalers**, les **données lissées** et la **matrice Markov** optimale (`X_MEILLEUR_MODELE`).
- Construit une **matrice de proximité** combinant transition Markov et **poids géographiques** (`geographic_weights.json`).
- **Redistribue** la prédiction GBM de chaque commune sur les voisins selon cette proximité ; produit une série de **matrices par date**.
- **Sorties principales :** `Data/gbm_proximity_distribution.json`, dossier **`visualizations_gbm_proximity/`** (figures Plotly / seaborn).

### `12_analyse_corridors_epidemiques.py` *(racine du dépôt)*

- S’appuie sur **`gbm_proximity_distribution.json`** pour identifier les **corridors épidémiques** (persistance sur *N* périodes consécutives, seuils dynamiques type percentile).
- Calcule les indicateurs de **bifurcation** (β, μ) au niveau global et local, conformément au pipeline mésoscopique décrit dans `Pipeline_Mesoscopique_Explicatif.md`.
- **Sorties principales :** `corridors_epidemiques_analyse.json`, `corridors_epidemiques_resume.json`, `corridors_persistance.json`, `corridors_identifies.json`, `corridors_metriques.json`, dossier **`visualizations_corridors/`** (Plotly ; export possible via **Kaleido**).

Tous ces scripts utilisent :
- **`pandas`** pour les manipulations,
- **`seaborn` / `matplotlib`** pour les heatmaps,
- **`plotly`** (et éventuellement **kaleido**) pour les visualisations interactives des scripts **11** et **12** à la racine,
- **`networkx`** pour les graphes,
- **`numpy`** pour les calculs.

Ils permettent de :
- fournir une évaluation visuelle des **performances des modèles**,
- relier spatialement les **erreurs et les prédictions**,
- produire des figures exploitables dans un rapport ou une présentation.

---

## **Installation**

### Prérequis
Python ≥ **3.10** recommandé pour les scripts d’analyse **corridors** et **distribution GBM** à la racine ; le reste du projet reste compatible avec des versions antérieures (≥ 3.7) selon l’environnement.

#### Packages nécessaires :
- numpy
- pandas
- geopandas
- matplotlib
- seaborn
- scipy
- scikit-learn
- scikit-optimize
- networkx
- requests
- joblib
- tqdm
- colorama
- plotly
- kaleido *(export image des figures Plotly, optionnel mais recommandé)*

# Liste des librairies utilisées et usages dans le projet

---

## `numpy`
- **Usage :** Manipulation des matrices de transition, calculs matriciels, normalisation, prédictions Markov.
- **Installation :** `pip install numpy`
  - [Calculs mathématiques avancés avec NumPy - Python4Games](https://www.python4games.fr/calculs-mathematiques-avances-avec-numpy/)
  - [Tableaux et calcul matriciel avec NumPy](https://courspython.com/tableaux-numpy.html)


## `pandas`
- **Usage :** Chargement et manipulation des données tabulaires COVID-19, transformation en matrices, prétraitement.
- **Installation :** `pip install pandas`
  - [30 commandes pandas pour manipuler les DataFrames](https://www.journaldufreenaute.fr/30-commandes-pandas-pour-manipuler-les-dataframes/)
  - [Convert a DataFrame to Matrix in Python (4 Methods)](https://pythonguides.com/convert-pandas-dataframe-to-numpy-array/)


## `geopandas`
- **Usage :** Gestion des données géographiques (GeoJSON), calcul des intersections de frontières, construction des graphes spatiaux.
- **Installation :** `pip install geopandas`
  - [Introduction aux données spatiales avec Geopandas](https://pythonds.linogaliana.fr/content/manipulation/03_geopandas_intro.html)
  - [Mini présentation de GeoPandas](https://juliedjidji.github.io/memocarto/geopandas.html)

## `matplotlib`
- **Usage :** Génération de courbes temporelles, heatmaps, cartes.
- **Installation :** `pip install matplotlib`
  - [Annotated heatmap](https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html)

## `seaborn`
- **Usage :** Visualisation avancée : heatmaps des matrices, comparaisons prédictions / réel.
- **Installation :** `pip install seaborn`
  - [Cartes thermiques de Seaborn : Guide de la visualisation des données](https://www.datacamp.com/fr/tutorial/seaborn-heatmaps)
  - [Diagrammes matriciels avec Seaborn](https://moncoachdata.com/tutos/diagrammes-matriciels/)

## `scipy`
- **Usage :** Filtre Savitzky-Golay (module `scipy.signal`), calculs numériques complémentaires.
- **Installation :** `pip install scipy`
  - [Lissage de vos données avec le filtre Savitzky-Golay et Python](https://python.19633.com/fr/Python/1001001183.html)
  - [Lisser les données en Python](https://www.delftstack.com/fr/howto/python/smooth-data-in-python/)

## `scikit-learn` (alias `sklearn`)
- **Usage :** Modèle Gradient Boosting, normalisation des données, évaluation des performances.
- **Installation :** `pip install scikit-learn`
  - [Gradient Boosting avec Scikit-Learn, XGBoost, LightGBM et CatBoost](https://ichi.pro/fr/gradient-boosting-avec-scikit-learn-xgboost-lightgbm-et-catboost-97733907108683)
  - [Les méthodes de normalisation avec Scikit-Learn](https://www.alliage-ad.com/tutoriels-python/les-methodes-de-normalisation/)
  - [Tutoriel sur normaliser, standardiser et redimensionner vos données](https://complex-systems-ai.com/analyse-des-donnees/normaliser-standardiser-redimensionner-vos-donnees/)
  - [Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html)

## `scikit-optimize` (alias `skopt`)
- **Usage :** Optimisation des hyperparamètres (Gradient Boosting).
- **Installation :** `pip install scikit-optimize`
  - [Scikit-Optimize pour le réglage des hyperparamètres dans l'apprentissage automatique](https://fr.python-3.com/?p=4507)
  - [Scikit-Optimize for Hyperparameter Tuning in Machine Learning](https://machinelearningmastery.com/scikit-optimize-for-hyperparameter-tuning-in-machine-learning/)

## `networkx`
- **Usage :** Création du graphe géographique des communes, calcul des plus courts chemins (Dijkstra).
- **Installation :** `pip install networkx`
  - [Théorie des graphes : la bibliothèque NetworkX](https://mmorancey.perso.math.cnrs.fr/TutorielPython_NetworkX.html)
  - [NetworkX : Théorie des graphes, fonctions de base et utilisation](https://datascientest.com/networkx-tout-savoir)

## `requests`
- **Usage :** Téléchargement des données via API Sciensano.
- **Installation :** `pip install requests`
  - [Comment démarrer avec la librairie Requests en Python ?](https://www.digitalocean.com/community/tutorials/how-to-get-started-with-the-requests-library-in-python-fr)
  - [Working with JSON in Python requests](https://datagy.io/python-requests-json/)

## `joblib`
- **Usage :** Sauvegarde et chargement des modèles/scalers (Gradient Boosting, normalisation).
- **Installation :** `pip install joblib`

  - [Utiliser joblib pour la sérialisation efficace de grands objets](https://datacraft.ovh/sous-section/utiliser-joblib-pour-la-serialisation-efficace-de-grands-objets/)
  - [Comment sauvegarder et réutiliser un modèle développé avec scikit learn en utilisant joblib ?](https://fr.moonbooks.org/Articles/Comment-sauvegarder-dans-un-fichier-un-modele-developpe-avec-scikit-learn-en-python-machine-learning-/)

## `tqdm`
- **Usage :** Affichage des barres de progression lors des boucles d’optimisation.
- **Installation :** `pip install tqdm`
  - [Comment créer une barre de progression de terminal Python à l'aide de tqdm ? ](https://python.19633.com/fr/Python/1001010177.html)

## `colorama`
- **Usage :** Affichage coloré dans le terminal (logs, état des traitements).
- **Installation :** `pip install colorama`
  - [Comprendre les techniques de coloration de texte du terminal Python ?](https://www.tempmail.us.com/fr/python/affichage-de-texte-colore-dans-le-terminal-python)

## `plotly`
- **Usage :** Visualisations interactives (distribution GBM–proximité, corridors épidémiques).
- **Installation :** `pip install plotly`

## `kaleido`
- **Usage :** Export PNG/SVG/PDF des figures Plotly (scripts racine 11 et 12).
- **Installation :** `pip install kaleido`

---