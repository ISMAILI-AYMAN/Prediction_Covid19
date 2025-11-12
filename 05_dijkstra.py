# -*- coding: utf-8 -*-

"""
05_dijkstra.py
--------------
Script principal pour le calcul et l’export de matrices de transition géographiques 
entre les communes de la région de Bruxelles-Capitale (modélisation type Markov spatial).

Ce script automatise toutes les étapes, du téléchargement des géométries officielles 
jusqu’à la génération et l’enregistrement d’une matrice de transition pondérée utilisable 
en modélisation, simulation ou analyse de mobilité.

Fonctionnalités principales :
    - Télécharge et charge le GeoJSON officiel des communes depuis l’open data Bruxelles.
    - Vérifie la correspondance entre noms internes du projet et noms du GeoJSON.
    - Calcule (ou recharge depuis un cache) les longueurs de frontières entre chaque paire 
      de communes adjacentes.
    - Calcule les poids géographiques (distance/influence, via Dijkstra pondéré) pour chaque 
      couple de communes.
    - Génère et exporte la matrice de transition markovienne normalisée (chaque ligne = somme 1).
    - Logging avancé, rapports d’erreur sur communes absentes, réutilisation des données en cache.

Objectif pédagogique :
    - Montrer à un·e débutant·e comment construire un pipeline complet, reproductible et 
      maintenable pour l’analyse spatiale, avec toutes les bonnes pratiques professionnelles :
        * gestion des imports, des logs, de la structure du projet ;
        * contrôle qualité (correspondances, logs, rapport d’erreurs) ;
        * architecture modulaire (division en fonctions réutilisables et modules séparés) ;
        * documentation structurée et exemples d’utilisation.

Prérequis :
    - Accès internet (pour télécharger le GeoJSON à jour).
    - Les fichiers/dictionnaires d’adjacence et de correspondance (cf. constantes.py).
    - Les dossiers "data/" existent ou peuvent être créés automatiquement.
    - Python 3.9+, packages `geopandas`, `logging`, et dépendances standards.

Workflow typique :
    1. Télécharge les données officielles (si besoin, sinon usage local possible).
    2. Vérifie et loggue la correspondance des noms (pour éviter toute perte de données).
    3. Calcule (ou recharge depuis cache) les longueurs de frontières.
    4. Génère le graphe et les poids de voisinage (modèle Dijkstra pondéré).
    5. Normalise la matrice des transitions (pour Markov spatial).
    6. Sauvegarde tous les résultats dans le dossier data/ avec logs détaillés.

Utilisation recommandée :
    - Lancer le script depuis la racine du projet avec :
        $ python 05_dijkstra.py
    - Adapte le script si tu changes la liste des communes, l’URL GeoJSON, ou la structure du projet.
    - Les fichiers produits sont prêts pour une utilisation dans la modélisation, la cartographie, 
      ou l’analyse des mobilités spatiales.

Best Practice — Logger
----------------------
    - Un logger unique est configuré au lancement, et transmis à tous les modules/fonctions.
    - Cela centralise la gestion des logs (info, warning, erreurs fatales) et permet une 
      personnalisation (sortie console, fichier, etc.).
    - En cas d’erreur critique (ex : GeoJSON inaccessible), le script s’arrête et loggue clairement 
      la cause du problème.

Limitation :
    - Spécifique à la région Bruxelles-Capitale (adapter les noms/adjacences pour d’autres régions).
    - Nécessite l’accès internet pour la première exécution (puis usage du cache).
    - Si le GeoJSON évolue (changement de noms/colonnes), adapter le parsing ou la table de 
      correspondance.
    - Ne gère pas les interruptions partielles du workflow : chaque étape est indépendante.

Conseils de maintenance :
    - Conservez tous les logs et fichiers générés pour traçabilité et reproductibilité.
    - Modifiez ce script pour toute nouvelle région/année, en adaptant les constantes et l’URL source.
    - Ajoutez/modifiez les vérifications ou les exports selon vos besoins (par exemple, CSV au lieu de JSON).

Auteur   : Harry FRANCIS (2025)
Contact  : github.com/documentHarry

Version  : 1.0.0 (2025-07-24)
License  : Code source ouvert à la consultation.
           Toute redistribution ou usage commercial est soumis à accord écrit préalable.
"""


__version__ = "1.0.0"

# Compatible Python 3.9+ | Typage PEP 484 | PEP 257 | PEP 8 (88 caractères max)

# --- Librairies standards
import os
import logging
from typing import FrozenSet, Any, Optional, Dict

# --- Librairies tiers
import geopandas

# --- Modules locaux
from constantes import COMMUNES, ADJACENTES, CORRESPONDANCE_NOMS
from utils_io import (
    telecharger_json_depuis_url, charger_json, sauvegarder_json, creer_dossier_si_absent
)
from utils_log import configurer_logging, log_debut_fin
from utils_geojson import (calculer_longueurs_frontieres, log_communes_non_trouvees,
    log_resume_frontieres)
from constantes import deserialiser_frontieres_dict

from geograph import GeoGraph, controle_voisin_min

TRANSITION_MATRIX: str = "matrice_de_transition"
METADATA: str = "metadata"

#: Niveau du logging (INFO, DEBUG, WARNING...).
NIVEAU_LOG: int = logging.INFO
logger: logging.Logger = configurer_logging(NIVEAU_LOG, "05_dijkstra")


@log_debut_fin(logger)
def charger_et_controler_geojson(
        url_geojson: str, communes: list[str], correspondances: dict[str, str]
        ) -> geopandas.GeoDataFrame:
    """
    Télécharge le GeoJSON des communes, le charge en GeoDataFrame et vérifie la
    présence de toutes les communes attendues.

    Args:
        url_geojson (str): URL du fichier GeoJSON à télécharger.
        communes (list[str]): Liste des communes à contrôler (noms internes du projet).
        correspondances (dict[str, str]): Table correspondance noms internes/noms GeoJSON.
        logger (logging.Logger): Logger pour journaliser chaque étape.

    Returns:
        geopandas.GeoDataFrame: DataFrame géographique des communes chargées.

    Raises:
        SystemExit: Si le téléchargement ou le parsing du GeoJSON échoue.

    Note:
        - Loggue les colonnes et noms de communes détectés dans le GeoJSON.
        - Loggue les communes manquantes avec log_communes_non_trouvees.
        - S'arrête si le GeoJSON ne peut pas être téléchargé.

    Example:
        >>> df = charger_et_controler_geojson(
        ...   "https://exemple.com/mon.geojson",
        ...   ["Ixelles", "Schaerbeek"],
        ...   {"Ixelles": "Ixelles", "Schaerbeek": "Schaerbeek"},
        ...   logger
        ... )
        >>> "Ixelles" in df["name_fr"].values
        True

    Étapes:
        1. Télécharge le GeoJSON avec gestion des erreurs.
        2. Transforme les features en GeoDataFrame.
        3. Affiche les colonnes et noms détectés.
        4. Contrôle et loggue les communes manquantes.

    Tips:
        - À utiliser pour tout nouveau territoire avec adaptation de `communes` et
          `correspondances`.
    """
    geojson_data: Optional[dict[str, Any]] = telecharger_json_depuis_url(
        url_geojson, logger, nbr_reessais=3, pause_reessai=10, timeout=60
    )

    if geojson_data is None:
        logger.critical("Impossible de télécharger le GeoJSON, arrêt du script.",
                        exc_info = True)
        raise SystemExit(1)
    
    dataframe: geopandas.GeoDataFrame = geopandas.GeoDataFrame.from_features(
        geojson_data["features"]
    )

    logger.info(f"{len(dataframe)} communes chargées depuis le GeoJSON.")
    logger.info(f"Colonnes disponibles : {list(dataframe.columns)}")
    logger.info(f"Noms FR disponibles : {sorted(dataframe['name_fr'].tolist())}")
    return dataframe


@log_debut_fin(logger)
def charger_ou_calculer_frontieres(
        emplacement_cache: str, communes: list[str], adjacentes: dict[str, list[str]], 
        dataframe: geopandas.GeoDataFrame, correspondances: dict[str, str], 
        ecramement_du_fichier = True
        ) -> dict[FrozenSet[str], float]:
    """
    Charge ou calcule les longueurs de frontières entre toutes les communes adjacentes.
    Utilise un cache JSON pour accélérer les exécutions futures.

    Args:
        emplacement_cache (str): Emplacement du fichier JSON de cache.
        communes (list[str]): Liste des communes à traiter.
        adjacentes (dict[str, list[str]]): Dictionnaire d'adjacence (voisinage).
        dataframe (geopandas.GeoDataFrame): Table géographique avec toutes les communes.
        correspondances (dict[str, str]): Table de correspondance des noms.

    Returns:
        dict[FrozenSet[str], float]: Dictionnaire de longueurs des frontières (mètres).

    Note:
        - Recharge le cache si existant, sinon recalcule tout.
        - Sérialise les clés pour JSON (listes converties en str).
        - Loggue la source (cache ou calcul) et sauvegarde si besoin.

    Example:
        >>> frontieres = charger_ou_calculer_frontieres(
        ...   "data/frontieres_longueurs.json",
        ...   ["Ixelles", "Schaerbeek"],
        ...   {"Ixelles": ["Schaerbeek"]},
        ...   dataframe,
        ...   {"Ixelles": "Ixelles", "Schaerbeek": "Schaerbeek"},
        ...   logger
        ... )
        >>> any("Ixelles" in pair for fs in frontieres.keys() for pair in fs)
        True

    Étapes:
        1. Tente de charger le cache si le fichier existe.
        2. Si absent, calcule avec calculer_longueurs_frontieres, puis sauvegarde.
        3. Retourne le dictionnaire des longueurs.
    """
    if os.path.exists(emplacement_cache):
        cache_json: dict[str, Any] = charger_json(emplacement_cache, logger)
        frontieres = deserialiser_frontieres_dict(cache_json, logger)
        logger.info("Longueurs de frontières chargées depuis le cache.")
    else:
        frontieres = calculer_longueurs_frontieres(
            communes, adjacentes, dataframe, correspondances, logger
        )
        dict_a_sauvegarder: dict[str, float] = {
            str(list(fs)): v for fs, v in frontieres.items()
        }
        sauvegarder_json(dict_a_sauvegarder, emplacement_cache, logger = logger,
                         ecrasement = ecramement_du_fichier)
        logger.info("Longueurs de frontières sauvegardées dans le cache.")
    return frontieres


@log_debut_fin(logger)
def generer_matrice_markov(
        poids_geographiques: Dict[str, Dict[str, float]]
        ) -> Dict[str, Dict[str, float]]:
    """
    Crée une matrice de transition markovienne normalisée à partir des poids
    géographiques de voisinage entre communes.

    Args:
        poids_geographiques (dict[str, dict[str, float]]): Poids (ex : longueurs, influences)
          entre chaque commune et ses voisines.

    Returns:
        dict[str, dict[str, float]]: Matrice normalisée. Pour chaque commune, la somme des
          poids vers ses voisins = 1 (ou 0 si isolée).

    Note:
        - Adaptée à toute modélisation markovienne, simulation de diffusion, etc.
        - Aucune division par zéro possible : ligne nulle si commune isolée.

    Example:
        >>> generer_matrice_markov({'A': {'B': 2.0, 'C': 1.0}})
        {'A': {'B': 0.666..., 'C': 0.333...}}

    Étapes:
        1. Pour chaque commune, somme les poids de ses voisins.
        2. Si la somme est >0, normalise ; sinon, ligne de zéros.
    """
    poids_normalises: Dict[str, Dict[str, float]] = {}
    for commune, voisins in poids_geographiques.items():
        somme_poids: float = sum(voisins.values())
        if somme_poids == 0:
            poids_normalises[commune] = {v: 0.0 for v in voisins}
        else:
            poids_normalises[commune] = {v: p / somme_poids for v, p in voisins.items()}
    return poids_normalises


@log_debut_fin(logger)
def main() -> None:
    """
    Pipeline principal pour la génération, la normalisation et la sauvegarde de la matrice
    de transition géographique Markov (pour Bruxelles-Capitale).

    Étapes:
        1. Télécharge et charge le GeoJSON officiel des communes.
        2. Vérifie la correspondance des noms internes/projet vs GeoJSON.
        3. Calcule ou recharge les longueurs de frontières (avec cache JSON).
        4. Calcule les poids de voisinage (influence géographique pondérée).
        5. Génère la matrice markovienne normalisée.
        6. Sauvegarde la matrice et les métadonnées au format JSON.

    Args:
        Aucun argument (tout est paramétré par les constantes/imports).

    Returns:
        None. Cette fonction crée ou met à jour des fichiers dans le dossier "data/".

    Raises:
        SystemExit: Arrêt si téléchargement ou parsing du GeoJSON impossible.

    Note:
        - Log toutes les étapes clés, erreurs, conseils de maintenance.
        - Sauvegarde un rapport d’erreur si des communes sont absentes du GeoJSON.
        - Résultat compatible modélisation Markov, analyse de graphes, etc.

    Example:
        >>> python 05_dijkstra.py
        (logs) Fichier data/matrix_markov_models.json généré.
    """

    logger.info("=== Début du calcul des frontières intercommunales ===")

    url_geojson: str = (
        "https://opendata.bruxelles.be/api/explore/v2.1/catalog/datasets/"
        "limites-administratives-des-communes-en-region-de-bruxelles-capitale/"
        "exports/geojson"
    )
    emplacement_data: str = "data"
    creer_dossier_si_absent(emplacement_data, logger)
    FRONTIERES_PATH: str = os.path.join(emplacement_data, "frontieres_longueurs.json")

    # 1. Chargement et contrôle des géométries
    dataframe: geopandas.GeoDataFrame = charger_et_controler_geojson(
        url_geojson, COMMUNES, CORRESPONDANCE_NOMS,
    )

    rapport_erreur: str = "data/communes_absentes_geojson.txt"
    log_communes_non_trouvees(
        COMMUNES, dataframe, CORRESPONDANCE_NOMS, logger,
        emplacement_rapport_d_erreurs=rapport_erreur,
    )

    # 2. Chargement ou calcul des longueurs de frontières
    frontieres: dict[FrozenSet[str], float] = charger_ou_calculer_frontieres(
        FRONTIERES_PATH, COMMUNES, ADJACENTES, dataframe, CORRESPONDANCE_NOMS,
        ecramement_du_fichier = True
    )

    log_resume_frontieres(frontieres, logger)
    controle_voisin_min(COMMUNES, ADJACENTES, logger)

    # 3. Calcul du graphe et des poids géographiques
    graph: GeoGraph = GeoGraph(COMMUNES, ADJACENTES, frontieres, logger)
    graph.afficher_les_infos_sur_communes_adjacentes()
    poids_geographiques: Dict[str, Dict[str, float]] = (
        graph.calculer_tous_les_poids_geographiques(epsilon=0.1)
    )
    graph.sauvegarder_les_poids_geographiques(poids_geographiques, logger = logger)

    # 4. Génération de la matrice markovienne normalisée
    poids_normalises: Dict[str, Dict[str, float]] = generer_matrice_markov(
        poids_geographiques
    )

    # 5. Sauvegarde de la matrice markovienne
    poids_normalises_dico: dict[str, Any] = {
        METADATA: {
            "create_at": "2025-01-19",
            "communes_count": len(graph.communes),
        },
        TRANSITION_MATRIX: poids_normalises,
    }
    emplacement_sauvegarde: str = "data/matrix_markov_models.json"
    sauvegarder_json(poids_normalises_dico, emplacement_sauvegarde, logger = logger,
                     ecrasement = True)
    logger.info(f"Fichier {emplacement_sauvegarde} généré.")


if __name__ == "__main__":
    main()
