# -*- coding: utf-8 -*-

"""
utils_geojson.py
----------------
Outils de traitement géographique pour l’analyse des frontières entre communes.

Ce module fournit toutes les fonctions nécessaires pour :
    - Extraire la géométrie d’une commune (par son nom, avec correspondance flexible) depuis
      un GeoDataFrame.
    - Calculer la longueur de frontière entre toutes les paires de communes adjacentes 
      (nécessaire pour pondérer un graphe spatial ou modéliser des transitions).
    - Journaliser (logger) les problèmes de données, résumés des calculs, erreurs sur les
      communes absentes, ou frontières nulles.
    - Contrôler la complétude d’un GeoDataFrame vis-à-vis d’une liste de communes de référence.

Objectif pédagogique :
    - Permettre à un·e débutant·e de manipuler et d’exploiter des données géographiques,
      même sans connaissance avancée en SIG (système d’information géographique).
    - Montrer la bonne pratique d’une architecture modulaire et “testable” :
        * chaque fonction effectue une tâche précise, avec arguments typés et docstring claire ;
        * aucune configuration globale cachée (le logger est toujours passé en argument) ;
        * les erreurs sont logguées mais ne plantent pas le programme (robustesse et testabilité).

Prérequis :
    - GeoDataFrame contenant les géométries des communes (cf. fichier GeoJSON officiel).
    - Dictionnaire d’adjacence des communes (qui est voisin de qui).
    - Table de correspondance noms internes <-> noms du GeoDataFrame (flexibilité et robustesse).
    - Logger Python configuré (transmis en argument, jamais en variable globale).

Best Practice — Logger
----------------------
    - Toutes les fonctions de ce module attendent un objet `logger` en argument.
    - Cela garantit :
        * La centralisation de la configuration des logs (format, niveau, etc.)
        * L’absence de logger global ou caché dans le module (important pour tests/unitaires)
        * Une réutilisabilité totale (le script appelant contrôle la destination des logs, console ou fichier)
        * Un comportement conforme aux standards des librairies Python sérieuses.

Conseils :
    - Créez un seul logger dans votre script principal (`configurer_logging()`), et passez-le à chaque
      fonction/méthode qui en a besoin.
    - Tous les avertissements et messages d’information sont tracés via le logger.

Exemple d’utilisation :
    from geojson_tools import (
        calculer_longueurs_frontieres, log_communes_non_trouvees, log_resume_frontieres
    )
    frontieres = calculer_longueurs_frontieres([...], {...}, dataframe, {...}, logger)
    log_communes_non_trouvees([...], dataframe, {...}, logger)
    log_resume_frontieres(frontieres, logger)

Limitation :
    - Ce module ne gère pas les CRS/projections : le GeoDataFrame doit déjà être dans un système 
      métrique cohérent (ex : EPSG:31370 ou 3812 pour Bruxelles). Si ce n’est pas le cas, faites
      une reprojection AVANT d’appeler ces fonctions.
    - Les longueurs retournées sont dans l’unité du GeoDataFrame (souvent mètres).
    - Les noms de colonnes et les correspondances sont sensibles à l’orthographe et à la langue
      (adapter `NOM_COLONNE_COMMUNE` si besoin).

Pour aller plus loin :
    - Ce module s’intègre dans un pipeline de préparation de matrices de transition spatiale 
      (modélisation Markov, simulation de diffusion, analyses de voisinage, etc.).
    - Peut être adapté pour d’autres territoires ou d’autres entités (quartiers, arrondissements, etc.).

Auteur   : Harry FRANCIS (2025)
Contact  : github.com/documentHarry

Version  : 1.0.0 (2025-07-24)
License  : Code source ouvert à la consultation.
           Toute redistribution ou usage commercial est soumis à accord écrit préalable.
"""


__version__ = "1.0.0"

# Compatible Python 3.9+ | Typage PEP 484 | PEP 257 | PEP 8 (88 caractères max)

# --- Librairies standards
import logging
from typing import FrozenSet, Optional

# --- Librairies tiers
import geopandas
import shapely

# --- Modules locaux
from utils_io import sauvegarder_lignes_texte
from utils_log import log_debut_fin_logger_dynamique

# === Paramètres de configuration ===
NOM_COLONNE_COMMUNE = "name_fr"

# Alias pour les types de dictionnaires de frontières
Frontieresdict = dict[FrozenSet[str], float]


@log_debut_fin_logger_dynamique("logger")
def calculer_longueurs_frontieres(
        communes: list[str], adjacentes: dict[str, list[str]],
        dataframe: geopandas.GeoDataFrame, correspondance: dict[str, str],
        logger: logging.Logger) -> Frontieresdict:
    """
    Calcule la longueur de frontière pour chaque paire de communes adjacentes.
    Retourne un dictionnaire {frozenset({A, B}): longueur (float)}.

    Args:
        communes (list[str]): Liste des communes à traiter (noms internes).
        adjacentes (dict[str, list[str]]): Dictionnaire des voisins pour chaque commune.
        dataframe (geopandas.GeoDataFrame): Table des géométries de toutes les communes.
        correspondance (dict[str, str]): Correspondance noms internes <-> noms GeoDataFrame.
        logger (logging.Logger): Logger pour journaliser les étapes de l'export et anomalies.

    Returns:
        Frontieresdict: Dictionnaire des longueurs (en mètres ou unité du GeoDataFrame).
        Clé : frozenset de 2 communes. Valeur : longueur de frontière (float).

    Note:
        - Log un avertissement si une commune n'est pas trouvée dans le GeoJSON.
        - Les paires sont ordonnées par frozenset : "A/B" = "B/A".
        - Les longueurs nulles (0.0) indiquent frontière absente ou commune manquante.

    Example:
        >>> dico = calculer_longueurs_frontieres(
        ...     ["A", "B"], {"A": ["B"], "B": ["A"]}, dataframe,
        ...     {"A": "Anderlecht", "B": "Bruxelles"}, logger
        ... )
        >>> dico[frozenset({"A", "B"})]
        1012.44

    Étapes:
        1. Pour chaque commune et chacun de ses voisins :
            - Forme une paire non ordonnée (frozenset).
            - Ignore la paire si déjà calculée.
        2. Cherche la géométrie de chaque commune (via trouver_geometrie_commune).
        3. Calcule l’intersection et la longueur de la frontière.
        4. Si l’une des géométries est absente, log un warning et met la longueur à 0.
        5. Stocke le résultat dans le dictionnaire final.

    Tips:
        - Tous les avertissements et messages d’information sont tracés via le logger.
        - Adapté aux scripts de pondération de graphe spatial.

    Utilisation:
        À intégrer dans un pipeline de préparation de matrices de voisinage/transition.

    Limitation:
        - Le GeoDataFrame doit être dans un CRS métrique (EPSG:31370 ou 3812 conseillé).
        - Sensible à la correspondance des noms (bien configurer `correspondance`).

    See also:
        - trouver_geometrie_commune
        - log_resume_frontieres, log_communes_non_trouvees
    """
    longueur_commune_adj_dico: Frontieresdict = {}
    for commune, voisins in adjacentes.items():
        for voisin in voisins:
            communes_voisines = frozenset([commune, voisin])
            if communes_voisines in longueur_commune_adj_dico:
                continue
            geo_commune = trouver_geometrie_commune(
                dataframe, commune, correspondance, logger
            )
            geo_commune_adjacente = trouver_geometrie_commune(
                dataframe, voisin, correspondance, logger
            )
            if geo_commune is None or geo_commune_adjacente is None:
                logger.warning(
                    "[FRONTIERE] %s-%s absente du GeoJSON",
                    commune,
                    voisin,
                    extra={
                        "event": "frontiere_absente",
                        "commune": commune,
                        "voisin": voisin,
                    },
                )
                longueur_commune_adj_dico[communes_voisines] = 0.0
                continue
            intersection = geo_commune.intersection(geo_commune_adjacente)
            longueur_commune_adj = (
                intersection.length if not intersection.is_empty else 0.0
            )
            longueur_commune_adj_dico[communes_voisines] = longueur_commune_adj
    return longueur_commune_adj_dico


@log_debut_fin_logger_dynamique("logger")
def trouver_geometrie_commune(
        dataframe: geopandas.GeoDataFrame, commune: str,
        correspondance: dict[str, str], logger: logging.Logger,
        ) -> Optional[shapely.geometry.base.BaseGeometry]:
    """
    Recherche et retourne la géométrie d’une commune dans un GeoDataFrame, avec tolérance
    sur les noms (grâce à la table de correspondance).

    Args:
        dataframe (geopandas.GeoDataFrame): Table géographique contenant les géométries.
        commune (str): Nom interne de la commune recherchée.
        correspondance (dict[str, str]): Dictionnaire noms internes -> noms GeoDataFrame.
        logger (logging.Logger, optionnel): Logger pour warnings si la commune est absente.

    Returns:
        Optional[shapely.geometry.base.BaseGeometry]: La géométrie (Polygon/MultiPolygon), ou None
        si la commune n’est pas trouvée.

    Note:
        - La correspondance permet d’adapter à toutes les conventions de nommage (FR/NL).
        - Un warning est loggué si la commune est absente.

    Example:
        >>> geom = trouver_geometrie_commune(dataframe, "Anderlecht", correspondance, logger)

    Étapes:
        1. Cherche le nom réel via `correspondance`, sinon garde le nom donné.
        2. Sélectionne la ligne du GeoDataFrame correspondant à ce nom.
        3. Si aucune géométrie trouvée, log un avertissement et retourne None.
        4. Sinon, retourne la géométrie.

    Tips:
        - Fonction très robuste pour des scripts multi-langues ou multi-sources.
        - Peut servir pour tout découpage géographique équivalent.

    Utilisation:
        Appelée à chaque calcul de frontière (puisque l'ordre des communes n’a pas d’importance).

    Limitation:
        - La recherche se fait sur la colonne fixée par NOM_COLONNE_COMMUNE.
        - Les correspondances doivent être exhaustives.

    See also:
        - calculer_longueurs_frontieres
        - log_communes_non_trouvees
    """
    nom_recherche = correspondance.get(commune, commune.split(" (")[0])
    selection = dataframe[dataframe[NOM_COLONNE_COMMUNE] == nom_recherche]
    if selection.empty:
        if logger:
            logger.warning(
                f"Commune absente du GeoJSON : {commune} "
                f"(nom recherché : {nom_recherche})"
            )
        return None
    return selection.geometry.values[0]


@log_debut_fin_logger_dynamique("logger")
def log_communes_non_trouvees(
        communes: list[str], dataframe: geopandas.GeoDataFrame,
        correspondance: dict[str, str],logger: logging.Logger,
        emplacement_rapport_d_erreurs: str = None) -> None:
    """
    Loggue toutes les communes qui ne sont pas trouvées dans le GeoDataFrame (après correspondance).
    Écrit aussi un rapport texte détaillé si un emplacement est fourni.

    Args:
        communes (list[str]): Liste de toutes les communes attendues.
        dataframe (geopandas.GeoDataFrame): Table géographique de référence.
        correspondance (dict[str, str]): Table des correspondances pour les noms.
        logger (logging.Logger): Logger de suivi.
        emplacement_rapport_d_erreurs (str, optionnel): Fichier où écrire la liste (facultatif).

    Returns:
        None

    Note:
        - Utile pour auditer rapidement les oublis ou incohérences de découpage.
        - Les communes manquantes sont logguées et listées dans le rapport (si demandé).

    Example:
        >>> log_communes_non_trouvees(
        ...     ["Ixelles", "Woluwe-Saint-Pierre"], dataframe,
        ...     {"Ixelles": "Ixelles", "Woluwe-Saint-Pierre": "Woluwe-St-Pierre"},
        ...     logger, emplacement_rapport_d_erreurs="rapport_erreurs.txt"
        ... )

    Étapes:
        1. Pour chaque commune attendue, cherche sa présence (via correspondance) dans le GeoDataFrame.
        2. Regroupe toutes les absentes.
        3. Loggue la liste en warning.
        4. Si un fichier de rapport est demandé, écrit la liste dedans (tabulé).

    Tips:
        - Toujours exécuter ce contrôle avant toute analyse spatiale.
        - Génère un rapport facilement partageable avec d'autres équipes.

    Utilisation:
        Après import d’un GeoJSON ou d’un shapefile, pour valider la cohérence de la donnée.

    Limitation:
        - Ne corrige pas les erreurs : à faire manuellement dans le GeoJSON ou la table de correspondance.

    See also:
        - trouver_geometrie_commune
        - controle_communes_geojson
    """
    non_trouvees = []
    noms_geo = set(dataframe[NOM_COLONNE_COMMUNE])
    for commune in communes:
        nom_geo = correspondance.get(commune, commune)
        if nom_geo not in noms_geo:
            # Cas où le nom attendu et cherché sont différents
            if commune != nom_geo:
                ligne = f"{commune} → {nom_geo}"
            else:
                ligne = f"{commune} (absente du GeoJSON)"
            non_trouvees.append(ligne)
    if non_trouvees:
        for ligne in non_trouvees:
            logger.warning(ligne)
        if emplacement_rapport_d_erreurs:
            sauvegarder_lignes_texte([f"{ligne}\n" for ligne in non_trouvees],
                                     emplacement_rapport_d_erreurs, logger)
            logger.info(
                f"Rapport d’erreur sauvegardé : {emplacement_rapport_d_erreurs}"
            )


@log_debut_fin_logger_dynamique("logger")
def log_resume_frontieres(
        longueur_dict: Frontieresdict, logger: logging.Logger) -> None:
    """
    Loggue un résumé synthétique du dictionnaire des longueurs de frontières.

    Args:
        longueur_dict (Frontieresdict): Dictionnaire {frozenset({A, B}): longueur}.
        logger (logging.Logger): Logger pour afficher les infos.

    Returns:
        None

    Note:
        - Affiche le nombre total, le nombre de frontières nulles, et les 6 premiers exemples.
        - Facilite le debug ou la vérification du calcul.

    Example:
        >>> log_resume_frontieres(mon_dico_frontieres, logger)

    Étapes:
        1. Log le nombre total de paires.
        2. Log le nombre de frontières nulles (longueur 0.0).
        3. Log les 6 premières paires avec leurs longueurs.

    Tips:
        - Idéal pour contrôler rapidement une matrice spatiale après calcul.

    Utilisation:
        Appelée en toute fin de pipeline de calcul, avant visualisation ou export.

    Limitation:
        - Affichage tronqué si le dictionnaire est volumineux.

    See also:
        - calculer_longueurs_frontieres
    """
    logger.info(f"Nombre de frontières calculées : {len(longueur_dict)}")
    logger.info(
        f"{sum(1 for v in longueur_dict.values() 
               if v == 0.0)} "
        f"frontières nulles (0.0)"
    )
    # Seulement les 6 premières
    for communes_voisines, longueur_commune_adjacente in list(
        longueur_dict.items()
    )[:]:
        logger.info(f"{communes_voisines}: {longueur_commune_adjacente:.2f}")
    
    logger.info("... (affichage tronqué)")


@log_debut_fin_logger_dynamique("logger")
def controle_communes_geojson(
        communes: list[str], dataframe: geopandas.GeoDataFrame, logger: logging.Logger
        ) -> None:
    """
    Vérifie que toutes les communes attendues figurent bien dans le GeoDataFrame.

    Args:
        communes (list[str]): Liste des communes de référence (celles attendues).
        dataframe (geopandas.GeoDataFrame): Table géographique à contrôler.
        logger (logging.Logger): Logger utilisé pour afficher les absences éventuelles.

    Returns:
        None

    Note:
        - Loggue toutes les communes absentes : rapide pour repérer un oubli dans le GeoJSON.

    Example:
        >>> controle_communes_geojson(["A", "B"], dataframe, logger)

    Étapes:
        1. Parcourt la liste des communes attendues.
        2. Pour chaque commune, vérifie sa présence dans le GeoDataFrame (colonne `name_fr`).
        3. Loggue un warning si une ou plusieurs sont absentes.
        4. Sinon, loggue un succès.

    Tips:
        - À exécuter après tout chargement de fichier GeoJSON ou shapefile.

    Utilisation:
        Test d'intégrité de données spatiales, en amont de tout calcul de graphe.

    Limitation:
        - Sensible au nom exact utilisé dans le GeoDataFrame (attention aux majuscules/accents).

    See also:
        - log_communes_non_trouvees
    """
    absentes = [c for c in communes if c not in dataframe[NOM_COLONNE_COMMUNE].tolist()]
    if absentes:
        logger.warning(f"Communes absentes du GeoJSON : {absentes}")
    else:
        logger.info("Toutes les communes attendues sont présentes dans le GeoJSON.")


# -----------------------------------------------------------------------------
# __all__ : Contrôle des imports publics du module
# -----------------------------------------------------------------------------
#
# En Python, la variable spéciale __all__ permet de définir explicitement la
# Liste des objets (fonctions, classes, etc.) qui seront importés lors d’un
# "from geograph import *".
#
# Cela permet :
#   - de ne rendre publiques que les fonctions et classes destinées à l'usage externe,
#   - de masquer les fonctions ou variables internes,
#   - d'améliorer la clarté de l'API et de la documentation.
#
# Exemple :
#   __all__ = ["GeoGraph", "controle_voisin_min"]
#   => Seuls GeoGraph et controle_voisin_min seront accessibles via "import *".
#
# Note :
#   - Ce n’est pas obligatoire, mais recommandé dans les modules un peu sérieux.
#   - Les noms non listés ici restent accessibles via import direct (import geograph).
# -----------------------------------------------------------------------------

__all__ = [
    "calculer_longueurs_frontieres",
    "trouver_geometrie_commune",
    "log_communes_non_trouvees",
    "log_resume_frontieres",
    "controle_communes_geojson",
]
