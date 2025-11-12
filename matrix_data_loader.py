# -*- coding: utf-8 -*-

"""
matrix_data_loader.py
---------------------
Module pour le chargement et la préparation des données matricielles
utilisées dans la modélisation de la propagation COVID par chaînes de Markov.

Fonctions :
    - Lecture d’un fichier JSON et extraction des matrices X_t, X_t1.
    - Construction de la matrice de poids géographiques selon l’ordre des communes.
    - Vérification de la cohérence des listes de communes entre fichiers.
    - Extraction des listes de communes et de dates pour contrôle ou test.

Dépendances :
    - numpy, utils (pour charger_json)

Auteur   : Harry FRANCIS (2025)
Contact  : github.com/documentHarry

License  : Code source ouvert à la consultation.
           Toute redistribution ou usage commercial est soumis à accord écrit préalable.

Version  : 1.0.0 (2025-07-23)
"""


__version__ = "1.0.0"

# Compatible Python 3.9+ | Typage PEP 484 | PEP 257 | PEP 8 (88 caractères max)

# --- Librairies standards
import logging

# --- Librairies tiers
import numpy as np

# --- Modules locaux
from utils_io import charger_json
from utils_log import log_debut_fin_logger_dynamique


@log_debut_fin_logger_dynamique("logger")
def preparer_matrices_markov_depuis_json(
        emplacement: str, logger: logging.Logger
        ) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Lit un fichier JSON de données COVID et prépare les matrices nécessaires à 
    l’entraînement d’un modèle Markov : cas à t, cas à t+1, liste des communes 
    et liste des dates.

    Args:
        emplacement (str): Chemin vers le fichier JSON à lire.
        logger (logging.Logger): Logger utilisé pour tracer les étapes et résultats.

    Returns:
        tuple:
            - X_t (np.ndarray): Matrice des cas à t, taille (n_communes, n_dates-1).
            - X_t1 (np.ndarray): Matrice des cas à t+1 (même forme, décalée d’un jour).
            - communes (list[str]): Liste ordonnée des noms de communes.
            - dates (list[str]): Liste triée de toutes les dates présentes.

    Note:
        - Si une commune n’existe pas à une date, le nombre de cas est mis à 0.
        - Les matrices X_t et X_t1 sont alignées pour permettre la modélisation
          Markov.
        - Le logger indique le nombre de communes et de dates détectées.

    Example:
        >>> X_t, X_t1, communes, dates = preparer_matrices_markov_depuis_json("data.json", logger)
        >>> logger.info(f"Shape X_t: {X_t.shape}, communes: {len(communes)}, dates: {len(dates)}")

    Étapes:
        1. Charge le JSON des données COVID par date et commune.
        2. Rassemble toutes les communes, trie les noms.
        3. Trie toutes les dates.
        4. Construit la matrice X (communes x dates) avec zéros si données absentes.
        5. Décale X pour obtenir X_t et X_t1 (prêt pour apprentissage Markov).

    Tips:
        - Vérifier la cohérence des communes avant de mélanger plusieurs fichiers.
        - Cette fonction prépare la base de tout entraînement Markov.

    Utilisation:
        Utiliser pour tout pipeline de modélisation, en amont de la construction 
        d’un modèle Markov ou d’un test statistique.

    Limitation:
        - Suppose que le JSON est au format {date: {commune: valeur}}.
        - Ne gère ni doublons, ni valeurs manquantes autres que zéro.

    See also:
        - charger_matrice_poids_geographiques
        - extraire_communes_et_dates_depuis_json
    """
    donnees_geo = charger_json(emplacement, logger)
    communes = set()
    for jour in donnees_geo.values():
        communes.update(jour.keys())
    communes = sorted(communes)
    dates = sorted(donnees_geo.keys())
    nbr_communes, n_dates = len(communes), len(dates)
    logger.debug(f"Nombre de communes détectées: {nbr_communes}")
    logger.debug(f"Nombre de dates détectées: {n_dates}")
    X = np.zeros((nbr_communes, n_dates))
    for index_date, d in enumerate(dates):
        for index_commune, c in enumerate(communes):
            X[index_commune, index_date] = donnees_geo[d].get(c, 0)
    X_t = X[:, :-1]
    X_t1 = X[:, 1:]
    return X_t, X_t1, communes, dates


@log_debut_fin_logger_dynamique("logger")
def charger_matrice_poids_geographiques(
        emplacement: str, communes: list[str], logger: logging.Logger) -> np.ndarray:
    """
    Charge une matrice de poids géographiques alignée sur la liste de communes
    fournie, à partir d’un fichier JSON. Utilisée pour pondérer l’influence
    spatiale dans un modèle Markov.

    Args:
        emplacement (str): Chemin du fichier JSON à lire.
        communes (list[str]): Liste ordonnée de communes à utiliser comme référence.
        logger (logging.Logger): Logger pour tracer le chargement et les erreurs.

    Returns:
        np.ndarray: Matrice carrée (n_communes x n_communes) de poids géographiques.

    Raises:
        ValueError: Si la liste de communes du JSON ne correspond pas à celle attendue.

    Note:
        - La cohérence des listes de communes (ordre, présence) est vérifiée.
        - Le logger trace la taille et l’alignement de la matrice.
        - Les poids sont extraits selon l’ordre fourni (très important !).

    Example:
        >>> poids = charger_matrice_poids_geographiques("geo.json", communes, logger)
        >>> logger.info(f"Taille : {poids.shape}")

    Étapes:
        1. Charge la matrice des poids depuis le JSON.
        2. Vérifie que les communes correspondent à la référence.
        3. Construit une matrice NumPy alignée sur la liste de communes.

    Tips:
        - Toujours passer la même liste de communes dans toutes les fonctions du projet.
        - Vérifier la cohérence avec la fonction dédiée avant tout calcul.

    Utilisation:
        À appeler avant tout calcul Markov géographiquement contraint.

    Limitation:
        - Les poids doivent exister pour chaque paire de communes attendue.

    See also:
        - preparer_matrices_markov_depuis_json
        - verifier_coherence_communes
    """
    logger.info(f"Chargement de la matrice de poids géographiques depuis {emplacement}")
    geo = charger_json(emplacement, logger)
    matrice = geo['matrice_de_transition']
    commune_geo = list(matrice.keys())
    # Vérification cohérence avec log
    try:
        verifier_coherence_communes(communes, commune_geo, logger = logger)
    except ValueError as exception:
        logger.error("Erreur de cohérence des communes", exc_info = True)
        raise
    poids = np.array([
        [matrice[commune_index_ligne][commune_index_colonne] 
            for commune_index_colonne in communes]
        for commune_index_ligne in communes
    ])
    logger.info(f"Matrice de poids géographiques chargée: shape={poids.shape}")
    return poids


@log_debut_fin_logger_dynamique("logger")
def verifier_coherence_communes(
        communes_data: list[str], communes_geo: list[str], logger: logging.Logger
        ) -> None:
    """
    Vérifie que deux listes de communes contiennent exactement les mêmes noms,
    pour éviter toute erreur d’alignement de matrice.  

    Args:
        communes_data (list[str]): Communes des données principales (ex: matrice X).
        communes_geo (list[str]): Communes des données géographiques à vérifier.
        logger (logging.Logger): Logger pour signaler l’état ou les discordances.

    Raises:
        ValueError: Si au moins une commune manque dans une des deux listes.

    Note:
        - Compare les deux listes comme des ensembles, sans tenir compte de l’ordre.
        - Affiche dans le logger la différence en cas d’incohérence.
        - Aucun retour en cas de succès (fonction de contrôle pure).

    Example:
        >>> verifier_coherence_communes(["Ixelles", "Uccle"], ["Uccle", "Ixelles"], logger)
        # OK, pas d’erreur

        >>> verifier_coherence_communes(["Ixelles", "Uccle"], ["Ixelles", "Schaerbeek"], logger)
        # ValueError, message détaillé dans les logs (logger.error)


    Étapes:
        1. Transforme chaque liste en ensemble.
        2. Compare les deux ensembles.
        3. En cas de différence, loggue et soulève une erreur.

    Tips:
        - Toujours utiliser avant toute opération combinant matrices de plusieurs sources.
        - Permet d’anticiper des erreurs “silencieuses” de résultats.

    Utilisation:
        Fonction de contrôle à appeler systématiquement après import ou jointure.

    Limitation:
        - Ne vérifie que la présence, pas l’ordre des communes.

    See also:
        - charger_matrice_poids_geographiques
    """
    set_data, set_geo = set(communes_data), set(communes_geo)
    diff_data = set_data - set_geo
    diff_geo = set_geo - set_data
    if diff_data or diff_geo:
        msg = (
            f"Discordance communes !\n"
            f"Data : {set_data}\nGeo : {set_geo}\n"
            f"Diff Data not in Geo: {diff_data}\n"
            f"Diff Geo not in Data: {diff_geo}"
        )
        logger.error(msg, exc_info = True)
        raise ValueError(msg)
    logger.debug("Cohérence des communes validée.")


@log_debut_fin_logger_dynamique("logger")
def extraire_communes_et_dates_depuis_json(
        emplacement: str, logger: logging.Logger) -> tuple[list[str], list[str]]:
    """
    Extrait rapidement la liste des communes et des dates à partir d’un fichier JSON
    {date: {commune: valeur}}, sans charger de matrices. Utile pour inspecter le
    contenu ou valider la structure.

    Args:
        emplacement (str): Chemin du fichier JSON à lire.
        logger (logging.Logger): Logger pour tracer le contenu extrait.

    Returns:
        tuple:
            - communes (list[str]): Liste ordonnée des communes présentes dans le fichier.
            - dates (list[str]): Liste triée des dates présentes dans le fichier.

    Note:
        - Extraction rapide, aucun calcul ou contrôle avancé effectué.
        - Pratique pour les scripts de test ou QA (contrôle de structure).

    Example:
        >>> communes, dates = extraire_communes_et_dates_depuis_json("data.json", logger)
        >>> logger.info(f"Communes : {communes[:3]}, Dates : {dates[:3]}")

    Étapes:
        1. Charge le JSON.
        2. Parcourt chaque jour pour extraire tous les noms de communes.
        3. Trie les listes de communes et de dates pour garantir l’ordre.

    Tips:
        - Permet de détecter rapidement si une commune ou une date manque.
        - Peut être intégrée dans un notebook pour valider un nouveau jeu de données.

    Utilisation:
        À appeler pour inspection ou contrôle de structure avant tout apprentissage.

    Limitation:
        - Suppose le format strict {date: {commune: valeur}}.

    See also:
        - preparer_matrices_markov_depuis_json
    """
    logger.info(f"Extraction rapide des communes/dates depuis {emplacement}")
    donnees = charger_json(emplacement, logger)
    communes = set()
    for jour in donnees.values():
        communes.update(jour.keys())
    dates = sorted(donnees.keys())
    communes_sorted = sorted(communes)
    logger.debug(f"{len(communes_sorted)} communes extraites, {len(dates)} dates.")
    return communes_sorted, dates
