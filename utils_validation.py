"""

utils_validation.py
--------------------------------



Auteur   : Harry FRANCIS (2025)
Contact  : github.com/documentHarry

License  : Code source ouvert à la consultation.
           Toute redistribution ou usage commercial est soumis à accord écrit préalable.

Version  : 1.0.0 (2025-07-24)
"""


import logging
from typing import Any

import numpy as np
import pandas as pd

from utils_io import creer_dossier_si_absent
from utils_log import log_debut_fin_logger_dynamique

@log_debut_fin_logger_dynamique("logger")
def validation_variables_et_donnees(
        COMMUNES: list[str], X_MEILLEUR_MODELE: np.ndarray,
        donnees_covid: dict[str, dict[str, Any]],
        donnees_lissees: dict[str, dict[str, Any]],
        EMPLACEMENT_DONNEES: str, logger: logging.Logger) -> None:
    """
    Vérifie la présence et la cohérence des variables essentielles et du dossier Data/.
    Lève une exception si une variable essentielle est manquante ou incohérente.

    Args:
        COMMUNES (list[str]): Liste des communes.
        X_MEILLEUR_MODELE (np.ndarray): Matrice de transition Markov.
        donnees_covid (dict): Données brutes Covid.
        donnees_lissees (dict): Données lissées.
        EMPLACEMENT_DONNEES (str): Chemin vers le dossier Data/.
        logger (logging.Logger): Pour journaliser les étapes et erreurs.

    Returns:
        None: Lève une ValueError si un argument clé est absent ou incohérent.

    Note:
        - Crée le dossier Data s’il n’existe pas, sans écraser de données.

    Example:
        >>> validation_variables_et_donnees(COMMUNES, X, dc, dl, DATA_PATH, logger)

    Étapes:
        1. Vérifie que chaque variable clé n’est pas None ou vide.
        2. Crée le dossier Data si besoin (sécurisé).

    Tips:
        - À appeler toujours en tout début de pipeline pour fiabiliser les traitements.

    Utilisation:
        Avant toute création de dataset ou split, pour garantir l’intégrité des inputs.

    Limitation:
        - Ne valide pas la structure interne des objets (ex : types de valeurs).
        - Suppose que les noms de colonnes sont cohérents partout.

    See also:
        - creer_dossier_si_absent pour la création sécurisée de dossier.
    """
    logger.info("Vérification des variables clés et des données")
    # Vérification des variables clés
    if COMMUNES is None or len(COMMUNES) == 0:
        raise ValueError("La liste COMMUNES est vide ou non initialisée.")
    if X_MEILLEUR_MODELE is None:
        raise ValueError("La matrice X_MEILLEUR_MODELE n'est pas initialisée.")
    if donnees_covid is None or donnees_lissees is None:
        raise ValueError("Les données Covid/lissées ne sont pas chargées.")
    # Vérification des dossiers
    creer_dossier_si_absent(EMPLACEMENT_DONNEES, logger)


@log_debut_fin_logger_dynamique("logger")
def validation_colonnes_dataframe(df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Vérifie la présence des colonnes Markov attendues dans le DataFrame fourni.
    Lève une erreur si une colonne attendue n’existe pas.

    Args:
        df (pd.DataFrame): DataFrame à vérifier.

    Returns:
        None: Lève KeyError si une colonne obligatoire manque.

    Note:
        - Les colonnes obligatoires sont predict_markov, total_markov, moyenne_markov.

    Example:
        >>> validation_colonnes_dataframe(df)

    Étapes:
        1. Parcourt la liste des colonnes obligatoires.
        2. Lève une KeyError si une colonne manque.

    Tips:
        - À utiliser avant toute transformation ou standardisation des features.

    Utilisation:
        Juste avant standardisation ou split train/test.

    Limitation:
        - Ne vérifie pas les types ou la qualité des valeurs.
        - À compléter si tu ajoutes de nouvelles features critiques.

    See also:
        - standardiser_variables pour la normalisation après validation.
    """
    logger.info("Vérification de l'existence des colonnes attendues dans le dataframe")
    colonnes_attendues = ["predict_markov", "total_markov", "moyenne_markov"]
    for col in colonnes_attendues:
        if col not in df.columns:
            raise KeyError(f"La colonne {col} manque dans le DataFrame !")
