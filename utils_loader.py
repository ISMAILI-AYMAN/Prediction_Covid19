# -*- coding: utf-8 -*-
"""
utils_loader.py
---------------
Module centralisé pour charger, mettre en cache et logguer les objets lourds (modèles,
jeux de données JSON, matrices…) du projet COVID. Toutes les fonctions reçoivent un logger
pour assurer un suivi complet des accès disques.

- Accès direct aux fichiers (pas de chargement à l’import)
- Un cache RAM pour éviter de relire le même fichier à chaque appel
- Toutes les dépendances sont explicites (constantes, utils_io, joblib…)

Auteur: Harry FRANCIS (2025)
"""

import os
from constantes import (
    EMPLACEMENT_MODELE_FINAL, EMPLACEMENT_SCALER_PREDICT,
    EMPLACEMENT_SCALER_TOTAL, EMPLACEMENT_SCALER_MOYENNE,
    EMPLACEMENT_DONNEES_COVID, EMPLACEMENT_DONNEES_LISSEES,
    EMPLACEMENT_MODELE_MARKOV, EMPLACEMENT_MEILLEUR_MODELE, Cles
)
from utils_io import charger_json
import joblib
import numpy as np
import logging


EMPLACEMENT_PAR_CLE = {
    Cles.DONNEES_COVID: (EMPLACEMENT_DONNEES_COVID, "json"),
    Cles.DONNEES_LISSEES: (EMPLACEMENT_DONNEES_LISSEES, "json"),
    Cles.X_MEILLEUR_MODELE: (EMPLACEMENT_MEILLEUR_MODELE, "json"),
    Cles.MODELE_FINAL: (EMPLACEMENT_MODELE_FINAL, "joblib"),
    Cles.SCALER_PREDICT: (EMPLACEMENT_SCALER_PREDICT, "joblib"),
    Cles.SCALER_TOTAL: (EMPLACEMENT_SCALER_TOTAL, "joblib"),
    Cles.SCALER_MOYENNE: (EMPLACEMENT_SCALER_MOYENNE, "joblib"),
}

# --- Cache centralisé interne (privé)
_CACHE = {}

# === Modèles/scalers ===

'''
def charger_avec_cache(
    cle: Cles, logger: logging.Logger, forcer_chargement: bool = False):
    """Charge et met en cache l'objet désigné par cle, selon son emplacement."""
    if cle not in EMPLACEMENT_PAR_CLE:
        raise ValueError(f"Clé inconnue : {cle}")
    if forcer_chargement or cle not in _CACHE:
        try:
            emplacement = EMPLACEMENT_PAR_CLE[cle]
            _CACHE[cle] = joblib.load(emplacement)
            logger.info(f"{cle.name} chargé depuis {emplacement}")
        except Exception as exc:
            logger.error(f"Échec chargement {cle.name} : {exc}", exc_info = True)
            raise
    return _CACHE[cle]
'''


def charger_avec_cache(cle_enum, logger):
    if cle_enum in _CACHE:
        logger.info(f"{cle_enum.name} chargé depuis la RAM (cache interne).")
        return _CACHE[cle_enum]

    chemin_fichier, type_fichier = EMPLACEMENT_PAR_CLE[cle_enum]
    if not os.path.exists(chemin_fichier):
        logger.warning(
            f"Le fichier {chemin_fichier} n'existe pas pour {cle_enum.name}. "
            "Un nouveau fichier sera créé si besoin (par la suite)."
        )
        raise FileNotFoundError(f"Fichier absent: {chemin_fichier}")

    try:
        if type_fichier == "joblib":
            objet = joblib.load(chemin_fichier)
        elif type_fichier == "json":
            objet = charger_json(chemin_fichier, logger)
        else:
            raise ValueError(f"Type de fichier non supporté: {type_fichier}")
        _CACHE[cle_enum] = objet
        logger.info(f"{cle_enum.name} chargé depuis {chemin_fichier}.")
        return objet
    except Exception as exception:
        logger.error(f"Erreur lors du chargement de {chemin_fichier}: {exception}", 
                     exc_info = True)
        raise



def obtenir_modele_final(logger: logging.Logger, forcer_chargement = False):
    """Charge et met en cache le modèle Gradient Boosting final."""
    cle = Cles.MODELE_FINAL
    if forcer_chargement or cle not in _CACHE:
        try:
            _CACHE[cle] = joblib.load(EMPLACEMENT_MODELE_FINAL)
            logger.info(f"MODELE_FINAL chargé depuis {EMPLACEMENT_MODELE_FINAL}")
        except Exception as exception:
            logger.error(f"Échec chargement MODELE_FINAL : {exception}", 
                         exc_info = True)
            raise
    return _CACHE[cle]

def obtenir_scaler_predict(logger: logging.Logger, forcer_chargement = False):
    if "SCALER_PREDICT" not in _CACHE:
        _CACHE["SCALER_PREDICT"] = joblib.load(EMPLACEMENT_SCALER_PREDICT)
        logger.info(f"SCALER_PREDICT chargé depuis {EMPLACEMENT_SCALER_PREDICT}")
    return _CACHE["SCALER_PREDICT"]

def obtenir_scaler_total(logger: logging.Logger, forcer_chargement = False):
    if "SCALER_TOTAL" not in _CACHE:
        _CACHE["SCALER_TOTAL"] = joblib.load(EMPLACEMENT_SCALER_TOTAL)
        logger.info(f"SCALER_TOTAL chargé depuis {EMPLACEMENT_SCALER_TOTAL}")
    return _CACHE["SCALER_TOTAL"]

def obtenir_scaler_moyenne(logger: logging.Logger, forcer_chargement = False):
    if "SCALER_MOYENNE" not in _CACHE:
        _CACHE["SCALER_MOYENNE"] = joblib.load(EMPLACEMENT_SCALER_MOYENNE)
        logger.info(f"SCALER_MOYENNE chargé depuis {EMPLACEMENT_SCALER_MOYENNE}")
    return _CACHE["SCALER_MOYENNE"]

# === Données JSON ===

def obtenir_donnees_covid(logger: logging.Logger, forcer_chargement = False):
    if "DONNEES_COVID" not in _CACHE:
        _CACHE["DONNEES_COVID"] = charger_json(EMPLACEMENT_DONNEES_COVID, logger)
        logger.info(f"Données COVID chargées depuis {EMPLACEMENT_DONNEES_COVID}")
    return _CACHE["DONNEES_COVID"]

def obtenir_donnees_lissees(logger: logging.Logger, forcer_chargement = False):
    if "DONNEES_LISSEES" not in _CACHE:
        _CACHE["DONNEES_LISSEES"] = charger_json(EMPLACEMENT_DONNEES_LISSEES, logger)
        logger.info(f"Données lissées chargées depuis {EMPLACEMENT_DONNEES_LISSEES}")
    return _CACHE["DONNEES_LISSEES"]

def obtenir_modele_markov(logger: logging.Logger, forcer_chargement = False):
    if "MODELE_MARKOV" not in _CACHE:
        _CACHE["MODELE_MARKOV"] = charger_json(EMPLACEMENT_MODELE_MARKOV, logger)
        logger.info(f"Modèle Markov chargé depuis {EMPLACEMENT_MODELE_MARKOV}")
    return _CACHE["MODELE_MARKOV"]

def obtenir_meilleur_modele(logger: logging.Logger, forcer_chargement = False):
    if "MEILLEUR_MODELE" not in _CACHE:
        _CACHE["MEILLEUR_MODELE"] = charger_json(EMPLACEMENT_MEILLEUR_MODELE, logger)
        logger.info(f"Meilleur modèle chargé depuis {EMPLACEMENT_MEILLEUR_MODELE}")
    return _CACHE["MEILLEUR_MODELE"]

# === Matrices numpy extraites des JSONs ===

def obtenir_x_markov(logger: logging.Logger, forcer_chargement = False):
    """Retourne la matrice numpy du modèle markov."""
    modele_markov = obtenir_modele_markov(logger)
    if "X_MARKOV" not in _CACHE:
        _CACHE["X_MARKOV"] = np.array(modele_markov["matrice_de_transition"])
    return _CACHE["X_MARKOV"]

def obtenir_x_meilleur_modele(logger: logging.Logger, forcer_chargement = False):
    meilleur_modele = obtenir_meilleur_modele(logger)
    if "X_MEILLEUR_MODELE" not in _CACHE:
        _CACHE["X_MEILLEUR_MODELE"] = np.array(meilleur_modele["matrice_de_transition"])
    return _CACHE["X_MEILLEUR_MODELE"]

# --- Optionnel : fonction pour vider le cache (pour tests ou RAM management)
def clear_cache():
    """Vide tout le cache RAM (utile pour debug/test)."""
    _CACHE.clear()

