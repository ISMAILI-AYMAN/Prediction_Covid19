# -*- coding: utf-8 -*-

"""
Auteur   : Harry FRANCIS (2025)
Contact  : github.com/documentHarry

License  : Code source ouvert à la consultation.
           Toute redistribution ou usage commercial est soumis à accord écrit préalable.

Version  : 1.0.0 (2025-07-23)
"""

__version__ = "1.0.0"

# Compatible Python 3.9+ | Typage PEP 484 | PEP 257 | PEP 8 (88 caractères max)

# --- Librairies standards
from enum import Enum, auto
import json
import ast
import logging
from os.path import join, dirname, abspath
from typing import Any, TypedDict, Union

# --- Librairies tiers
import numpy as np
import joblib

# --- Modules locaux
#from utils_io import charger_json

# === Paramètres PROXY (optionnel) ===
"""
Pour utiliser le script derrière un proxy :
    - Passez USE_PROXY à True
    - Remplissez le dictionnaire proxies avec vos identifiants et l'URL du proxy
"""
USE_PROXY: bool = False  # <--- Mets à True pour activer le proxy
PROXIES = {
    "http":  "http://user:password@proxy-server:port",  # Mets les bonnes infos si besoin
    "https": "http://user:password@proxy-server:port"
} if USE_PROXY else None


class Cles(Enum):
    DONNEES_COVID = auto()
    DONNEES_LISSEES = auto()
    X_MEILLEUR_MODELE = auto()
    MODELE_FINAL = auto()
    SCALER_PREDICT = auto()
    SCALER_TOTAL = auto()
    SCALER_MOYENNE = auto()

# === Dossier du projet ===
EMPLACEMENT_FICHIER: str = dirname(abspath(__file__))
EMPLACEMENT_DONNEES: str = join(EMPLACEMENT_FICHIER, "Data")
EMPLACEMENT_LISSAGE: str = join(EMPLACEMENT_DONNEES, "savitzky_golay")

def emplacement_donnees(nom_fichier: str) -> str:
    """
    Génère le chemin absolu vers un fichier du dossier Data.

    Args:
        nom_fichier (str): Nom du fichier (ex: "C0VID19BE_CASES_MUNI.json").

    Returns:
        str: Chemin absolu du fichier dans Data/.
    """
    return join(EMPLACEMENT_DONNEES, nom_fichier)

# === Chemins de fichiers principaux ===
EMPLACEMENT_DONNEES_BRUTES_COVID: str = emplacement_donnees("C0VID19BE_CASES_MUNI.json")
EMPLACEMENT_DONNEES_COVID: str = emplacement_donnees("C0VID19BE_CASES_MUNI_CLEAN.json")
EMPLACEMENT_DONNEES_LISSEES: str = emplacement_donnees("communes_lissage_savgol.json")
EMPLACEMENT_POIDS_GEO: str = emplacement_donnees("matrix_markov_models.json")
EMPLACEMENT_MODELE_MARKOV: str = emplacement_donnees( 
    "model_combinations/model_combination_idx0000_ls0.0000_em1.0000.json")
EMPLACEMENT_MEILLEUR_MODELE: str = emplacement_donnees(
    "model_combinations/best_combination_model.json")
EMPLACEMENT_MODELS: str = join(EMPLACEMENT_DONNEES, "model_combinations")

'''
def obtenir_donnees_covid(logger, cache=None):
    """
    Charge les données COVID nettoyées.
    Args:
        logger (logging.Logger): logger pour la traçabilité.
        cache (dict, optionnel): cache de lecture.
    Returns:
        dict: données JSON lues.
    """
    return charger_json(EMPLACEMENT_DONNEES_COVID, logger, cache)
'''

# === Entraînement et sauvegarde du modèle Gradient Boosting ===
# Si un modèle existe déjà, le charger, sinon lancer une recherche bayésienne des hyperparamètres.
# === Chemins de sauvegarde ===
EMPLACEMENT_MODELE_FINAL: str = emplacement_donnees("final_gb_model.joblib")
EMPLACEMENT_SCALER_PREDICT: str = emplacement_donnees("scaler_pred.joblib")
EMPLACEMENT_SCALER_TOTAL: str = emplacement_donnees("scaler_total.joblib")
EMPLACEMENT_SCALER_MOYENNE: str = emplacement_donnees("scaler_mean.joblib")

# === Communes ===
COMMUNES = [
    "Anderlecht", "Auderghem", "Berchem-Sainte-Agathe", "Bruxelles",
    "Etterbeek", "Evere", "Forest (Bruxelles-Capitale)", "Ganshoren", "Ixelles",
    "Jette", "Koekelberg", "Molenbeek-Saint-Jean", "Saint-Gilles",
    "Saint-Josse-ten-Noode", "Schaerbeek", "Uccle",
    "Watermael-Boitsfort", "Woluwe-Saint-Lambert", "Woluwe-Saint-Pierre"
]

ADJACENTES = {
    "Anderlecht": ["Molenbeek-Saint-Jean", "Bruxelles", "Saint-Gilles",
                   "Forest (Bruxelles-Capitale)"],
    "Auderghem": ["Woluwe-Saint-Pierre", "Etterbeek", "Ixelles", "Watermael-Boitsfort"],
    "Berchem-Sainte-Agathe": ["Ganshoren", "Koekelberg", "Molenbeek-Saint-Jean"],
    "Bruxelles": ["Jette", "Molenbeek-Saint-Jean", "Anderlecht", "Saint-Gilles",
                  "Ixelles", "Uccle", "Watermael-Boitsfort", "Etterbeek",
                  "Saint-Josse-ten-Noode", "Schaerbeek", "Evere"],
    "Etterbeek": ["Bruxelles", "Ixelles", "Schaerbeek", "Woluwe-Saint-Lambert",
                  "Woluwe-Saint-Pierre", "Auderghem"],
    "Evere": ["Bruxelles", "Schaerbeek", "Woluwe-Saint-Lambert"],
    "Forest (Bruxelles-Capitale)": ["Anderlecht", "Saint-Gilles", "Ixelles", "Uccle"],
    "Ganshoren": ["Berchem-Sainte-Agathe", "Koekelberg", "Jette"],
    "Ixelles": ["Forest (Bruxelles-Capitale)", "Uccle", "Saint-Gilles", "Bruxelles",
                "Etterbeek", "Auderghem", "Watermael-Boitsfort"],
    "Jette": ["Ganshoren", "Koekelberg", "Molenbeek-Saint-Jean", "Bruxelles"],
    "Koekelberg": ["Molenbeek-Saint-Jean", "Berchem-Sainte-Agathe", "Ganshoren",
                   "Jette"],
    "Molenbeek-Saint-Jean": ["Anderlecht", "Berchem-Sainte-Agathe", "Koekelberg",
                             "Jette", "Bruxelles"],
    "Saint-Gilles": ["Forest (Bruxelles-Capitale)", "Anderlecht", "Bruxelles",
                     "Ixelles"],
    "Saint-Josse-ten-Noode": ["Bruxelles", "Schaerbeek"],
    "Schaerbeek": ["Bruxelles", "Saint-Josse-ten-Noode", "Evere", "Etterbeek",
                   "Woluwe-Saint-Lambert"],
    "Uccle": ["Forest (Bruxelles-Capitale)", "Ixelles", "Bruxelles",
              "Watermael-Boitsfort"],
    "Watermael-Boitsfort": ["Uccle", "Bruxelles", "Ixelles", "Auderghem"],
    "Woluwe-Saint-Lambert": ["Evere", "Schaerbeek", "Etterbeek", "Woluwe-Saint-Pierre"],
    "Woluwe-Saint-Pierre": ["Woluwe-Saint-Lambert", "Etterbeek", "Auderghem"],
}

CORRESPONDANCE_NOMS = {
    "Forest": "Forest (Bruxelles-Capitale)"
     # ajoute d'autres correspondances si tu en détectes
}

# Dictionnaire des mois en français
mois_fr = {
    1: "janvier", 2: "février", 3: "mars", 4: "avril",
    5: "mai", 6: "juin", 7: "juillet", 8: "août",
    9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre"
}

class EnregistrementCovidBrut(TypedDict):
    """
    Structure typée représentant un enregistrement brut des cas COVID-19 par commune.

    Attributes :
        DATE (str): Date du relevé au format 'YYYY-MM-DD'.
        TX_DESCR_FR (str): Nom de la commune (en français).
        CASES (Union[int, str, None]): Nombre de cas. entier, chaîne "<5" (anonymisé) ou None (donnée manquante).
    """
    DATE: str
    TX_DESCR_FR: str
    CASES: Union[int, str, None]

# ============ Typage centralisé ============
DonneesCovid = dict[str, dict[str, int]]


def extraire_serie(
        donnees: DonneesCovid, commune: str, logger: logging.Logger
        ) -> tuple[list[str], list[int]]:
    """
    Extrait la série temporelle (dates, valeurs) pour une commune.

    Args:
        donnees (DonneesCovid): Données Covid par date et commune.
        commune (str): Nom de la commune.
        logger (logging.Logger): Logger pour debug.

    Returns:
        tuple[list[str], list[int]]:
            - Liste des dates (str)
            - Liste des valeurs (int)

    Example:
        dates, valeurs = extraire_serie(data, "Ixelles")
    """
    dates: list[str] = []
    valeurs: list[int] = []

    for date_str in sorted(donnees.keys()):
        try:
            entree = donnees[date_str]
            valeur = entree.get(commune)
            # Vérification stricte : doit être un int non négatif (par ex.)
            if isinstance(valeur, int) and valeur >= 0:
                dates.append(date_str)
                valeurs.append(valeur)
            elif valeur is not None and logger is not None:
                logger.warning(
                    f"Valeur inattendue pour {commune} le {date_str}: {valeur!r}")
        except Exception as e:
            if logger is not None:
                logger.error(f"Erreur extraction ({commune}, {date_str}): {e}",
                             exc_info = True)

    if logger is not None:
        logger.debug(f"Série extraite pour {commune}: {len(dates)} dates valides.")

    return dates, valeurs


def deserialiser_frontieres_dict(
        dico_json: dict, logger: logging.Logger) -> dict[frozenset, Any]:
    """
    Désérialise un dictionnaire dont les clés sont des str représentant des listes 
    (ex: "['Ixelles', 'Uccle']") en frozenset des communes.

    Args:
        dico_json (dict): Dictionnaire JSON à désérialiser (clé=str, valeur=any).
        logger (logging.Logger, optionnel): Logger pour warnings.

    Returns:
        dict[frozenset, Any]: Dictionnaire avec clés frozenset au lieu de str.

    Example:
        >>> d = {"['Ixelles', 'Uccle']": 42}
        >>> deserialiser_frontieres_dict(d)
        {frozenset({'Ixelles', 'Uccle'}): 42}
    """
    dictionnaire_json = {}
    for cle, valeur in dico_json.items():
        try:
            noms = ast.literal_eval(cle)
            if isinstance(noms, str):
                noms = [noms]
            if isinstance(noms, (list, tuple)):
                dictionnaire_json[frozenset(noms)] = valeur
            else:
                logger.warning(f"Clé non liste ou tuple ignorée dans le cache: "
                                   f"{cle} ({type(noms)})")
        except Exception as exception:
            logger.warning(f"Clé mal formée dans le cache: {cle} | Erreur: {exception}")
    return dictionnaire_json

