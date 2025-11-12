# -*- coding: utf-8 -*-

"""
09_interroger_modele_de_prediction_integre.py
---------------------
Module de prédiction des cas pour une commune donnée à une date spécifique.

Ce module charge les données et les modèles pré-entraînés pour prédire le nombre
de cas attendus dans une commune à partir d'une date et des données Markov.
Il affiche également la différence entre la prédiction et la réalité si disponible.

Exemple :
    prediction = predire_cas("2023-04-05", "Bruxelles")

Auteur   : Harry FRANCIS (2025)
Contact  : github.com/documentHarry
Version  : 1.0.0 (2025-07-23)
License  : Code source ouvert à la consultation.
           Toute redistribution ou usage commercial est soumis à accord écrit préalable.
"""


__version__ = "1.0.0"

# Compatible Python 3.10+  | Typage PEP 484 | Type PEP257

# --- Librairies standards
import logging
from datetime import datetime
from typing import Any, TypedDict, NotRequired, Sequence

# --- Librairies tiers
import pandas as pd
import numpy as np

# --- Modules locaux
from utils_dates import dates_apres, dates_autour
from utils_loader import charger_avec_cache
from utils_log import configurer_logging
from constantes import ADJACENTES, Cles, COMMUNES






# ========== CONFIG LOGGING ==========
NIVEAU_LOG = logging.INFO
logger = configurer_logging(NIVEAU_LOG, "09_interroger_modele_de_prediction_integre")

# -----------------------------------------------------------------------------
# CHARGEMENT DES MODÈLES ET SCALERS VIA CACHE
# -----------------------------------------------------------------------------
# Ces variables globales chargent en mémoire tous les modèles, scalers et
# données prétraitées nécessaires pour les prédictions.
#
# MODELE_FINAL              : Modèle de prédiction final (régression, réseau, etc.).
# PREDICT_MISE_A_L_ECHELLE  : Scaler pour la prédiction markovienne d'une commune.
# TOTAL_MIS_A_L_ECHELLE     : Scaler pour le total des prédictions sur toutes les communes.
# MISE_A_L_ECHELLE_MOYENNE  : Scaler pour la moyenne des prédictions.
# DONNEES_LISSEES           : Dictionnaire de données lissées réelles par date et commune.
# X_MEILLEUR_MODELE         : Matrice de transition Markov optimisée.
#
# Toutes ces variables sont chargées via la fonction charger_avec_cache qui
# optimise le temps d'accès en évitant des rechargements inutiles.
#
# Conseils :
#   - Le cache permet d'accélérer les tests ou traitements en série.
#   - Si une variable n'est pas chargée, la prédiction échouera.
# -----------------------------------------------------------------------------
MODELE_FINAL = charger_avec_cache(Cles.MODELE_FINAL, logger)
PREDICT_MISE_A_L_ECHELLE = charger_avec_cache(Cles.SCALER_PREDICT, logger)
TOTAL_MIS_A_L_ECHELLE = charger_avec_cache(Cles.SCALER_TOTAL, logger)
MOYENNE_MISE_A_L_ECHELLE = charger_avec_cache(Cles.SCALER_MOYENNE, logger)
DONNEES_LISSEES = charger_avec_cache(Cles.DONNEES_LISSEES, logger)
X_MEILLEUR_MODELE = charger_avec_cache(Cles.X_MEILLEUR_MODELE, logger)

# -----------------------------------------------------------------------------
# DÉFINITION DES FEATURES UTILISÉES POUR LA PRÉDICTION
# -----------------------------------------------------------------------------
# FEATURES liste les noms des variables (features) qui seront utilisées comme
# entrées du modèle de prédiction.
#
# - predict_markov_mise_a_l_echelle  : Prédiction markovienne normalisée pour la commune cible.
# - total_markov_mis_a_l_echelle     : Somme des prédictions markoviennes pour toutes les communes.
# - moyenne_markov_mise_a_l_echelle  : Moyenne des prédictions markoviennes.
# - jour_semaine                     : Numéro du jour de la semaine (0=lundi, 6=dimanche).
# - est_weekend                      : Indicateur binaire (1=week-end, 0=semaine).
# - mois                             : Numéro du mois (1 à 12).
#
# Cette liste doit être cohérente avec le modèle et les scalers chargés.
# -----------------------------------------------------------------------------
FONCTIONNALITES_LIST: list[str] = [
    "predict_markov_mise_a_l_echelle",
    "total_markov_mis_a_l_echelle",
    "moyenne_markov_mise_a_l_echelle",
    "jour_semaine",
    "est_weekend",
    "mois"
]

class ResultatPrediction(TypedDict, total = False):
    """
    Structure standardisée pour les résultats retournés par les fonctions de prédiction.

    Args:
        date (str): Date de la prédiction ("YYYY-MM-DD").
        commune (str): Nom de la commune concernée.
        prediction (float | None): Valeur prédite par le modèle, ou None en cas d'erreur.
        cas_reel (float | None): Nombre de cas réellement observé, ou None si inconnu.
        diff (float | None): Différence (prediction - cas_reel), None si données manquantes.
        erreur (str, optionnel): Message d'erreur présent uniquement si une exception est survenue.

    Note:
        - Permet de structurer les sorties des prédictions pour les traitements en batch.
        - "erreur" est présent uniquement si une exception a été rencontrée pendant la prédiction.
        - Facilite l'analyse et le reporting des performances du modèle.

    Example:
        {
            "date": "2023-04-05",
            "commune": "Anderlecht",
            "prediction": 20.5,
            "cas_reel": 18.0,
            "diff": 2.5
        }
        {
            "date": "2023-04-06",
            "commune": "Bruxelles",
            "prediction": None,
            "cas_reel": None,
            "diff": None,
            "erreur": "Commune non reconnue"
        }

    Tips:
        - Toujours vérifier la clé "erreur" avant d'utiliser la prédiction.
        - Utile pour générer des tableaux de résultats ou exporter en CSV.

    Limitation:
        - Les champs "cas_reel" et "diff" peuvent être absents si la donnée réelle n'est pas connue.
        - "erreur" est optionnel et ne doit être utilisé que pour diagnostiquer des problèmes.

    See also:
        - predire_cas_multiple (fonction qui retourne ce type de structure).
        - La documentation du modèle utilisé pour la prédiction.
    """
    date: str
    commune: str
    prediction: float | None
    cas_reel: float | None
    diff: float | None
    erreur: NotRequired[str]


class Scenario(TypedDict):
    """
    Structure représentant un scénario de prédiction (batch ou test).

    Args:
        label (str): Description courte et explicite du scénario.
        communes_principales (list[str]): Communes concernées par la prédiction.
        dates_recherchees (list[str]): Liste des dates à prédire ("YYYY-MM-DD").

    Note:
        - Permet d'organiser et de regrouper plusieurs tests de prédiction.
        - Utilisé dans les appels batch pour automatiser l'évaluation du modèle.
        - Le champ label doit être descriptif pour faciliter l'analyse des résultats.

    Example:
        {
            "label": "Simple - 1 commune, 1 date",
            "communes_principales": ["Anderlecht"],
            "dates_recherchees": ["2023-04-05"]
        }

    Tips:
        - Générer les dates dynamiquement avec dates_apres ou dates_autour si besoin.
        - Utiliser plusieurs scénarios pour couvrir différents cas d'usage.
        - Adapter les communes selon le périmètre de votre analyse.

    Limitation:
        - Les champs doivent être correctement remplis pour garantir le bon déroulement.
        - Les dates doivent respecter le format "YYYY-MM-DD".

    See also:
        - La variable scenarios (liste d'objets Scenario).
        - La fonction predire_cas_multiple qui utilise ces objets.
    """
    """Un scénario de prédiction (pour exécution batch)."""
    label: str
    communes_principales: list[str]
    dates_recherchees: list[str]

scenarios: Sequence[Scenario] = [
    {
        "label": "Simple - 1 commune, 1 date",
        "communes_principales": ["Anderlecht"],
        "dates_recherchees": ["2023-04-05"]
    },
    {
        "label": "Plusieurs dates",
        "communes_principales": ["Anderlecht"],
        "dates_recherchees": ["2023-04-12", "2023-04-20"]
    },
    {
        "label": "3 jours après le 5 avril",
        "communes_principales": ["Anderlecht"],
        "dates_recherchees": dates_apres("2023-04-05", 3)
    },
    {
        "label": "2 jours autour du 5 avril",
        "communes_principales": ["Anderlecht"],
        "dates_recherchees": dates_autour("2023-04-05", 2)
    },
    {
        "label": "Avec adjacentes",
        "communes_principales": ["Anderlecht"],
        "dates_recherchees": ["2023-04-05"]
    }
]


def charger_matrice_markov(obj: Any) -> np.ndarray:
    """
    Extrait et valide la matrice Markov, en acceptant dict, list ou ndarray.

    Args:
        obj (Any): Objet représentant la matrice Markov. Peut être un dictionnaire avec
            clé "matrice_de_transition", une liste de listes, ou un tableau numpy.

    Returns:
        np.ndarray: Matrice Markov sous forme de tableau numpy 2D, prête à l'emploi.

    Raises:
        ValueError: Si l'objet ne contient pas une matrice compatible (ni dict/list/ndarray),
            ou si la matrice n'est pas 2D.

    Note:
        - La fonction accepte plusieurs formats d'entrée pour plus de souplesse.
        - Lève une erreur explicite en cas de problème de format ou de dimension.

    Example:
        >>> m = charger_matrice_markov({"matrice_de_transition": [[0.8, 0.2],[0.3, 0.7]]})
        >>> type(m)
        <class 'numpy.ndarray'>

    Étapes:
        1. Détecte le type d'objet (dict, list, ndarray).
        2. Convertit en numpy array si besoin.
        3. Vérifie que la matrice est bien 2D.
        4. Retourne la matrice.

    Tips:
        - Utile lors du chargement de modèles Markov sauvegardés sous différents formats.
        - Pour tester un format, utilisez `type(obj)`.

    Limitation:
        - N'accepte pas d'autres formats exotiques.
        - La clé du dict doit être "matrice_de_transition" exactement.

    See also:
        - numpy.array, pour conversion de listes en ndarray.
        - Les utilitaires de chargement du module.
    """
    matrice: np.ndarray
    if isinstance(obj, dict) and "matrice_de_transition" in obj:
        matrice = np.array(obj["matrice_de_transition"])
    elif isinstance(obj, np.ndarray):
        matrice = obj
    elif isinstance(obj, list):
        matrice = np.array(obj)
    else:
        raise ValueError(f"Format Markov non géré: {type(obj)}")
    if matrice.ndim != 2:
        raise ValueError(f"Matrice Markov attendue 2D, reçu shape = {matrice.shape}")
    return matrice


X_MEILLEUR_MODELE = charger_matrice_markov(X_MEILLEUR_MODELE)


def predire_cas_multiple(
        dates: Sequence[str], communes: Sequence[str], commune_principale: str
        ) -> list[ResultatPrediction]:
    """
    Prédit les cas attendus pour chaque couple (date, commune) donné.

    Args:
        dates (Sequence[str]): Liste des dates de prédiction ("YYYY-MM-DD").
        communes (Sequence[str]): Liste des noms de communes cibles.
        commune_principale (str): Nom de la commune principale du scénario.

    Returns:
        list[ResultatPrediction]: Liste de dictionnaires résumant la date, la commune,
            la prédiction, la valeur réelle (si disponible) et la différence.

    Note:
        - Utilise la fonction predire_cas pour chaque prédiction.
        - En cas d'erreur, ajoute un champ "erreur" au résultat.

    Example:
        >>> res = predire_cas_multiple(["2023-04-05"], ["Anderlecht"], "Anderlecht")
        >>> res[0]['prediction']
        23.5

    Étapes:
        1. Pour chaque date, pour chaque commune, appelle predire_cas.
        2. Récupère la valeur réelle si disponible.
        3. Calcule la différence (prédiction - réel).
        4. Gère les exceptions et les reporte.

    Tips:
        - Peut s'utiliser en batch sur plusieurs scénarios.
        - Idéal pour automatiser les tests de modèles.

    Utilisation:
        Pour évaluer la qualité du modèle sur des données connues ou en simulation.

    Limitation:
        - Le paramètre commune_principale n'est pas utilisé dans le corps de la fonction.
        - Lève une erreur pour chaque prédiction en échec, sans stopper l'ensemble.

    See also:
        - predire_cas (pour la prédiction unitaire).
        - Les classes Scenario et ResultatPrediction.
    """
    resultats: list[ResultatPrediction] = []
    for date_str in dates:
        for commune in communes:
            try:
                pred: float = predire_cas(date_str, commune)
                cas_reel: float | None = (
                    DONNEES_LISSEES.get(date_str, {}).get(commune, None))
                diff: float | None = pred - cas_reel if cas_reel is not None else None
                resultats.append({
                    "date": date_str, "commune": commune, "prediction": pred,
                    "cas_reel": cas_reel, "diff": diff
                })
            except Exception as exception:
                logger.error(f"Erreur pour {commune} le {date_str} : {exception}")
                resultats.append({
                    "date": date_str, "commune": commune, "prediction": None,
                    "cas_reel": None, "diff": None, "erreur": str(exception)
                })
    return resultats


def etendre_la_recherche_aux_adjacentes(communes: Sequence[str]) -> list[str]:
    """Retourne la liste des communes avec les adjacentes si applicable."""
    if len(communes) == 1 and communes[0] in ADJACENTES:
        voisines: list[str] = ADJACENTES[communes[0]]
        return list(dict.fromkeys([communes[0]] + voisines))
    else:
        return communes


def predire_cas(date_str: str, commune_cible: str) -> float:
    """
    Prédit le nombre de cas attendus dans une commune à une date précise, 
    en utilisant une matrice de transition Markov et divers scalers.

    Args:
        date_str (str): Date cible pour la prédiction ("YYYY-MM-DD").
        commune_cible (str): Nom de la commune à prédire.

    Returns:
        float: Nombre de cas prévus (après dés-log transformation).

    Raises:
        ValueError: Si la commune n'est pas reconnue ou la date absente des données.

    Note:
        - Combine une prédiction Markov avec une normalisation/scaling et un modèle 
          final de type régression ou réseau.
        - Prend en compte le jour de la semaine, le mois, et si c'est un week-end.

    Example:
        >>> predire_cas("2023-04-05", "Bruxelles")
        42.5

    Étapes:
        1. Vérifie la validité de la commune et de la date.
        2. Récupère les cas actuels pour toutes les communes.
        3. Applique la matrice Markov pour obtenir la prédiction brute.
        4. Calcule les features additionnelles (total, moyenne, jour, etc.).
        5. Utilise les scalers pour normaliser les features.
        6. Prédit le log-cas avec le modèle, puis applique np.expm1.
        7. Retourne la valeur finale.

    Tips:
        - Adapte le code si de nouveaux features sont ajoutés.
        - Pour le debug, regarde les logs générés.

    Utilisation:
        Pour prédire un cas futur isolé ou alimenter un batch de scénarios.

    Limitation:
        - Lève ValueError si la date ou la commune est absente des données d'entrée.
        - Supposé fonctionner uniquement pour les communes et dates chargées.

    See also:
        - Les modules utils_dates, utils_loader.
        - La documentation du modèle de prédiction utilisé.
    """
    if commune_cible not in COMMUNES:
        logger.error(f"Commune {commune_cible} non reconnue.")
        raise ValueError(f"Commune {commune_cible} non reconnue")

    if date_str not in DONNEES_LISSEES:
        logger.error(f"Date {date_str} absente des données.")
        raise ValueError(f"Date {date_str} absente des données")

    cas_actuels: np.ndarray = np.array([
        DONNEES_LISSEES.get(date_str, {}).get(c, 0) for c in COMMUNES
    ]).reshape(-1, 1)

    if cas_actuels.shape[0] != X_MEILLEUR_MODELE.shape[1]:
        logger.error(f"Dimensions incompatibles entre cas_actuels {cas_actuels.shape} "
                     f"et matrice Markov {X_MEILLEUR_MODELE.shape}")

    predict_markov: np.ndarray = (X_MEILLEUR_MODELE @ cas_actuels).flatten()
    total_markov: float = predict_markov.sum()
    moyenne_markov: float = predict_markov.mean()

    date_obj: datetime = datetime.strptime(date_str, "%Y-%m-%d")
    jour_semaine: int = date_obj.weekday()
    est_weekend: int = 1 if jour_semaine >= 5 else 0
    mois: int = date_obj.month

    # Construire les features
    index_commune: int = COMMUNES.index(commune_cible)

    # --- Scaler sur DataFrame pour enlever les warnings
    pred_markov_df: pd.DataFrame = pd.DataFrame(
        [[predict_markov[index_commune]]], columns=["predict_markov"])
    total_markov_df: pd.DataFrame = pd.DataFrame(
        [[total_markov]], columns=["total_markov"])
    moyenne_markov_df: pd.DataFrame = pd.DataFrame(
        [[moyenne_markov]], columns=["moyenne_markov"])

    predict_markov_mis_a_l_echelle: float = (
        PREDICT_MISE_A_L_ECHELLE.transform(pred_markov_df)[0][0])
    total_markov_mis_a_l_echelle: float = (
        TOTAL_MIS_A_L_ECHELLE.transform(total_markov_df)[0][0])
    moyenne_markov_mise_a_l_echelle: float = (
        MOYENNE_MISE_A_L_ECHELLE.transform(moyenne_markov_df)[0][0])

    # --- Créer un DataFrame pour la prédiction
    features_df: pd.DataFrame = pd.DataFrame([[
        predict_markov_mis_a_l_echelle,
        total_markov_mis_a_l_echelle,
        moyenne_markov_mise_a_l_echelle,
        jour_semaine,
        est_weekend,
        mois
    ]], columns = FONCTIONNALITES_LIST)

    # Prédiction
    log_predict: float = MODELE_FINAL.predict(features_df)[0]
    predict: float = np.expm1(log_predict)
    logger.info(f"{predict:.2f} cas attendus - prédiction pour {commune_cible} "
                f"le {date_str}")

    return predict


def main() -> None:
    """
    Exécute tous les scénarios définis dans 'scenarios', affiche les prédictions 
    pour chaque combinaison de date et commune, et gère les erreurs.

    Args:
        Aucun.

    Returns:
        None

    Note:
        - Fonction d'entrée principale du module, à exécuter en tant que script.
        - Log tous les résultats de façon structurée, y compris les erreurs.

    Example:
        >>> python 09_interroger_modele_de_prediction_integre.py
        (Affiche les résultats dans la console ou un fichier log.)

    Étapes:
        1. Parcourt tous les scénarios.
        2. Étend la liste des communes si nécessaire.
        3. Prédit les cas sur chaque (date, commune).
        4. Logge les résultats formatés.
        5. Gère et logge les exceptions de haut niveau.

    Tips:
        - À exécuter pour tester plusieurs hypothèses ou périodes rapidement.
        - Utiliser le niveau de log pour filtrer les infos affichées.

    Utilisation:
        Lancer ce script pour obtenir un rapport de prédiction multi-scénario.

    Limitation:
        - Ne retourne rien, tout passe par le logging.
        - Les exceptions critiques sont affichées mais non relancées.

    See also:
        - La fonction predire_cas_multiple pour la logique de batch.
        - Le paramétrage du logger dans utils_log.
    """
    try:
        index_scenario: int
        scenario: Scenario
        for index_scenario, scenario in enumerate(scenarios):
            logger.info("="*80)
            logger.info(f"SCÉNARIO n°{index_scenario + 1} : {scenario['label']}")
            communes_ciblees: list[str] = etendre_la_recherche_aux_adjacentes(
                scenario["communes_principales"])

            resultats: list[ResultatPrediction] = predire_cas_multiple(
                scenario["dates_recherchees"], communes_ciblees,
                scenario["communes_principales"])
            logger.info("Résultats des prédictions :")
            logger.info(f"{'Date':<12} | {'Prévu':>6} | {'Réel':>6} | "
                        f"{'Diff':>6}  | {'Commune':<20}")
            logger.info("-" * 75)
            for resultat in resultats:
                prevu: str = (f"{resultat['prediction']:.2f}" if resultat["prediction"]
                        is not None else "ERR")
                reel: str = (f"{resultat['cas_reel']:.2f}" if resultat["cas_reel"]
                        is not None else "-")
                diff: str = (f"{resultat['diff']:.2f}" 
                             if resultat["diff"] is not None else "-")
                logger.info(f"{resultat['date']:<12} | {prevu:>6} | {reel:>6} | "
                            f"{diff:>7} | {resultat['commune']:<20} ")
                if resultat.get("erreur"):
                    logger.error(f"Erreur: {resultat['erreur']}")
    except Exception as exception:
        logger.critical(f"Erreur lors de la prédiction: {exception}", exc_info = True)

# === Exemple d'appel ===
if __name__ == "__main__":
    main()
