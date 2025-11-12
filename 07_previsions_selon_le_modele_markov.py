# -*- coding: utf-8 -*-

"""

07_previsions_selon_le_modele_markov.py
---------------------------------------
Ce module applique un **modèle de Markov spatial** pour prédire l’évolution quotidienne des cas COVID-19 
dans les communes de Bruxelles-Capitale.

**Principe :**
- La matrice de transition encode la probabilité (ou le “flux”) pour qu’un cas “se déplace” d’une commune à l’autre entre deux jours.
- On simule l’évolution du vecteur de cas jour après jour en multipliant l’état courant par la matrice.
- On ajuste la prédiction pour conserver le même nombre total de cas (propriété importante si la matrice est imparfaite ou les données incomplètes).

**Objectif :**
- Comparer, pour chaque modèle de matrice testé, les prévisions avec les observations réelles, commune par commune et date par date.
- Fournir un “différentiel” pour analyser la qualité des modèles, par commune.

**Interpretation :**
- Si la diagonale de la matrice domine, les cas restent majoritairement “locaux” (peu de diffusion spatiale).
- Les termes hors-diagonale modélisent la mobilité/diffusion entre communes.
- La conservation de la somme des cas suppose une population fermée (pas de nouveaux cas ni de disparition soudaine).
- Le “différentiel” permet d’identifier où le modèle sur/sous-estime les cas réels.

Example:    
    Exécution directe depuis la ligne de commande :

        $ python 07_previsions_selon_le_modele_markov.py

    Affichage type :
        === Résultats pour le modèle : best_combination_model.json ===
        2022-05-13
        Prédiction :     [3.04, 1.00, ...]
        Réelles :        [4.00, 0.00, ...]
        Différentiel:    [-0.96, 1.00, ...]
    
    Génère des tableaux de comparaison entre prédictions et données réelles.

Auteur   : Harry FRANCIS (2025)
Contact  : github.com/documentHarry

License  : Code source ouvert à la consultation.
           Toute redistribution ou usage commercial est soumis à accord écrit préalable.

Version  : 1.0.0 (2025-07-24)
"""

__version__ = "1.0.0"

# Compatible Python 3.9+ | Typage PEP 484 | PEP 257 | PEP 8 (88 caractères max)

# --- Librairies standards
from datetime import datetime, timedelta
from typing import Any, Optional
import os
import logging

# --- Librairies tiers
import numpy as np
import csv

# --- Modules locaux
from constantes import EMPLACEMENT_DONNEES_LISSEES, EMPLACEMENT_MODELS, COMMUNES
from utils_io import charger_json, creer_dossier_si_absent
from utils_log import configurer_logging, obtenir_logger, log_debut_fin

# === PARAMÈTRES GLOBAUX (modifiable facilement) ===
EST_DETAILLE = False          # True pour logs DEBUG, False = logs INFO
EXPORT_CSV = False            # True pour exporter chaque différentiel en CSV
DATE_DEPART = '2022-05-12'    # Date de départ de la simulation (format 'YYYY-MM-DD')
NBR_JOURS_CONSECUTIFS = 1     # Nombre de jours à simuler consécutivement

# === EMPLACEMENTS ===
MATRICE_DE_TRANSITION = "matrice_de_transition"
MODELES_A_TESTER = [
    "best_combination_model.json",
    "model_combination_idx0000_ls0.0000_em1.0000.json",
    "model_combination_idx0100_ls0.2500_em0.7500.json",
    "model_combination_idx0200_ls0.5000_em0.5000.json",
    "model_combination_idx0300_ls0.7500_em0.2500.json",
    "model_combination_idx0400_ls1.0000_em0.0000.json",
]

#: Niveau du logging (INFO, DEBUG, WARNING...).
NIVEAU_LOG: int = logging.INFO
logger: logging.Logger = configurer_logging(
    NIVEAU_LOG, "07_previsions_selon_le_modele_markov")

"""
Chargement des modèles de transition et des données COVID-19.
Construction des matrices de cas par commune et par date.
Appel de la fonction de prédiction et comparaison pour plusieurs modèles.
Affichage des résultats des prédictions et des valeurs réelles pour inspection.
"""

@log_debut_fin(logger)
def charger_donnees_covid(
        emplacement: str, communes: list[str]
        ) -> tuple[np.ndarray, list[str], dict[str, dict[str, float]]]:
    """
    Charge les données COVID et retourne la matrice des cas (communes × dates),
    la liste triée des dates et le dictionnaire brut.

    Args:
        emplacement (str): Emplacement du fichier JSON ({date: {commune: valeur}}).
        communes (list[str]): Liste des communes (ordre des lignes du résultat).

    Returns:
        tuple:
            - X (np.ndarray): Matrice des cas [n_communes, n_dates], chaque case
                correspond au nombre de cas pour une commune donnée à une date.
            - dates (list[str]): Liste triée de toutes les dates présentes dans
                les données.
            - donnees_covid (dict[str, dict[str, float]]): 
                Dictionnaire brut des données chargées pour lookup rapide.
                (clé: date, valeur: {commune: nombre de cas})

    Note:
        - Toutes les communes du paramètre sont forcées, absents → 0.
        - X[i, j] correspond à commune i à la date j.

    Example:
        X, dates, data = charger_donnees_covid("file.json", ["Ixelles",...])

    Étapes:
        1. Charge le JSON et trie les dates.
        2. Construit la matrice X, en complétant les absents à zéro.

    Tips:
        - Vérifie bien que la liste de communes correspond à l’ordre voulu.
        - Si la commune/date n’existe pas dans le JSON, la valeur = 0.

    Utilisation:
        À utiliser avant toute prédiction Markov pour obtenir les matrices alignées.

    Limitation:
        - Suppose que le JSON a bien une structure {date: {commune: valeur}}.
    """
    donnees_covid = charger_json(emplacement, logger)
    dates = sorted(donnees_covid.keys())
    assert communes, "Liste de communes vide"
    assert dates, "Liste de dates vide"
    X = np.zeros((len(communes), len(dates)))
    for index_date, date in enumerate(dates):
        for index_commune, commune in enumerate(communes):
            X[index_commune, index_date] = donnees_covid.get(date, {}).get(commune, 0)
    return X, dates, donnees_covid


@log_debut_fin(logger)
def charger_modele_transition(emplacement: str) -> np.ndarray:
    """
    Charge une matrice de transition Markov à partir d'un fichier JSON.

    Args:
        emplacement (str): Emplacement du fichier JSON modèle.

    Returns:
        np.ndarray: Matrice de transition (shape: N communes x N communes).

    Note:
        - La matrice doit être carrée (une colonne/ligne par commune).
        - Utilise la clé "matrice_de_transition" du JSON.

    Example:
        mat = charger_modele_transition("model.json")

    Étapes:
        1. Charge le JSON du modèle.
        2. Convertit la matrice en numpy.array et vérifie la forme.

    Tips:
        - Idéal pour charger des modèles issus du pipeline Markov LS/EM.
        - Lève une assertion si la matrice n'est pas carrée.

    Utilisation:
        Utilisé dans toutes les prédictions, une matrice par modèle à tester.
    """
    modele = charger_json(emplacement, logger)
    matrice = np.array(modele[MATRICE_DE_TRANSITION])
    assert matrice.shape[0] == matrice.shape[1], "Matrice non carrée"
    return matrice


@log_debut_fin(logger)
def predire_et_comparer(
        matrice_de_transition: np.ndarray, X_t: np.ndarray, dates: list[str], 
        communes: list[str], donnees_covid: dict, date_depart: str, 
        nombre_de_jours_consecutifs: int = 1) -> dict[str, dict[str, Any]]:
    """
    Réalise des prédictions sur plusieurs jours consécutifs à partir d'une matrice de transition et compare aux valeurs réelles.

    La fonction applique la matrice de transition sur l'état initial (X_t à date_depart)
    et simule les états futurs sur le nombre de jours souhaité. Les prédictions sont ajustées
    pour conserver un total cohérent des cas. Si les données réelles sont disponibles, elles
    sont extraites pour comparaison.

    Args:
        matrice_de_transition (np.ndarray): Matrice N×N de transition Markov.
        X_t (np.ndarray): Matrice N×T des cas (communes × dates).
        dates (list[str]): Liste triée des dates (colonnes de X_t).
        communes (list[str]): Liste des communes (lignes de X_t).
        donnees_covid (dict): Dictionnaire brut {date: {commune: valeur}}.
        date_depart (str): Date de départ au format 'YYYY-MM-DD'.
        nombre_de_jours_consecutifs (int): Nombre de jours à simuler.

    Returns:
        dict[str, dict[str, list]]:
            Dictionnaire contenant pour chaque date prédite :
                - "prediction" : liste des valeurs prédites arrondies par commune
                - "valeurs_reelles" : liste des valeurs réelles si disponibles, sinon None.

    Note:
        - Pas d'effet de bord (ne modifie ni n'exporte de fichier).
        - Les prédictions sont normalisées pour conserver le total de cas à chaque étape.        

    Théorie:
        Pour chaque jour à prédire :
            - Le vecteur des cas (X_t) à la date t est multiplié à gauche par la matrice de transition (M) :
                X_{t+1} = M @ X_t
            - Chaque colonne de la matrice représente la distribution (flux sortant) pour une commune donnée.
            - Les prédictions sont normalisées à chaque étape pour garantir la conservation du nombre total de cas.

    Example:
        >>> import numpy as np
        >>> mat = np.eye(3)
        >>> X = np.array([[1,2,3],[4,5,6],[7,8,9]])
        >>> dates = ['2023-01-01','2023-01-02','2023-01-03']
        >>> communes = ['A','B','C']
        >>> donnees = {'2023-01-02': {'A':2,'B':5,'C':8}}
        >>> predire_et_comparer(mat, X, dates, communes, donnees, '2023-01-01', 1)
        {'2023-01-02': {'prediction': [1.0, 4.0, 7.0], 'valeurs_reelles': [2, 5, 8]}}
    
    Limitation:
        - Suppose que la matrice de transition est bien normalisée.
        - Ne gère pas l'ajout/suppression de cas exogènes.

    See also:
        - charger_donnees_covid, charger_modele_transition
    """
    if date_depart not in dates:
        raise ValueError(f"La date {date_depart} est absente des données")
    nbr_communes = len(communes)
    assert matrice_de_transition.shape == (nbr_communes, nbr_communes), (
        "La matrice de transition doit être carrée de taille N communes."
    )
    index_depart = dates.index(date_depart)
    X_courant = X_t[:, index_depart].reshape(-1, 1)   # Colonne N×1

    resultats = {}

    for jour in range(1, nombre_de_jours_consecutifs + 1):
        # Propagation Markovienne : X_{t+1} = M @ X_t
        X_predict = matrice_de_transition @ X_courant
        X_predict = X_predict.flatten()
        
        # Ajuster la somme des cas prédits (recalage)
        total_reel = X_courant.sum()
        if X_predict.sum() > 0:
            X_predict = X_predict * (total_reel / X_predict.sum())

        date_future = (
            datetime.strptime(date_depart, "%Y-%m-%d") + timedelta(days=jour)
                       ).strftime("%Y-%m-%d")

        # Récupère la vraie valeur si possible
        X_reel = np.array([
            donnees_covid.get(date_future, {}).get(commune, 0) 
            for commune in communes
        ])

        resultats[date_future] = {
            "prediction": [round(float(x), 2) for x in X_predict],
            "valeurs_reelles": [int(x) for x in X_reel] if X_reel.sum() > 0 else None
        }

        # On prépare la prochaine itération sur la prédiction (auto-régressif)
        X_courant = X_predict.reshape(-1, 1)

    return resultats


@log_debut_fin(logger)
def afficher_resultats(
        date: str, prediction: list[float], valeurs_reelles: Optional[list[int]], 
        logger: logging.Logger) -> None:
    """
    Affiche dans les logs la prédiction, la valeur réelle et le différentiel
    pour une date.

    Args:
        date (str): Date au format 'YYYY-MM-DD'.
        prediction (list[float]): Prédictions du modèle.
        valeurs_reelles (list[int] | None): Observé (ou None si non dispo).
        logger (logging.Logger): Logger configuré.

    Returns:
        None

    Example:
        afficher_resultats("2022-01-01", [1,2,3], [2,2,2], logger)

    Étapes:
        1. Log la date et la prédiction.
        2. Si valeur réelle dispo, log la vraie valeur et le différentiel.
        3. Sinon, note l'absence de valeur réelle.

    Tips:
        - Pour de la comparaison rapide, active le niveau DEBUG dans le logger.
    """
    predict_tableau = np.array(prediction)
    reel_tableau = np.array(valeurs_reelles) if valeurs_reelles is not None else None
    predict_str = ", ".join([f"{v:.2f}" for v in predict_tableau])
    logger.info(f"{date}")
    logger.info(f"Prédiction : \t[{predict_str}]")
    if reel_tableau is not None:
        real_str = ", ".join([f"{v:.2f}" for v in reel_tableau])
        logger.info(f"Réelles : \t[{real_str}]")
        differentiel_tableau = predict_tableau - reel_tableau
        diff_str = ", ".join([f"{v:.2f}" for v in differentiel_tableau])
        logger.info(f"Différentiel: \t[{diff_str}]")
    else:
        logger.info(f"Réelles    : Aucune donnée réelle disponible")


@log_debut_fin(logger)
def exporter_differentiel_csv(
        differentiel: dict[str, dict[str, Any]], communes: list[str], 
        emplacement: str, logger) -> None:
    """
    Exporte le différentiel (prédiction - réalité) en CSV pour chaque date.
    avec journalisation des étapes via un logger.
    
    Args:
        differentiel (dict): Résultat de predire_et_comparer.
        communes (list[str]): Liste de communes (ordre des colonnes).
        emplacement (str): Emplacement du fichier à écrire.
        logger (logging.Logger): Logger pour journaliser l’export
            (succès, erreurs…).
    Returns:
        None

    Example:
        exporter_differentiel_csv(resultats, LISTE_COMMUNES, "diff.csv", logger)

    Étapes:
        1. Crée le dossier du fichier CSV si besoin.
        2. Écrit chaque date avec le différentiel prédiction-réalité par commune.

    Tips:
        - Ouvre le CSV dans un tableur pour explorer les erreurs par commune/date.
        - Pour les analyses, combine ce CSV avec la carte des communes.

    See also:
        - logging pour la journalisation avancée.   
    """
    logger.info(f"Début de l'export CSV des différentiels dans {emplacement}")
    try:
        creer_dossier_si_absent(os.path.dirname(emplacement), logger)
        with open(emplacement, "w", newline="", encoding="utf-8") as fichier_csv:
            fichier = csv.writer(fichier_csv)
            fichier.writerow(["date"] + communes)
            for date, infos in differentiel.items():
                predict = np.array(infos["prediction"])
                if infos["valeurs_reelles"]:
                    reel = np.array(infos["valeurs_reelles"])
                else:
                    reel = np.zeros_like(predict)
                differentiel = predict - reel
                fichier.writerow([date] + list(differentiel))
            logger.info(f"CSV exporté avec succès : {emplacement}")
    except Exception as exception:
        logger.error(f"Erreur lors de l'export du CSV '{emplacement}': {exception}",
                     exc_info = True)
        raise  # Propager l’exception pour ne pas masquer l’erreur


@log_debut_fin(logger)
def _calc_erreur(predict: list[float], reel: Optional[list[int]]) -> float:
    """
    Calcule l'écart quadratique moyen (RMSE) entre prédiction et réalité.

    Args:
        predict (list[float]): Prédiction.
        reel (list[int] or None): Observé.

    Returns:
        float: RMSE ou NaN si reel is None.

    Tips:
        - RMSE = Erreur moyenne quadratique
        - Plus RMSE est bas, meilleure est la prédiction.

    Example:
        _calc_erreur([1,2,3], [1,2,3])
    """
    if reel is None:
        return float("nan")
    return np.sqrt(np.mean((np.array(predict) - np.array(reel)) ** 2))


@log_debut_fin(logger)
def main(est_detaille: bool = False, export_csv: bool = EXPORT_CSV) -> None:
    """
    Pipeline complet de prévisions COVID Markov (multi-modèles, multi-jours),
    logs détaillés, et export CSV optionnel.

    Args:
        est_detaille (bool): True pour logs DEBUG, False pour logs INFO.
        export_csv (bool): True pour exporter le différentiel en CSV.

    Returns:
        None

    Étapes:
        1. Charge la liste des modèles de transition à tester.
        2. Charge la matrice des cas COVID lissés.
        3. Pour chaque modèle :
            - Prédit l'évolution sur la période.
            - Compare à la réalité (RMSE, différentiel).
            - Log tout, exporte le CSV si demandé.

    Tips:
        - Pour des logs détaillés, passe est_detaille=True.
        - Adapte la liste des modèles à tester dans MODELES_A_TESTER.

    Utilisation:
        À exécuter en script pour tester l’ensemble des modèles sur tous les jours.

    Limitation:
        - Suppose que les modèles/données sont alignés sur les mêmes communes.
        - Le différentiel peut être bruité pour peu de cas par commune.
    """
    logger = obtenir_logger( "07_previsions_selon_le_modele_markov", est_detaille )

    logger.info("Début des prévisions Markov COVID…")

    # 1. Chargement des modèles
    emplacement_models = [os.path.join(EMPLACEMENT_MODELS, nom) 
                      for nom in MODELES_A_TESTER]
    logger.info(f"{len(emplacement_models)} modèles à tester.")

    # 2. Chargement des données COVID
    logger.info("Chargement des données COVID lissées...")
    X_t, dates, donnees_covid = charger_donnees_covid(
        EMPLACEMENT_DONNEES_LISSEES, COMMUNES)

    # 3. Boucle sur les modèles
    for emplacement in emplacement_models:
        logger.info(f"--- Test du modèle : {os.path.basename(emplacement)} ---")
        try:
            matrice = charger_modele_transition(emplacement)
        except Exception as erreur:
            logger.error(f"Erreur au chargement du modèle : {erreur}", exc_info = True)
            continue
        logger.debug(f"Matrice de transition shape={matrice.shape}")

        predictions = predire_et_comparer(
            matrice_de_transition = matrice, X_t = X_t, dates = dates,
            communes = COMMUNES, donnees_covid = donnees_covid,
            date_depart = DATE_DEPART, 
            nombre_de_jours_consecutifs = NBR_JOURS_CONSECUTIFS
        )
        logger.info(f"Prédictions effectuées sur {len(predictions)} jour(s).")

        # _______________________________________________________________________

        for date, infos in predictions.items():
            if est_detaille:
                afficher_resultats(date, infos["prediction"], 
                                   infos["valeurs_reelles"], logger)
            else:
                erreur = _calc_erreur(infos["prediction"], infos["valeurs_reelles"])
                logger.info(f"{date} | écart quadratique moyen: {erreur:.2f}")

    logger.info("=== Fin du pipeline prévisions Markov COVID ===")


if __name__ == "__main__":
    main(EST_DETAILLE)