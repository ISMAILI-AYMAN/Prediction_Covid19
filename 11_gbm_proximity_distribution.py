# -*- coding: utf-8 -*-

"""
11_gbm_proximity_distribution.py
---------------------------------

Module de distribution des prédictions GBM selon une matrice de proximité combinée
(Markov × poids géographiques), montrant comment les communes voisines contribuent
à la propagation du COVID-19.

Ce module prend les prédictions du Gradient Boosting Machine et les distribue sur
les communes en utilisant une matrice de proximité où les coefficients plus élevés
indiquent une contribution plus importante des communes voisines à la propagation.

Auteur   : Harry FRANCIS (2025)
Contact  : github.com/documentHarry
Version  : 1.0.0 (2025-01-19)
License  : Code source ouvert à la consultation.
           Toute redistribution ou usage commercial est soumis à accord écrit préalable.
"""

__version__ = "1.0.0"

# Compatible Python 3.10+ | Typage PEP 484 | PEP 257 | PEP 8 (88 caractères max)

# --- Librairies standards
import logging
import os
from datetime import datetime
from typing import Any, Dict

# --- Librairies tiers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Modules locaux
from utils_loader import charger_avec_cache
from utils_log import configurer_logging, log_debut_fin
from utils_io import charger_json, sauvegarder_json
from constantes import Cles, COMMUNES, emplacement_donnees

# ========== CONFIG LOGGING ==========
NIVEAU_LOG: int = logging.INFO
logger: logging.Logger = configurer_logging(
    NIVEAU_LOG, "11_gbm_proximity_distribution")

# ========== CHARGEMENT DES DONNÉES ==========
# Chargement des modèles, scalers et données nécessaires via le cache
MODELE_FINAL = charger_avec_cache(Cles.MODELE_FINAL, logger)
PREDICT_MISE_A_L_ECHELLE = charger_avec_cache(Cles.SCALER_PREDICT, logger)
TOTAL_MIS_A_L_ECHELLE = charger_avec_cache(Cles.SCALER_TOTAL, logger)
MOYENNE_MISE_A_L_ECHELLE = charger_avec_cache(Cles.SCALER_MOYENNE, logger)
DONNEES_LISSEES = charger_avec_cache(Cles.DONNEES_LISSEES, logger)
X_MEILLEUR_MODELE_dict = charger_avec_cache(Cles.X_MEILLEUR_MODELE, logger)
# Extraire la matrice numpy depuis le dictionnaire JSON
if isinstance(X_MEILLEUR_MODELE_dict, dict) and "matrice_de_transition" in X_MEILLEUR_MODELE_dict:
    X_MEILLEUR_MODELE = np.array(X_MEILLEUR_MODELE_dict["matrice_de_transition"])
elif isinstance(X_MEILLEUR_MODELE_dict, np.ndarray):
    X_MEILLEUR_MODELE = X_MEILLEUR_MODELE_dict
else:
    raise ValueError(f"Format Markov non géré: {type(X_MEILLEUR_MODELE_dict)}")
# COMMUNES est une constante définie dans constantes.py
COMMUNES_LIST = COMMUNES

# Liste des features pour le modèle GBM
FONCTIONNALITES_LIST: list[str] = [
    "predict_markov_mise_a_l_echelle",
    "total_markov_mis_a_l_echelle",
    "moyenne_markov_mise_a_l_echelle",
    "jour_semaine",
    "est_weekend",
    "mois"
]

# Chemins des fichiers
EMPLACEMENT_POIDS_GEO: str = emplacement_donnees("geographic_weights.json")
EMPLACEMENT_SORTIE_JSON: str = emplacement_donnees("gbm_proximity_distribution.json")
EMPLACEMENT_VISUALISATIONS: str = "visualizations_gbm_proximity"


@log_debut_fin(logger)
def charger_poids_geographiques(emplacement: str) -> Dict[str, Dict[str, float]]:
    """
    Charge les poids géographiques depuis un fichier JSON.

    Args:
        emplacement (str): Chemin vers le fichier geographic_weights.json.

    Returns:
        Dict[str, Dict[str, float]]: Dictionnaire des poids géographiques
            {commune_source: {commune_cible: poids}}.

    Note:
        - Le fichier doit contenir une clé "weights" avec la structure des poids.
        - Les poids sont basés sur l'algorithme Dijkstra et les longueurs de frontières.
    """
    try:
        logger.info(f"Chargement des poids géographiques depuis {emplacement}")
        donnees = charger_json(emplacement, logger)
        if "weights" in donnees:
            return donnees["weights"]
        return donnees
    except Exception as e:
        logger.error(f"Erreur lors du chargement des poids géographiques : {e}",
                     exc_info=True)
        raise


@log_debut_fin(logger)
def generer_prediction_gbm(date_str: str, commune_cible: str) -> float:
    """
    Génère une prédiction GBM pour une commune à une date donnée.

    Args:
        date_str (str): Date au format "YYYY-MM-DD".
        commune_cible (str): Nom de la commune.

    Returns:
        float: Prédiction GBM (après transformation expm1).

    Note:
        - Utilise le même pattern que predire_cas() dans 09_interroger_modele_de_prediction_integre.py.
        - Calcule les features Markov, les normalise, puis prédit avec le modèle GBM.
    """
    if commune_cible not in COMMUNES_LIST:
        raise ValueError(f"Commune {commune_cible} non reconnue")

    if date_str not in DONNEES_LISSEES:
        raise ValueError(f"Date {date_str} absente des données")

    # Cas actuels pour toutes les communes
    cas_actuels: np.ndarray = np.array([
        DONNEES_LISSEES.get(date_str, {}).get(c, 0) for c in COMMUNES_LIST
    ]).reshape(-1, 1)

    # Prédiction Markov
    predict_markov: np.ndarray = (X_MEILLEUR_MODELE @ cas_actuels).flatten()
    total_markov: float = predict_markov.sum()
    moyenne_markov: float = predict_markov.mean()

    # Features temporelles
    date_obj: datetime = datetime.strptime(date_str, "%Y-%m-%d")
    jour_semaine: int = date_obj.weekday()
    est_weekend: int = 1 if jour_semaine >= 5 else 0
    mois: int = date_obj.month

    # Index de la commune cible
    index_commune: int = COMMUNES_LIST.index(commune_cible)

    # Normalisation des features Markov
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

    # DataFrame des features
    features_df: pd.DataFrame = pd.DataFrame([[
        predict_markov_mis_a_l_echelle,
        total_markov_mis_a_l_echelle,
        moyenne_markov_mise_a_l_echelle,
        jour_semaine,
        est_weekend,
        mois
    ]], columns=FONCTIONNALITES_LIST)

    # Prédiction GBM
    log_predict: float = MODELE_FINAL.predict(features_df)[0]
    predict: float = np.expm1(log_predict)

    return predict


@log_debut_fin(logger)
def construire_matrice_proximite_combinee(
        matrice_markov: np.ndarray,
        poids_geographiques: Dict[str, Dict[str, float]],
        communes: list[str],
        alpha: float = 0.5) -> np.ndarray:
    """
    Construit une matrice de proximité combinée à partir de la matrice Markov
    et des poids géographiques.

    Args:
        matrice_markov (np.ndarray): Matrice de transition Markov (19×19).
        poids_geographiques (Dict[str, Dict[str, float]]): Poids géographiques
            {commune_source: {commune_cible: poids}}.
        communes (list[str]): Liste des communes dans le même ordre que la matrice.
        alpha (float): Poids pour la combinaison (0.0 = uniquement géographique,
            1.0 = uniquement Markov). Défaut: 0.5.

    Returns:
        np.ndarray: Matrice de proximité combinée (19×19), normalisée par lignes.

    Note:
        - La matrice résultante est normalisée pour que chaque ligne somme à 1.
        - Formule: P_proximity = alpha * P_markov + (1-alpha) * P_geographic
    """
    try:
        logger.info(f"Construction de la matrice de proximité combinée (alpha={alpha})")
        n = len(communes)

        # Convertir les poids géographiques en matrice numpy
        matrice_geo = np.zeros((n, n))
        for i, commune_i in enumerate(communes):
            if commune_i in poids_geographiques:
                for j, commune_j in enumerate(communes):
                    if commune_j in poids_geographiques[commune_i]:
                        matrice_geo[i, j] = poids_geographiques[commune_i][commune_j]

        # Normaliser la matrice géographique par lignes
        sommes_geo = matrice_geo.sum(axis=1, keepdims=True)
        sommes_geo[sommes_geo == 0] = 1.0  # Éviter division par zéro
        matrice_geo_normalisee = matrice_geo / sommes_geo

        # Combiner les matrices
        matrice_combinee = alpha * matrice_markov + (1 - alpha) * matrice_geo_normalisee

        # Normaliser la matrice combinée par lignes
        sommes_comb = matrice_combinee.sum(axis=1, keepdims=True)
        sommes_comb[sommes_comb == 0] = 1.0
        matrice_combinee_normalisee = matrice_combinee / sommes_comb

        logger.info("Matrice de proximité combinée construite et normalisée")
        return matrice_combinee_normalisee

    except Exception as e:
        logger.error(f"Erreur lors de la construction de la matrice : {e}",
                     exc_info=True)
        raise


@log_debut_fin(logger)
def distribuer_prediction_gbm(
        predictions_gbm: Dict[str, float],
        matrice_proximite: np.ndarray,
        cas_actuels: np.ndarray,
        communes: list[str],
        date_str: str) -> np.ndarray:
    """
    Distribue les prédictions GBM selon la matrice de proximité.

    Args:
        predictions_gbm (Dict[str, float]): Prédictions GBM par commune
            {commune: prediction}.
        matrice_proximite (np.ndarray): Matrice de proximité combinée (19×19).
        cas_actuels (np.ndarray): Cas actuels par commune (vecteur 19×1).
        communes (list[str]): Liste des communes.
        date_str (str): Date de la prédiction.

    Returns:
        np.ndarray: Matrice de distribution (19×19) où [i,j] = contribution
            de la commune j à la commune i.

    Note:
        - Pour chaque commune cible i:
          - contribution_ij = P_proximity[i,j] * cases_j(t)
          - w_ij = contribution_ij / sum(contribution_ij)
          - distributed_ij = w_ij * GBM_prediction_i
        - Basé sur Pipeline_Mesoscopique_Explicatif.md Étape 3.
    """
    try:
        logger.debug(f"Distribution des prédictions GBM pour {date_str}")
        n = len(communes)
        matrice_distribution = np.zeros((n, n))

        # Convertir les prédictions GBM en vecteur
        predictions_vecteur = np.array([
            predictions_gbm.get(commune, 0.0) for commune in communes
        ])

        # Pour chaque commune cible i
        for i, commune_i in enumerate(communes):
            if predictions_vecteur[i] == 0:
                continue

            # Calculer les contributions de chaque commune source j
            contributions = np.zeros(n)
            for j in range(n):
                # contribution_ij = P_proximity[i,j] * cases_j(t)
                contributions[j] = matrice_proximite[i, j] * cas_actuels[j, 0]

            # Normaliser les contributions
            somme_contributions = contributions.sum()
            if somme_contributions > 0:
                poids = contributions / somme_contributions
            else:
                # Si aucune contribution, distribuer uniformément
                poids = np.ones(n) / n

            # Distribuer la prédiction GBM selon les poids
            for j in range(n):
                matrice_distribution[i, j] = poids[j] * predictions_vecteur[i]

        return matrice_distribution

    except Exception as e:
        logger.error(f"Erreur lors de la distribution : {e}", exc_info=True)
        raise


@log_debut_fin(logger)
def generer_matrices_distribution(
        dates: list[str],
        matrice_proximite: np.ndarray,
        communes: list[str]) -> Dict[str, np.ndarray]:
    """
    Génère les matrices de distribution pour une plage de dates.

    Args:
        dates (list[str]): Liste des dates à traiter.
        matrice_proximite (np.ndarray): Matrice de proximité combinée.
        communes (list[str]): Liste des communes.

    Returns:
        Dict[str, np.ndarray]: Dictionnaire {date: matrice_distribution}.

    Note:
        - Pour chaque date, génère les prédictions GBM et les distribue.
        - Ignore les dates absentes des données lissées.
    """
    try:
        logger.info(f"Génération des matrices de distribution pour {len(dates)} dates")
        matrices_par_date = {}

        for date_str in dates:
            if date_str not in DONNEES_LISSEES:
                logger.debug(f"Date {date_str} absente, ignorée")
                continue

            # Cas actuels
            cas_actuels = np.array([
                DONNEES_LISSEES.get(date_str, {}).get(c, 0) for c in communes
            ]).reshape(-1, 1)

            # Générer les prédictions GBM pour toutes les communes
            predictions_gbm = {}
            for commune in communes:
                try:
                    predictions_gbm[commune] = generer_prediction_gbm(date_str, commune)
                except Exception as e:
                    logger.warning(f"Erreur prédiction GBM pour {commune} le {date_str}: {e}")
                    predictions_gbm[commune] = 0.0

            # Distribuer selon la matrice de proximité
            matrice_dist = distribuer_prediction_gbm(
                predictions_gbm, matrice_proximite, cas_actuels, communes, date_str
            )
            matrices_par_date[date_str] = matrice_dist

        logger.info(f"{len(matrices_par_date)} matrices de distribution générées")
        return matrices_par_date

    except Exception as e:
        logger.error(f"Erreur lors de la génération des matrices : {e}",
                     exc_info=True)
        raise


@log_debut_fin(logger)
def sauvegarder_matrice_distribution(
        matrices_par_date: Dict[str, np.ndarray],
        communes: list[str],
        emplacement: str,
        alpha: float) -> None:
    """
    Sauvegarde les matrices de distribution au format JSON.

    Args:
        matrices_par_date (Dict[str, np.ndarray]): Matrices par date.
        communes (list[str]): Liste des communes.
        emplacement (str): Chemin de sauvegarde.
        alpha (float): Paramètre alpha utilisé pour la combinaison.

    Note:
        - Convertit les numpy arrays en dictionnaires imbriqués.
        - Structure: {date: {commune_cible: {commune_source: contribution}}}
        - Inclut des métadonnées (date de création, méthode, paramètres).
    """
    try:
        logger.info(f"Sauvegarde des matrices de distribution dans {emplacement}")

        # Convertir en dictionnaire imbriqué
        donnees_export = {}
        for date_str, matrice in matrices_par_date.items():
            donnees_export[date_str] = {}
            for i, commune_i in enumerate(communes):
                donnees_export[date_str][commune_i] = {}
                for j, commune_j in enumerate(communes):
                    donnees_export[date_str][commune_i][commune_j] = float(matrice[i, j])

        # Ajouter les métadonnées
        resultat = {
            "metadata": {
                "create_at": datetime.now().strftime("%Y-%m-%d"),
                "method": "gbm_proximity_distribution",
                "alpha": alpha,
                "communes_count": len(communes),
                "dates_count": len(matrices_par_date)
            },
            "distributions": donnees_export
        }

        sauvegarder_json(resultat, emplacement, logger, ecrasement=True)
        logger.info(f"Matrices sauvegardées : {emplacement}")

    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde : {e}", exc_info=True)
        raise


@log_debut_fin(logger)
def visualiser_matrice_distribution(
        matrice_distribution: np.ndarray,
        communes: list[str],
        date_str: str,
        emplacement_dossier: str) -> None:
    """
    Génère une heatmap de la matrice de distribution.

    Args:
        matrice_distribution (np.ndarray): Matrice de distribution (19×19).
        communes (list[str]): Liste des communes.
        date_str (str): Date de la distribution.
        emplacement_dossier (str): Dossier de sauvegarde.

    Note:
        - X-axis: Communes sources (contribuant).
        - Y-axis: Communes cibles (recevant).
        - Intensité de couleur: Coefficient de contribution (plus élevé = plus foncé).
    """
    try:
        logger.debug(f"Génération de la heatmap pour {date_str}")

        # Créer le dossier si nécessaire
        os.makedirs(emplacement_dossier, exist_ok=True)

        # Créer un DataFrame pour la heatmap
        df_heatmap = pd.DataFrame(
            matrice_distribution,
            index=communes,
            columns=communes
        )

        # Formater les noms des communes pour les labels
        def formater_nom(nom: str) -> str:
            """Ajoute des retours à la ligne pour les labels."""
            return nom.replace(" ", "\n").replace("-", "\n-")

        communes_formatees = [formater_nom(c) for c in communes]
        df_heatmap.index = communes_formatees
        df_heatmap.columns = communes_formatees

        # Créer la figure
        fig, ax = plt.subplots(figsize=(16, 14))

        # Générer la heatmap
        sns.heatmap(
            df_heatmap,
            cmap="YlOrRd",
            annot=False,
            linewidths=0.5,
            cbar_kws={"label": "Contribution"},
            ax=ax
        )

        ax.set_title(f"Distribution GBM par Proximité - {date_str}", fontsize=14, pad=20)
        ax.set_xlabel("Commune Source (Contribuant)", fontsize=12)
        ax.set_ylabel("Commune Cible (Recevant)", fontsize=12)
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=8)

        plt.tight_layout()

        # Sauvegarder
        filename = os.path.join(
            emplacement_dossier,
            f"gbm_proximity_distribution_{date_str.replace('-', '_')}.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Heatmap sauvegardée : {filename}")

    except Exception as e:
        logger.error(f"Erreur lors de la visualisation : {e}", exc_info=True)
        raise


@log_debut_fin(logger)
def main() -> None:
    """
    Fonction principale orchestrant le pipeline complet.

    Étapes:
        1. Charge toutes les données nécessaires.
        2. Construit la matrice de proximité combinée.
        3. Génère les prédictions GBM pour une plage de dates.
        4. Distribue les prédictions selon la matrice de proximité.
        5. Sauvegarde les résultats en JSON.
        6. Génère des heatmaps pour des dates clés.
        7. Affiche des statistiques de synthèse.
    """
    try:
        logger.info("==== DÉMARRAGE 11_gbm_proximity_distribution ====")

        # 1. Charger les poids géographiques
        poids_geo = charger_poids_geographiques(EMPLACEMENT_POIDS_GEO)

        # 2. Construire la matrice de proximité combinée
        alpha = 0.5  # Poids égal pour Markov et géographique
        matrice_proximite = construire_matrice_proximite_combinee(
            X_MEILLEUR_MODELE, poids_geo, COMMUNES_LIST, alpha
        )

        # 3. Obtenir les dates disponibles
        dates_disponibles = sorted(DONNEES_LISSEES.keys())
        # Traiter un échantillon (par exemple, une date par mois) pour éviter un traitement trop long
        # Pour un traitement complet, utiliser toutes les dates
        dates_a_traiter = dates_disponibles[::30]  # Une date tous les 30 jours
        logger.info(f"Traitement de {len(dates_a_traiter)} dates sur {len(dates_disponibles)} disponibles")

        # 4. Générer les matrices de distribution
        matrices_par_date = generer_matrices_distribution(
            dates_a_traiter, matrice_proximite, COMMUNES_LIST
        )

        # 5. Sauvegarder en JSON
        sauvegarder_matrice_distribution(
            matrices_par_date, COMMUNES_LIST, EMPLACEMENT_SORTIE_JSON, alpha
        )

        # 6. Générer des heatmaps pour des dates clés (échantillon)
        dates_cles = dates_a_traiter[::len(dates_a_traiter)//10] if len(dates_a_traiter) > 10 else dates_a_traiter[:5]
        for date_str in dates_cles:
            if date_str in matrices_par_date:
                visualiser_matrice_distribution(
                    matrices_par_date[date_str],
                    COMMUNES_LIST,
                    date_str,
                    EMPLACEMENT_VISUALISATIONS
                )

        # 7. Statistiques de synthèse
        logger.info("==== Statistiques de synthèse ====")
        if matrices_par_date:
            # Trouver la contribution maximale
            max_contrib = 0.0
            max_pair = None
            max_date = None
            for date_str, matrice in matrices_par_date.items():
                max_val = matrice.max()
                if max_val > max_contrib:
                    max_contrib = max_val
                    max_date = date_str
                    i, j = np.unravel_index(matrice.argmax(), matrice.shape)
                    max_pair = (COMMUNES_LIST[i], COMMUNES_LIST[j])

            logger.info(f"Contribution maximale : {max_contrib:.4f}")
            logger.info(f"  Date : {max_date}")
            logger.info(f"  De {max_pair[1]} vers {max_pair[0]}")

        logger.info("==== Fin du traitement 11_gbm_proximity_distribution ====")

    except Exception as e:
        logger.critical(f"[FATAL ERROR - main()] : {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

