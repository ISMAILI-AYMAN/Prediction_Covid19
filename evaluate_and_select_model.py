# -*- coding: utf-8 -*-

"""
Module d'évaluation et de sélection des modèles Markov
======================================================

Fournit des fonctions pour :
- Calculer l'erreur d'un modèle (MAE)
- Évaluer la performance d'une matrice de transition sur les données
- Chercher le meilleur hyperparamètre (alpha géographique)
- Générer les modèles pondérés (LS/EM) et sélectionner le meilleur

Auteur : Harry FRANCIS (2025)
Version : 1.0.0
Compatibilité : Python 3.9+
"""


# --- Librairies standards
import logging
import os
from typing import Any

# --- Librairies tiers
import numpy as np
from tqdm import tqdm

# --- Modules locaux
from utils_io import sauvegarder_json
from utils_log import log_debut_fin_logger_dynamique
from markov_models import (
    appliquer_des_contraintes_geographiques, esperance_maximisation_markov_avec_geo, 
    estimer_matrice_de_transition_par_moindre_carre, 
    forcer_valeurs_positives_et_normaliser
)
POIDS_MOINDRE_CARRE = "poids_LS"
POIS_ESPERANCE_MAXIMISATION = "poids_EM"
MAE_VALIDATION = "mae_validation"
TRANSITION_MATRIX = "matrice_de_transition"
ALPHA_GEO = "alpha_geo"
BEST_ALPHA_GEO = "meilleur_alpha_geo"
BEST_MAE = "meilleur_mae"
NBR_OBSERVATION = "nbr_observation"
METADATA = "metadata"
NOM_DU_FICHIER = "nom_du_fichier"
NOMBRE_DE_TESTS_SOUHAITES = 400  # ou ce que tu veux


def calculer_mae(predictions: np.ndarray, cibles: np.ndarray) -> float:
    """
    Calcule l’erreur absolue moyenne (MAE) entre deux matrices ou vecteurs.

    Args:
        predictions (np.ndarray): Valeurs prédites par un modèle ou une matrice.
        cibles (np.ndarray): Valeurs réelles de référence à comparer.

    Returns:
        float: MAE (erreur absolue moyenne) entre predictions et cibles.

    Note:
        - Utilise la formule standard de la MAE :
          moyenne des valeurs absolues des écarts.
        - Fonctionne avec n’importe quelle forme compatible de ndarray.
        - La MAE est toujours positive ou nulle ; plus elle est faible, 
          meilleure est la prédiction.

    Example:
        >>> y_pred = np.array([1.1, 2.0, 3.4])
        >>> y_true = np.array([1.0, 2.5, 3.0])
        >>> calculer_mae(y_pred, y_true)
        0.333333...

    Étapes:
        1. Calcule l’écart absolu élément par élément.
        2. Calcule la moyenne de tous les écarts.

    Tips:
        - Idéal pour comparer des modèles ou ajuster des hyperparamètres.

    Utilisation:
        À appeler à chaque étape d’évaluation pour comparer objectivement deux 
        ensembles de données numériques.

    Limitation:
        - Suppose que les tableaux sont de même dimension.

    See also:
        - np.mean, np.abs
    """
    return float(np.mean(np.abs(predictions - cibles)))


def evaluer_matrice_transition(
        matrice_transition: np.ndarray, X_t: np.ndarray, X_t1: np.ndarray) -> float:
    """
    Évalue la qualité d’une matrice de transition Markov sur des données COVID.

    Args:
        matrice_transition (np.ndarray): Matrice de transition (Markov, LS, EM…).
        X_t (np.ndarray): Matrice des cas au temps t (avant transition).
        X_t1 (np.ndarray): Matrice des cas au temps t+1 (après transition).

    Returns:
        float: Erreur MAE entre les valeurs prédites et les vraies valeurs (X_t1).

    Note:
        - Applique la matrice de transition sur X_t pour prédire X_t1.
        - Plus la MAE est basse, plus la matrice de transition est adaptée.

    Example:
        >>> score = evaluer_matrice_transition(M, X_t, X_t1)
        >>> logger.info(score)
        7.32

    Étapes:
        1. Calcule M @ X_t pour obtenir les prédictions.
        2. Calcule la MAE avec X_t1.

    Tips:
        - Peut servir de critère objectif pour sélectionner le meilleur modèle.

    Utilisation:
        Appeler à chaque test de matrice ou pour comparer différentes variantes
        de modèles.

    Limitation:
        - Les matrices doivent avoir la bonne dimension (cohérentes).

    See also:
        - calculer_mae
    """
    predictions = matrice_transition @ X_t
    return calculer_mae(predictions, X_t1)


@log_debut_fin_logger_dynamique("logger")
def chercher_meilleur_alpha_matrice_score(
        generateur_de_matrice, X_t: np.ndarray, X_t1: np.ndarray, 
        liste_alpha: list[float], logger: logging.Logger
        ) -> tuple[float, np.ndarray, float]:
    """
    Teste plusieurs valeurs d’alpha pour trouver la matrice de transition qui
    minimise la MAE.

    Args:
        generateur_de_matrice (callable): Fonction prenant alpha, renvoyant une 
            matrice de transition (ex : pour LS ou EM).
        X_t (np.ndarray): Cas à t (entrée modèle).
        X_t1 (np.ndarray): Cas à t+1 (cible à prédire).
        liste_alpha (list[float]): Liste des valeurs d’alpha à tester.
        logger (logging.Logger): Logger pour affichage et suivi de la progression.


    Returns:
        tuple: (meilleur_alpha, meilleure_matrice, meilleur_score)
            - meilleur_alpha (float): Valeur d’alpha donnant la plus faible MAE.
            - meilleure_matrice (np.ndarray): Matrice de transition optimale.
            - meilleur_score (float): Meilleure MAE obtenue.

    Note:
        - Affiche la MAE pour chaque alpha testé.
        - Très utile pour l’optimisation d’hyperparamètres par grille.

    Example:
        >>> a, M, score = chercher_meilleur_alpha_matrice_score(
                gen, X_t, X_t1, [0, 0.25, 0.5, 0.75, 1])
        >>> logger.info(a, score)

    Étapes:
        1. Pour chaque alpha de la liste :
            a. Génére la matrice de transition.
            b. Évalue la MAE sur les données.
            c. Conserve la meilleure MAE et l’alpha associé.
        2. Retourne l’alpha, la matrice et le score optimaux.

    Tips:
        - Peut s’utiliser pour EM ou moindres carrés avec contraintes.

    Utilisation:
        À appeler lors de la recherche du meilleur hyperparamètre alpha.

    Limitation:
        - Les performances peuvent dépendre de la granularité de la grille alpha.

    See also:
        - evaluer_matrice_transition
    """
    meilleur_score = float("inf")
    meilleur_alpha = None
    meilleure_matrice = None

    for alpha in liste_alpha:
        matrice = generateur_de_matrice(alpha)
        score = evaluer_matrice_transition(matrice, X_t, X_t1)
        logger.info(f"Alpha={alpha:.3f} | MAE={score:.4f}")
        if score < meilleur_score:
            meilleur_score = score
            meilleur_alpha = alpha
            meilleure_matrice = matrice

    return meilleur_alpha, meilleure_matrice, meilleur_score


@log_debut_fin_logger_dynamique("logger")
def generer_modeles_ponderes(
        matrice_moindre_carre: np.ndarray, matrice_esperance_maximisation: np.ndarray, 
        X_t: np.ndarray, X_t1: np.ndarray, emplacement_sauvegarde: str, 
        logger: logging.Logger, nombre_de_test: int = 400) -> dict[str, Any]:
    """
    Crée et enregistre plusieurs modèles combinés pondérés (LS/EM), 
    puis sélectionne la meilleure combinaison selon la MAE.

    Args:
        matrice_moindre_carre (np.ndarray): Matrice de transition LS.
        matrice_esperance_maximisation (np.ndarray): Matrice de transition EM.
        X_t (np.ndarray): Matrice des cas à t (entrée).
        X_t1 (np.ndarray): Matrice des cas à t+1 (cible).
        emplacement_sauvegarde (str): Dossier où sauvegarder les modèles.
        logger (logging.Logger): Logger pour affichage et suivi de la progression.
        nombre_de_test (int): Nombre de combinaisons à tester (défaut 400).

    Returns:
        dict[str, Any]: Dictionnaire détaillant la meilleure combinaison
            trouvée : poids LS, poids EM, MAE, nom du fichier, etc.

    Note:
        - Sauvegarde chaque combinaison testée dans un fichier distinct,
          et enregistre la meilleure en résumé.
        - Les poids testés vont de 0% LS/100% EM à 100% LS/0% EM.
        - Utilise tqdm pour afficher la progression de génération.

    Example:
        >>> res = generer_modeles_ponderes(M_LS, M_EM, X_t, X_t1, "dossier")
        >>> logger.info(res["poids_LS"], res["mae_validation"])

    Étapes:
        1. Pour chaque pondération (0 à 1, par pas régulier) :
            a. Calcule la matrice pondérée.
            b. Prédit X_t1 à partir de X_t.
            c. Calcule la MAE et enregistre le modèle.
            d. Mémorise la meilleure combinaison.
        2. Sauvegarde la meilleure au format JSON.

    Tips:
        - Permet d’explorer tous les mélanges LS/EM, pas juste 0, 0.5, 1.
        - Peut consommer du temps/disque : adapter nombre_de_test si besoin.

    Utilisation:
        À appeler à la fin de l’apprentissage pour sélectionner le
        meilleur modèle combiné (souvent avant déploiement/prédiction).

    Limitation:
        - La granularité dépend de nombre_de_test.
        - Peut générer beaucoup de fichiers si nombre_de_test est élevé.

    See also:
        - fusionner_les_modeles (markov_models)
    """

    if not os.path.exists(emplacement_sauvegarde):
        os.makedirs(emplacement_sauvegarde)

    meilleur_mae = float("inf")
    meilleur_combinaison = None

    logger.info(f"Début de la génération de {nombre_de_test} modèles pondérés LS/EM…")
    for i in tqdm(range(nombre_de_test + 1), desc="🔄 Génération des modèles"):
        poids_moindre_carre = i / nombre_de_test
        poids_esperance_maximisation = 1.0 - poids_moindre_carre
        matrice_fusionnee = (
            poids_moindre_carre * matrice_moindre_carre +
            poids_esperance_maximisation * matrice_esperance_maximisation
        )
        preds = matrice_fusionnee @ X_t
        mae = np.mean(np.abs(X_t1 - preds))
        result = {
            "poids_LS": poids_moindre_carre,
            "poids_EM": poids_esperance_maximisation,
            "mae_validation": mae,
            "matrice_de_transition": matrice_fusionnee.tolist()
        }
        nom_fichier = (
            f"model_combination_idx{i:04d}"
            f"_ls{poids_moindre_carre:.4f}"
            f"_em{poids_esperance_maximisation:.4f}.json"
        )
        emplacement_fichier = os.path.join(emplacement_sauvegarde, nom_fichier)
        try:
            sauvegarder_json(result, emplacement_fichier, ecrasement = True,
                             logger = logger)
        except Exception as exception:
            logger.error(f"Erreur de sauvegarde du modèle {nom_fichier}: {exception}",
                         exc_info = True)

        if mae < meilleur_mae:
            meilleur_mae = mae
            meilleur_combinaison = result
            meilleur_combinaison["nom_du_fichier"] = nom_fichier

    meilleur_emplacement_json = os.path.join(
        emplacement_sauvegarde, "best_combination_model.json")   
    try:
        # Sauvegarder la meilleure combinaison de modèle
        sauvegarder_json(meilleur_combinaison, meilleur_emplacement_json, 
                         ecrasement = True, logger = logger)
    except Exception as exception:
        logger.error(f"Erreur de sauvegarde du meilleur modèle : {exception}", 
                     exc_info = True)

    logger.info(f"Fin de la génération. Meilleur MAE={meilleur_mae:.6f} "
                f"sauvegardé dans {meilleur_emplacement_json}")

    logger.info(f"\nMeilleur modèle combiné : {meilleur_combinaison['nom_du_fichier']} "
          f"avec MAE = {meilleur_mae:.6f}")
    logger.info(f"Résumé sauvegardé dans : {meilleur_emplacement_json}")

    return meilleur_combinaison


@log_debut_fin_logger_dynamique("logger")
def entrainer_modele_moindre_carre(
        X_t, X_t1, matrice_des_poids_geo, nbr_communes, logger: logging.Logger,
        valeurs_alpha_geo: list[float] = None) -> dict:
    """
    Entraîne un modèle moindres carrés géographiquement contraint
    pour différents alpha et sélectionne le meilleur.

    Args:
        X_t (np.ndarray): Matrice des cas à t.
        X_t1 (np.ndarray): Matrice des cas à t+1.
        matrice_des_poids_geo (np.ndarray): Matrice des poids géographiques.
        nbr_communes (int): Nombre de communes.
        logger (logging.Logger): Logger pour affichage et suivi de la progression.
        valeurs_alpha_geo (list[float], optionnel): Valeurs d’alpha à tester.

    Returns:
        dict: Dictionnaire des résultats pour chaque alpha, plus les métadonnées
            (meilleur alpha, meilleure MAE, nombre d’observations).

    Note:
        - Pour chaque alpha, génère une matrice contrainte, la normalise,
          et évalue sa MAE.
        - Enregistre la meilleure configuration selon la MAE.

    Example:
        >>> d = entrainer_modele_moindre_carre(X_t, X_t1, M_geo, 19, [0, 0.5, 1])
        >>> logger.info(d["metadata"]["meilleur_alpha_geo"])

    Étapes:
        1. Estime la matrice de base (LS).
        2. Pour chaque alpha, applique les contraintes et évalue la MAE.
        3. Récupère et retourne la meilleure configuration.

    Tips:
        - Adapte la grille d’alpha selon le besoin de précision.

    Utilisation:
        À utiliser dans tout pipeline Markov pour la validation des modèles
        moindres carrés avec contraintes.

    Limitation:
        - Les résultats dépendent fortement de la cohérence des données et des poids.

    See also:
        - appliquer_des_contraintes_geographiques
        - forcer_valeurs_positives_et_normaliser
    """
    if valeurs_alpha_geo is None:
        valeurs_alpha_geo = generer_les_valeurs_alpha(0.025)
    # Estimation de la matrice de base
    matrice_de_base = estimer_matrice_de_transition_par_moindre_carre(
        X_t, X_t1, nbr_communes, logger)
    # Test de différents alpha_geo
    modeles = {}
    meilleur_alpha = None
    meilleur_taux_d_erreurs = float("inf")
    for valeur_alpha_geo in valeurs_alpha_geo:
        logger.info(f"\n Test alpha_geo = {valeur_alpha_geo}")
        # Application des contraintes géographiques
        matrice_avec_contraintes = appliquer_des_contraintes_geographiques(
            matrice_de_base, matrice_des_poids_geo, nbr_communes, logger,
            valeur_alpha_geo)
        # Post-traitement : forcer les valeurs positives + normaliser
        matrice_avec_contraintes = forcer_valeurs_positives_et_normaliser(
            matrice_avec_contraintes, logger)
        # Évaluation
        mae = np.mean(np.abs((X_t1 - matrice_avec_contraintes @ X_t)))
        modeles[f"{ALPHA_GEO}{valeur_alpha_geo}"] = {
            ALPHA_GEO : valeur_alpha_geo,
            TRANSITION_MATRIX : matrice_avec_contraintes.tolist(),
            MAE_VALIDATION: mae
        }
        logger.info(f"MAE validation : {mae:.2f}")
        if mae < meilleur_taux_d_erreurs:
            meilleur_taux_d_erreurs = mae
            meilleur_alpha = valeur_alpha_geo
    logger.info(f"\n Meilleur modèle : alpha_geo = {meilleur_alpha} \n "
          f"(MAE = {meilleur_taux_d_erreurs:.2f})")
    # Métadonnées
    modeles[METADATA] = {
        BEST_ALPHA_GEO: meilleur_alpha,
        BEST_MAE: meilleur_taux_d_erreurs,
        NBR_OBSERVATION: X_t.shape[1]
    }
    return modeles


@log_debut_fin_logger_dynamique("logger")
def entrainer_le_modele_esperance_maximisation(
        X_t: np.ndarray, X_t1: np.ndarray, matrice_des_poids_geo: np.ndarray, 
        nbr_communes: int, logger: logging.Logger, 
        valeurs_alpha_geo: list[float] = None) -> dict[str, dict]:
    """
    Entraîne un modèle par EM (avec contraintes géographiques) pour
    différentes valeurs d’alpha et sélectionne le meilleur modèle.

    Args:
        X_t (np.ndarray): Cas à t.
        X_t1 (np.ndarray): Cas à t+1.
        matrice_des_poids_geo (np.ndarray): Matrice des poids géographiques.
        nbr_communes (int): Nombre de communes.
        logger (logging.Logger): Logger pour affichage et suivi de la progression.
        valeurs_alpha_geo (list[float]): Liste d’alpha à tester.

    Returns:
        dict[str, dict]: Résultats détaillés pour chaque alpha, plus les
            métadonnées (meilleur alpha, meilleure MAE, n observations).

    Note:
        - Pour chaque alpha, applique EM avec contrainte géo, puis évalue la MAE.
        - Affiche la progression et la performance pour chaque alpha.

    Example:
        >>> res = entrainer_le_modele_esperance_maximisation(
                X_t, X_t1, M_geo, 19, [0.0, 0.5, 1.0])
        >>> logger.info(res["metadata"]["meilleur_alpha_geo"])

    Étapes:
        1. Pour chaque alpha : entraînement EM puis évaluation.
        2. Sélectionne le modèle optimal et sauvegarde les métriques.

    Tips:
        - Le processus peut être lent pour de grandes matrices ou
          beaucoup d’alpha.

    Utilisation:
        À utiliser pour toute comparaison entre méthodes LS et EM,
        ou lors de la validation croisée.

    Limitation:
        - Peut ne pas converger pour certaines configurations extrêmes.
        - Le choix de la grille alpha influe sur la performance.

    See also:
        - esperance_maximisation_markov_avec_geo (markov_models)
    """
    if valeurs_alpha_geo is None:
        valeurs_alpha_geo = generer_les_valeurs_alpha( 1 / NOMBRE_DE_TESTS_SOUHAITES)    
    modeles = {}
    meilleur_alpha = None
    meilleur_mae = float('inf')
    for alpha_geo in valeurs_alpha_geo:
        logger.info(f"\nEM avec alpha_geo={alpha_geo}:.2f")
        matrice_esperance_maximisation = esperance_maximisation_markov_avec_geo(
            X_t, X_t1, matrice_des_poids_geo, logger, alpha_geo = alpha_geo)
        # Évaluer
        preds = matrice_esperance_maximisation @ X_t
        mae = np.mean(np.abs(X_t1 - preds))

        modeles[f"{ALPHA_GEO}{alpha_geo}"] = {
            ALPHA_GEO: alpha_geo,
            TRANSITION_MATRIX: matrice_esperance_maximisation.tolist(),
            MAE_VALIDATION: mae
        }
        logger.info(f"MAE validation : {mae:.4f}")
        if mae < meilleur_mae:
            meilleur_mae = mae
            meilleur_alpha = alpha_geo
    modeles[METADATA] = {
        BEST_ALPHA_GEO: meilleur_alpha,
        BEST_MAE: meilleur_mae,
        NBR_OBSERVATION: X_t.shape[1]
    }
    return modeles


@log_debut_fin_logger_dynamique("logger")
def generer_les_valeurs_alpha(incrementation:float = 1 / NOMBRE_DE_TESTS_SOUHAITES):
    """
    Génère une liste régulière de valeurs alpha de 0 à 1 (inclus), par pas donné.

    Args:
        incrementation (float): Pas entre chaque alpha (par défaut 1/N).

    Returns:
        list[float]: Liste de tous les alpha générés (0.0, 0.025, ..., 1.0).

    Note:
        - S’assure que la valeur 1.0 est présente, même si le pas ne tombe pas
          pile dessus.
        - Arrondit à 3 décimales pour la stabilité numérique et la lisibilité.

    Example:
        >>> generer_les_valeurs_alpha(0.25)
        [0.0, 0.25, 0.5, 0.75, 1.0]

    Étapes:
        1. Crée la liste via np.arange.
        2. Arrondit chaque valeur à 3 décimales.
        3. Ajoute 1.0 si absent.

    Tips:
        - Adapter l’incrément selon la résolution d’hyperparamètre souhaitée.

    Utilisation:
        Pour toute boucle sur alpha en tuning de modèle ou grid search.

    Limitation:
        - Peut générer des doublons si l’incrément tombe pile sur 1.0.

    See also:
        - entrainer_le_modele_esperance_maximisation
        - entrainer_modele_moindre_carre
    """
    valeurs_alpha = [round(a, 3) for a in np.arange(
        0.0, 1.0 + incrementation, incrementation)
    ]
    if valeurs_alpha[-1] != 1.0:
        valeurs_alpha.append(1.0)
    return valeurs_alpha