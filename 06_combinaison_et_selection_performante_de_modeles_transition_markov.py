# -*- coding: utf-8 -*-

"""
06_combinaison_et_selection_performante_de_modeles_transition_markov.py
-----------------------------------------------------------------------

Ce module applique un modèle de Markov spatial pour prédire l’évolution quotidienne
des cas COVID-19 dans les communes de Bruxelles-Capitale.

Script principal de génération et combinaison des modèles de transition Markov pour les données COVID.
Appelle toutes les étapes : chargement, apprentissage, évaluation, sélection, sauvegarde.

Example:
    # -------------------------------------------------------------
    # Pipeline complet d’utilisation (exécution typique)
    # -------------------------------------------------------------

    1. Placez les fichiers d’entrée aux emplacements par défaut :
        - Data/communes_lissage_savgol.json
        - data/model_combinations/best_combination_model.json
        - (et autres modèles dans data/model_combinations/)

    2. Lancez le script depuis le terminal :
        $ python 07_previsions_selon_le_modele_markov.py

    3. Suivez l’affichage console :
        2025-07-24 14:00:00 | INFO | Début des prévisions Markov COVID…
        2025-07-24 14:00:00 | INFO | 6 modèles à tester.
        2025-07-24 14:00:00 | INFO | Chargement des données COVID lissées...
        2025-07-24 14:00:00 | INFO | --- Test du modèle : best_combination_model.json ---
        2025-07-24 14:00:01 | INFO | Prédictions effectuées sur 1 jour(s).
        2025-07-24 14:00:01 | INFO | 2022-05-13
        2025-07-24 14:00:01 | INFO | Prédiction :    [3.04, 1.00, ...]
        2025-07-24 14:00:01 | INFO | Réelles :       [4.00, 0.00, ...]
        2025-07-24 14:00:01 | INFO | Différentiel:   [-0.96, 1.00, ...]
        ...
        2025-07-24 14:00:02 | INFO | === Fin du pipeline prévisions Markov COVID ===

    4. (Optionnel) Exporte le différentiel en CSV si export_csv=True :
        - Fichier généré : data/model_combinations/diff_best_combination_model.csv

    5. Pour adapter le pipeline :
        - Modifiez les paramètres en haut du script (ex : nombre de jours, modèles à tester).
        - Adaptez la liste LISTE_COMMUNES si votre jeu de données change.

    6. (Développement avancé)
        - Utilisez les fonctions séparément dans un notebook Jupyter pour des analyses
          ou visualisations personnalisées.

    # -------------------------------------------------------------
    # Fin de l’exemple pipeline
    # -------------------------------------------------------------

Auteur   : Harry FRANCIS (2025)
Contact  : github.com/documentHarry

License  : Code source ouvert à la consultation.
           Toute redistribution ou usage commercial est soumis à accord écrit préalable.

Version  : 1.0.0 (2025-07-24)
"""

__version__ = "1.0.0"

# Compatible Python 3.9+ | Typage PEP 484 | PEP 257 | PEP 8 (88 caractères max)

# --- Librairies standards
import logging
import os
import json

# --- Librairies tiers
import numpy as np

# --- Modules locaux
from constantes import (
    EMPLACEMENT_DONNEES_LISSEES, EMPLACEMENT_MODELS, EMPLACEMENT_POIDS_GEO)
from utils_io import creer_dossier_si_absent
from utils_log import configurer_logging, log_debut_fin, obtenir_logger
from matrix_data_loader import (
    charger_matrice_poids_geographiques, preparer_matrices_markov_depuis_json
)
from evaluate_and_select_model import (
    entrainer_modele_moindre_carre,
    entrainer_le_modele_esperance_maximisation,
    generer_modeles_ponderes,
    generer_les_valeurs_alpha
)


# --- Constantes et emplacements ---
NOMBRE_DE_TESTS_SOUHAITES = 400
# Passe à True pour recalculer même si fichiers présents
FORCER_RECALCUL_MODELES = True

#: Niveau du logging (INFO, DEBUG, WARNING...).
NIVEAU_LOG: int = logging.INFO
logger: logging.Logger = configurer_logging(
    NIVEAU_LOG, "06_combinaison_et_selection_performante_de_modeles_transition_markov"
)

@log_debut_fin(logger)
def main() -> None:
    """
    Pipeline complet pour l’apprentissage, la validation et la sélection des
    modèles de transition Markov sur les données COVID, avec gestion centralisée
    des logs via un logger Python.

    Args:
        Aucun argument d’entrée.

    Returns:
        None: La fonction exécute tout le workflow, affiche les étapes et résultats
              via le système de logging Python, et sauvegarde les modèles au fil
              du pipeline. Aucun objet n’est retourné.

    Note:
        - Toutes les informations, résumés, alertes ou erreurs sont transmises via 
          le logger configuré en début de script (niveau INFO ou DEBUG selon le choix).
        - Les logs s’affichent dans la console par défaut ; il est possible de 
          personnaliser la destination via le module `utils`.
        - Aucun affichage direct n’est réalisé : toutes les sorties passent par le logger,
          une meilleure traçabilité et un filtrage ultérieur.
        - Le pipeline orchestre le chargement, l’apprentissage, la recherche
          d’hyperparamètre, la combinaison et la sauvegarde des modèles, sans
          retour de valeur directe.

    Example:
        >>> python modeles_transition_markov.py
        2025-07-24 14:00:00 | INFO | ==== DÉMARRAGE 06_combinaison_et_selection_performante_de_modeles_transition_markov ====
        2025-07-24 14:00:00 | INFO | Chargement effectué : 19 communes, 420 dates.
        2025-07-24 14:00:01 | INFO | Meilleur modèle fusionné généré :
        2025-07-24 14:00:01 | INFO |   > poids LS = 52.500%
        2025-07-24 14:00:01 | INFO |   > poids EM = 47.500%
        2025-07-24 14:00:01 | INFO |   > MAE = 7.3821
        2025-07-24 14:00:01 | INFO |   > fichier = model_combination_idx0200_ls0.5250_em0.4750.json

    Étapes:
        1. Initialise le logger (console, niveau INFO/DEBUG).
        2. Charge les données COVID lissées et la matrice des poids géographiques.
        3. Génère une grille de valeurs alpha pour la recherche d’hyperparamètre.
        4. Entraîne le modèle moindres carrés sur tous les alpha, conserve le meilleur.
        5. Entraîne le modèle EM sur tous les alpha, conserve le meilleur.
        6. Génère toutes les combinaisons pondérées (LS/EM), sauvegarde chaque
           combinaison et sélectionne la meilleure selon la MAE.
        7. Logge un résumé final avec les hyperparamètres et la performance du
           meilleur modèle pondéré.

    Tips:
        - Change le niveau du logger à DEBUG pour afficher les détails intermédiaires.
        - Redirige les logs vers un fichier en adaptant la configuration du logger,
          si besoin d’un historique ou pour analyse ultérieure.
        - Adapter les emplacements d’entrée ou le nombre de tests via les constantes
          en haut de script.

    Utilisation:
        À exécuter directement depuis le terminal :
        $ python modeles_transition_markov.py
        Idéal pour la recherche d’hyperparamètre et la validation de modèles
        sur des séries temporelles épidémiologiques.

    Limitation:
        - Ne gère pas l’interactivité : c’est un pipeline automatisé.
        - Peut être lent sur très gros jeux de données ou pour un très grand nombre
          de tests (ex : >10 000).
        - Les logs sont sur la console par défaut (voir `utils` pour config avancée).
        - Les fichiers de données doivent exister et être au bon format.

    See also:
        - utils.py (gestion avancée des logs, configuration)
        - matrix_data_loader.py (chargement des données)
        - markov_models.py (algorithmes Markov et contraintes géographiques)
        - evaluate_and_select_model.py (évaluation, sélection et combinaison)
        - numpy, tqdm
    """
    logger = obtenir_logger(
        "06_combinaison_et_selection_performante_de_modeles_transition_markov", 
        est_detaille = True)
    logger.info(
        f"==== DÉMARRAGE 06_combinaison_et_selection_performante"
        f"_de_modeles_transition_markov ===="
    )

    try:
        # 1. Chargement des matrices (X_t, X_t1, communes, matrice des poids 
        #    géographiques)
        X_t: np.ndarray
        X_t1: np.ndarray
        communes: list[str]
        dates: np.ndarray
        X_t, X_t1, communes, dates = (
            preparer_matrices_markov_depuis_json(EMPLACEMENT_DONNEES_LISSEES, logger )
        )
        nbr_communes: int = len(communes)
        logger.info(f"[INFO] Chargement effectué : {nbr_communes} communes, "
            f"{X_t.shape[1]} dates.")
        
        matrice_poids_geo = charger_matrice_poids_geographiques(
            EMPLACEMENT_POIDS_GEO, communes, logger
        )

        # 2. Alpha grid pour la recherche d’hyperparamètre
        valeurs_alpha: list[float] = generer_les_valeurs_alpha(
            1 / NOMBRE_DE_TESTS_SOUHAITES
        )

        dernier_fichier = os.path.join(
            EMPLACEMENT_MODELS,
            f"model_combination_idx{NOMBRE_DE_TESTS_SOUHAITES:04d}_ls1.0000_em0.0000.json"
        )

        if not FORCER_RECALCUL_MODELES and os.path.exists(dernier_fichier):
            logger.info(f"[SKIP] Tous les modèles pondérés existent déjà, aucun calcul relancé.")
            best_model_file = os.path.join(EMPLACEMENT_MODELS, "best_combination_model.json")
            with open(best_model_file, "r", encoding="utf-8") as f:
                meilleur_modele = json.load(f)
            logger.info(f"Résumé du meilleur modèle : {meilleur_modele['nom_du_fichier']} (MAE={meilleur_modele['mae_validation']:.4f})")
        else:
            # 3. Apprentissage Moindres Carrés
            resultats_ls = entrainer_modele_moindre_carre(
                X_t, X_t1, matrice_poids_geo, nbr_communes, logger, 
                valeurs_alpha_geo = valeurs_alpha
            )
            alpha_ls: float = resultats_ls["metadata"]["meilleur_alpha_geo"]
            matrice_ls = np.array(
                resultats_ls[f"alpha_geo{alpha_ls}"]["matrice_de_transition"]
            )

            # 4. Apprentissage EM
            resultats_em: dict = entrainer_le_modele_esperance_maximisation(
                X_t, X_t1, matrice_poids_geo, nbr_communes, logger, 
                valeurs_alpha_geo = valeurs_alpha
            )
            alpha_em: float = resultats_em["metadata"]["meilleur_alpha_geo"]
            matrice_em = np.array(
                resultats_em[f"alpha_geo{alpha_em}"]["matrice_de_transition"]
            )

            logger.info("Modèles pondérés incomplets ou recalcul forcé, génération complète…")
            meilleur_modele = generer_modeles_ponderes(
                matrice_ls, matrice_em, X_t, X_t1, EMPLACEMENT_MODELS, logger,
                nombre_de_test=NOMBRE_DE_TESTS_SOUHAITES
            )

            # 5. Génération et sélection du meilleur modèle pondéré
            creer_dossier_si_absent(EMPLACEMENT_MODELS, logger)
            dernier_fichier = os.path.join(
                EMPLACEMENT_MODELS,
                f"model_combination_idx{NOMBRE_DE_TESTS_SOUHAITES:04d}_ls1.0000_em0.0000.json"
            )

            logger.info("\n[INFO] Meilleur modèle fusionné généré :")
            logger.info(f"  > poids LS = {meilleur_modele['poids_LS']*100:.3f}%")
            logger.info(f"  > poids EM = {meilleur_modele['poids_EM']*100:.3f}%")
            logger.info(f"  > MAE = {meilleur_modele['mae_validation']:.4f}")
            logger.info(f"  > fichier = {meilleur_modele['nom_du_fichier']}\n")
    except Exception as exception:
        logger.error(f"Erreur critique dans le pipeline : {exception}", 
                     exc_info = True)
        raise   

if __name__ == "__main__":
    main()