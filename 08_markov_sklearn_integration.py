# -*- coding: utf-8 -*-

"""

08_markov_sklearn_integration.py
--------------------------------



Auteur   : Harry FRANCIS (2025)
Contact  : github.com/documentHarry

License  : Code source ouvert à la consultation.
           Toute redistribution ou usage commercial est soumis à accord écrit préalable.

Version  : 1.0.0 (2025-07-24)
"""

# Compatible Python 3.9+ | Typage PEP 484 | PEP 257 | PEP 8 (88 caractères max)

# --- Librairies standards
import logging
import os

# --- Librairies tiers
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV

# --- Modules locaux
from utils_loader import _CACHE, charger_avec_cache
from utils_log import configurer_logging, log_debut_fin
from constantes import ( 
    COMMUNES, EMPLACEMENT_DONNEES, EMPLACEMENT_DONNEES, EMPLACEMENT_MODELE_FINAL,
    EMPLACEMENT_SCALER_MOYENNE, EMPLACEMENT_SCALER_PREDICT,
    EMPLACEMENT_SCALER_TOTAL, Cles
)
from utils_preprocessing import enrichir_le_dataset, split_train_test, standardiser_variables
from utils_validation import validation_colonnes_dataframe, validation_variables_et_donnees


#: Niveau du logging (INFO, DEBUG, WARNING...).
NIVEAU_LOG: int = logging.INFO
logger: logging.Logger = configurer_logging(
    NIVEAU_LOG, "08_markov_sklearn_integration")


@log_debut_fin(logger)
def evaluer_modele(    
        modele_gradient: GradientBoostingRegressor, scaler_predict: StandardScaler,
        X_test: pd.DataFrame, y_test: np.ndarray, df_test_info: pd.DataFrame
) -> None:
    """
    Évalue le modèle sur le jeu de test et affiche les résultats via le logger.
    Calcule la MAE réelle et log les prédictions exemples.

    Args:
        modele_gradient (GradientBoostingRegressor): Modèle entraîné.
        scaler_predict (StandardScaler): Scaler pour dénormaliser les prédictions.
        X_test (pd.DataFrame): Features du jeu de test.
        y_test (np.ndarray): Cible (log-transformée) du test.
        df_test_info (pd.DataFrame): Infos (date, commune) associées.

    Returns:
        None: Les résultats sont affichés dans le logger.

    Note:
        - Applique np.expm1 sur la prédiction et la cible pour revenir à l’échelle brute.
        - Les 25 premières prédictions sont affichées (date, commune, valeurs).

    Example:
        >>> evaluer_modele(model, scaler, Xte, yte, info_te)

    Étapes:
        1. Prédit sur le set de test.
        2. Applique np.expm1 pour délog-transformer.
        3. Calcule la MAE.
        4. Log les prédictions, cas réels, et Markov d’origine.

    Tips:
        - Utile pour valider rapidement la performance et détecter les biais.
        - Le scaler est indispensable pour interpréter la colonne Markov normalisée.

    Utilisation:
        À utiliser après entraînement pour contrôle qualité, debug ou reporting.

    Limitation:
        - Pas de retour de métriques (tout est loggé).
        - Pas de visualisation graphique intégrée.

    See also:
        - sklearn.metrics.mean_absolute_error pour le calcul de la MAE.
        - standardiser_variables pour la normalisation initiale.
    """
    try:
        logger.info("Évalue le modèle et affiche quelques résultats.")
        # Produit les prédictions sur le test set et évalue via MAE (après transformation inverse).
        y_predict_log = modele_gradient.predict(X_test)
        y_predict = np.expm1(y_predict_log)
        y_test_ok = np.expm1(y_test)

        mae = mean_absolute_error(y_test_ok, y_predict)
        logger.info(f"MAE après normalisation + log transform : {mae:.4f}\n")

        # Pour éviter de déclencher le warning SettingWithCopy
        df_test_info = df_test_info.copy()
        # Avant l'assignation de nouvelles colonnes
        df_test_info["prediction"] = y_predict
        df_test_info["cas_reel"] = y_test_ok.values
        df_test_info["predict_markov"] = scaler_predict.inverse_transform(
            X_test["predict_markov_mise_a_l_echelle"].values.reshape(-1, 1)
        )

        # Supprime les doublons sur les colonnes clés
        df_test_info_clean = df_test_info.drop_duplicates(subset=["date", "commune"])

        df_log = df_test_info_clean.copy()
        for col in df_log.select_dtypes(include="float"):
            df_log[col] = df_log[col].round(2)
        cols_log = ["date", "commune", "prediction", "cas_reel", "predict_markov"]
        nbr_lignes = 25
        logger.info(f"Exemples de {nbr_lignes} prédictions sans doublons :\n"
                    f"{df_log[cols_log].head(nbr_lignes).to_string(index = False)}\n")
    except Exception as e:
        logger.error(f"[ERREUR evaluer_modele] : {e}", exc_info = True)
        raise


@log_debut_fin(logger)
def charger_modele_et_scalers() -> tuple[
        GradientBoostingRegressor, StandardScaler, StandardScaler, StandardScaler]:
    """
    Charge le modèle entraîné et ses scalers associés depuis le cache/disque.
    Permet de recharger rapidement tous les artefacts ML nécessaires à la prédiction.

    Args:
        Aucun.

    Returns:
        tuple: (modele_gradient, scaler_predict, scaler_total, scaler_moyenne)
            - modele_gradient (GradientBoostingRegressor): Modèle ML chargé.
            - scaler_predict (StandardScaler): Scaler pour predict_markov.
            - scaler_total (StandardScaler): Scaler pour total_markov.
            - scaler_moyenne (StandardScaler): Scaler pour moyenne_markov.

    Note:
        - Utilise charger_avec_cache pour optimiser l'accès.
        - Nécessaire pour prédire sur de nouvelles données.

    Example:
        >>> model, s1, s2, s3 = charger_modele_et_scalers()

    Étapes:
        1. Charge modèle et scalers via le cache.
        2. Retourne tous les objets pour une utilisation immédiate.

    Tips:
        - Toujours charger tous les objets avant de lancer une prédiction.
        - Si un fichier manque, une exception est levée.

    Utilisation:
        À utiliser en début de pipeline prédictif ou après un redémarrage de session.

    Limitation:
        - Échoue si un fichier n’existe pas sur le disque.

    See also:
        - entrainer_et_sauvegarder_modele pour l'entraînement.
        - charger_avec_cache pour la gestion du cache.
    """
    try:
        logger.info("Charge le modèle et les scalers depuis les fichiers.")
        # === Vérifier existence modèle ===
        modele_gradient = charger_avec_cache(Cles.MODELE_FINAL, logger)
        scaler_predict = charger_avec_cache(Cles.SCALER_PREDICT, logger)
        scaler_total = charger_avec_cache(Cles.SCALER_TOTAL, logger)
        scaler_moyenne = charger_avec_cache(Cles.SCALER_MOYENNE, logger)

        return modele_gradient, scaler_predict, scaler_total, scaler_moyenne

    except Exception as exception:
        logger.error(f"[ERREUR charger_modele_et_scalers] : {exception}",
                     exc_info = True)
        raise


@log_debut_fin(logger)
def entrainer_et_sauvegarder_modele(
        X_train: pd.DataFrame, y_train: np.ndarray, scaler_predict: StandardScaler,
        scaler_total: StandardScaler, scaler_moyenne: StandardScaler
        ) -> tuple[GradientBoostingRegressor, StandardScaler, StandardScaler, 
                   StandardScaler]:
    """
    Entraîne un modèle Gradient Boosting avec recherche Bayesienne et sauvegarde tout.
    Stocke le modèle et les scalers sur disque pour usage futur.

    Args:
        X_train (pd.DataFrame): Données d’entraînement (features).
        y_train (np.ndarray): Données d’entraînement (cible).
        scaler_predict (StandardScaler): Scaler pour predict_markov.
        scaler_total (StandardScaler): Scaler pour total_markov.
        scaler_moyenne (StandardScaler): Scaler pour moyenne_markov.

    Returns:
        tuple: (modele_gradient, scaler_predict, scaler_total, scaler_moyenne)
            - Tous les objets entraînés et sauvegardés.

    Note:
        - Utilise BayesSearchCV pour optimiser les hyperparamètres.
        - Les fichiers sont sauvegardés avec joblib.dump dans le dossier projet.
        - Le cache mémoire du projet est mis à jour après entraînement.

    Example:
        >>> model, s1, s2, s3 = entrainer_et_sauvegarder_modele(Xt, yt, s1, s2, s3)

    Étapes:
        1. Recherche Bayesienne sur les hyperparamètres du modèle.
        2. Entraîne le modèle avec les meilleurs paramètres.
        3. Sauvegarde le modèle et les scalers.
        4. Met à jour le cache RAM pour les futurs accès rapides.

    Tips:
        - Surveille les logs pour vérifier les hyperparamètres choisis.
        - Toujours utiliser les mêmes scalers pour entraîner et prédire.

    Utilisation:
        À utiliser lors du premier entraînement ou lors d'une mise à jour du modèle.

    Limitation:
        - Prend du temps pour de grands datasets.
        - Les sauvegardes nécessitent les droits d’écriture sur le dossier cible.

    See also:
        - charger_modele_et_scalers pour la relecture après entraînement.
        - joblib.dump pour la sérialisation.
    """
    try:
        # === Optimisation des hyperparamètres via recherche bayésienne ===
        parametre_de_recherche = {
            "n_estimators": (50, 500),
            "max_depth": (2, 10),
            "min_samples_split": (2, 10)
        }

        modele_gradient = GradientBoostingRegressor(random_state = 42)
        optimise = BayesSearchCV(
            modele_gradient, parametre_de_recherche, n_iter = 40,
            scoring="neg_mean_squared_error", cv = 3, random_state = 42, verbose = 2
        )
        ''' explication sur les différentes méthodes de scoring
        neg_mean_squared_error	            Négatif de l'erreur quadratique moyenne (MSE)
                                            Pénalise plus fortement les grandes erreurs
        neg_root_mean_squared_error	        Négatif de la racine MSE (RMSE)
                                            Interprétable dans l'unité des y
        neg_mean_squared_log_error	        Négatif de l'erreur log quadratique
                                            Utile quand les erreurs relatives comptent plus
        neg_median_absolute_error	        Négatif de la médiane des erreurs absolues
                                            Moins sensible aux valeurs aberrantes
        r2Coefficient de détermination R²	Évalue la proportion de variance expliquée
        '''
        logger.info("Démarrage de l'optimisation des hyperparamètres...")
        optimise.fit(X_train, y_train)
        modele_gradient = optimise.best_estimator_
        logger.info(f"Meilleurs hyperparamètres trouvés : {optimise.best_params_}")

        try:
            # Sauvegarde des modèles et des scalers
            # === Sauvegarder ===
            joblib.dump(modele_gradient, EMPLACEMENT_MODELE_FINAL)
            logger.info(f"NOUVEAU modèle GradientBoosting entraîné et sauvegardé : "
                        f"{EMPLACEMENT_MODELE_FINAL}")
            joblib.dump(scaler_predict, EMPLACEMENT_SCALER_PREDICT)
            logger.info(f"NOUVEAU scaler_predict sauvegardé : "
                        f"{EMPLACEMENT_SCALER_PREDICT}")
            joblib.dump(scaler_total, EMPLACEMENT_SCALER_TOTAL)
            logger.info(f"NOUVEAU scaler_total sauvegardé : "
                        f"{EMPLACEMENT_SCALER_TOTAL}")
            joblib.dump(scaler_moyenne, EMPLACEMENT_SCALER_MOYENNE)
            logger.info(f"NOUVEAU scaler_moyenne sauvegardé : "
                        f"{EMPLACEMENT_SCALER_MOYENNE}")
            logger.info(f"Modèle et scalers sauvegardés dans : {EMPLACEMENT_DONNEES}")

            # --- Mise à jour du cache (optionnel mais recommandé)
            _CACHE[Cles.MODELE_FINAL] = modele_gradient
            logger.debug("Cache RAM mis à jour pour MODELE_FINAL")
            _CACHE[Cles.SCALER_PREDICT] = scaler_predict
            logger.debug("Cache RAM mis à jour pour SCALER_PREDICT")
            _CACHE[Cles.SCALER_TOTAL] = scaler_total
            logger.debug("Cache RAM mis à jour pour SCALER_TOTAL")
            _CACHE[Cles.SCALER_MOYENNE] = scaler_moyenne
            logger.debug("Cache RAM mis à jour pour SCALER_MOYENNE")

        except Exception as sauvegarde_exception:
            logger.error(f"[ERREUR SAUVEGARDE] : {sauvegarde_exception}", 
                         exc_info = True)
            raise

        return modele_gradient, scaler_predict, scaler_total, scaler_moyenne
    except Exception as exception:
        logger.error(f"[ERREUR entrainer_et_sauvegarder_modele] : {exception}",
                     exc_info = True)
        raise


@log_debut_fin(logger)
def charger_ou_entrainer_modele(
        X_train: pd.DataFrame, y_train: np.ndarray, scaler_predict: StandardScaler,
        scaler_total: StandardScaler, scaler_moyenne: StandardScaler
        ) -> tuple[GradientBoostingRegressor, StandardScaler, StandardScaler, 
                   StandardScaler]:
    """
    Charge le modèle et ses scalers si disponibles, sinon lance un nouvel entraînement.
    Permet un workflow unique, sans avoir à changer de code selon l’état du projet.

    Args:
        X_train (pd.DataFrame): Features d’entraînement.
        y_train (np.ndarray): Cible d’entraînement.
        scaler_predict (StandardScaler): Scaler pour predict_markov.
        scaler_total (StandardScaler): Scaler pour total_markov.
        scaler_moyenne (StandardScaler): Scaler pour moyenne_markov.

    Returns:
        tuple: (modele_gradient, scaler_predict, scaler_total, scaler_moyenne)
            - Les objets chargés (ou nouvellement entraînés).

    Note:
        - Si le modèle existe sur disque, il est chargé.
        - Sinon, la fonction entraîne un nouveau modèle avec les scalers donnés.

    Example:
        >>> model, s1, s2, s3 = charger_ou_entrainer_modele(Xt, yt, s1, s2, s3)

    Étapes:
        1. Vérifie si le modèle existe sur disque.
        2. Si oui, charge le modèle et les scalers.
        3. Sinon, lance entrainer_et_sauvegarder_modele avec les données fournies.

    Tips:
        - Utilisé pour automatiser les workflows de training/retraining.

    Utilisation:
        À appeler systématiquement au début du pipeline pour éviter le code spaghetti.

    Limitation:
        - Peut réentraîner même si les données n’ont pas changé (attention aux doublons).

    See also:
        - charger_modele_et_scalers (pour le chargement direct).
        - entrainer_et_sauvegarder_modele (pour l'entraînement).
    """
    try:
        logger.info(f"Tente de charger un modèle existant,"
                    f" sinon en entraîne un nouveau.")
        if os.path.exists(EMPLACEMENT_MODELE_FINAL):
            try:
                return charger_modele_et_scalers()
            except Exception as exception:
                logger.warning(f"Problème lors du chargement du modèle. "
                      f"Ré-entrainement forcé.", exc_info = True)
        return entrainer_et_sauvegarder_modele(
            X_train, y_train, scaler_predict, scaler_total, scaler_moyenne
        )
    except Exception as exception:
        logger.error(f"[ERREUR charger_ou_entrainer_modele] : {exception}", 
                     exc_info = True)
        raise


@log_debut_fin(logger)
def main() -> None:
    """
    Lance le pipeline principal de préparation, entraînement et évaluation du modèle.
    Exécute toutes les étapes, du chargement des données à l’évaluation finale.

    Args:
        Aucun.

    Returns:
        None: Exécute tout le pipeline de A à Z, résultats dans les logs.

    Note:
        - Tous les logs sont centralisés dans le logger du projet.
        - En cas d’erreur critique, l’exception est loggée niveau CRITICAL.

    Example:
        >>> main()

    Étapes:
        1. Charge toutes les ressources et jeux de données via le cache.
        2. Valide la cohérence et l'existence des variables/dossiers.
        3. Crée le DataFrame enrichi.
        4. Vérifie les colonnes, standardise les variables.
        5. Fait le split train/test, lance le training ou le chargement du modèle.
        6. Évalue la performance et logge les résultats.

    Tips:
        - À appeler dans le bloc if __name__ == "__main__".
        - Rejoue tout le workflow, utile pour debug ou reproductibilité.

    Utilisation:
        À utiliser comme point d'entrée principal du script.

    Limitation:
        - Si une étape échoue, tout le pipeline s’arrête (log CRITICAL).
        - Les logs doivent être consultés pour voir les résultats.
    
    See also:
        - Toutes les fonctions du pipeline importées et utilisées ci-dessus.
    """
    try:
        logger.info("Lancement de la fonction main")
        # 1. Charger les ressources nécessaires (en RAM via cache)
        donnees_covid = charger_avec_cache(Cles.DONNEES_COVID, logger)
        donnees_lissees = charger_avec_cache(Cles.DONNEES_LISSEES, logger)
        X_MEILLEUR_MODELE_dict = charger_avec_cache(Cles.X_MEILLEUR_MODELE, logger)
        x_meilleur_modele = np.array(X_MEILLEUR_MODELE_dict["matrice_de_transition"])
        '''
        modele_gradient = charger_avec_cache(Cles.MODELE_FINAL, logger)
        scaler_predict = charger_avec_cache(Cles.SCALER_PREDICT, logger)
        scaler_total = charger_avec_cache(Cles.SCALER_TOTAL, logger)
        scaler_moyenne = charger_avec_cache(Cles.SCALER_MOYENNE, logger)
        '''

        # 2. Pipeline métier
        validation_variables_et_donnees(
            COMMUNES, x_meilleur_modele, donnees_covid, donnees_lissees, 
            EMPLACEMENT_DONNEES, logger
        )
        # === Préparer le dataset pour le modèle de régression ===
        # Cette section transforme les données lissées et les prédictions Markov
        # en un jeu de données enrichi avec des variables supplémentaires temporelles et agrégées.
        df_fonctionnalites = enrichir_le_dataset(
            donnees_covid, donnees_lissees, x_meilleur_modele, COMMUNES, logger
        )
        validation_colonnes_dataframe(df_fonctionnalites, logger)
        df_fonctionnalites, scaler_predict, scaler_total, scaler_moyenne = (
            standardiser_variables(df_fonctionnalites, logger)
        )
        X_train, X_test, y_train, y_test, df_train_info, df_test_info = (
            split_train_test(df_fonctionnalites, logger)
        )
        model_gradient, scaler_predict, scaler_total, scaler_moyenne = (
            charger_ou_entrainer_modele(
                X_train, y_train, scaler_predict, scaler_total, scaler_moyenne
            )
        )
        # 3. Évaluation
        evaluer_modele(model_gradient, scaler_predict, X_test, y_test, df_test_info)
    except Exception as exception:
        logger.critical(f"[FATAL ERROR - main()] : {exception}", exc_info = True)


if __name__ == "__main__":
    main()