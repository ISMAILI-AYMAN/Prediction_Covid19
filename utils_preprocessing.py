"""

utils_preprocessing.py
--------------------------------



Auteur   : Harry FRANCIS (2025)
Contact  : github.com/documentHarry

License  : Code source ouvert à la consultation.
           Toute redistribution ou usage commercial est soumis à accord écrit préalable.

Version  : 1.0.0 (2025-07-24)
"""

from datetime import datetime
import logging
from typing import Any
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split


from utils_log import log_debut_fin_logger_dynamique

@log_debut_fin_logger_dynamique("logger")
def enrichir_le_dataset(
    donnees_covid: dict[str, dict[str, Any]], donnees_lissees: dict[str, dict[str, Any]],
    x_meilleur_modele: np.ndarray, COMMUNES: list[str], logger:logging.Logger
    ) -> pd.DataFrame:
    """
    Génère un DataFrame enrichi à partir de données Covid brutes, lissées et Markov.
    Ajoute des variables temporelles et des agrégats pour chaque commune/jour.

    Args:
        donnees_covid (dict): Données brutes Covid (dates -> communes -> valeurs).
        donnees_lissees (dict): Données lissées (dates -> communes -> valeurs).
        x_meilleur_modele (np.ndarray): Matrice de transition Markov.
        COMMUNES (list[str]): Liste des noms de communes à traiter.
        logger (logging.Logger): Pour affichage des étapes et erreurs.

    Returns:
        pd.DataFrame: DataFrame enrichi, une ligne par commune/jour, prêt pour ML.

    Note:
        - Soulève ValueError si les dimensions ne correspondent pas.
        - Ajoute les colonnes date, commune, predict_markov, total, moyenne, jour,
          est_weekend, mois, cas_reel.
        - Les dates proviennent toujours de donnees_lissees.

    Example:
        >>> df = enrichir_le_dataset(dc, dl, mat, COMMUNES, logger)
        >>> df.head()
        date     commune    predict_markov   ...  mois  cas_reel

    Étapes:
        1. Trie les dates des données lissées.
        2. Calcule la prédiction Markov à chaque date.
        3. Agrège total et moyenne Markov sur les communes.
        4. Ajoute variables temporelles et cas réels (J+1).
        5. Retourne le DataFrame complet.

    Tips:
        - Vérifier que la matrice Markov a la même taille que le nombre de communes.
        - Toujours contrôler la cohérence entre les deux sources de données.

    Utilisation:
        À appeler avant toute standardisation ou split train/test.
        Sert d’entrée au pipeline ML.

    Limitation:
        - Suppose les mêmes clés dans donnees_covid et donnees_lissees.
        - Erreur si des communes manquent ou si la matrice n’est pas carrée.

    See also:
        - standardiser_variables pour la mise à l'échelle.
        - split_train_test pour le découpage train/test.
        - Documentation pandas.DataFrame.
    """
    try:
        logger.info(
            f"Cette section transforme les données lissées et les prédictions Markov "
            f"en un jeu de données enrichi")
        logger.info("avec des variables supplémentaires temporelles et agrégées.\n")
        DATES_COVID = sorted(donnees_covid.keys())
        DATES_LISSEES = sorted(donnees_lissees.keys())
        lignes = []
        for index_date, date in enumerate(DATES_LISSEES[:-1]):
            cas_actuels = np.array([
                donnees_lissees.get(date, {}).get(commune, 0) for commune in COMMUNES
            ]).reshape(-1, 1)

            cas_reels_suiv = np.array([
                donnees_covid.get(DATES_COVID[index_date + 1], {}).get(commune, 0) 
                for commune in COMMUNES
            ])

            if cas_actuels.shape[0] != x_meilleur_modele.shape[1]:
                raise ValueError(
                    f"Dimension cas_actuels {cas_actuels.shape[0]} "
                    f"incompatible avec matrice Markov {x_meilleur_modele.shape[1]}")

            prediction_markov = (x_meilleur_modele @ cas_actuels).flatten()

            # Ajout des features temporelles
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            jour_semaine = date_obj.weekday()
            est_weekend = 1 if jour_semaine >= 5 else 0
            mois = date_obj.month

            total_markov = prediction_markov.sum()
            moyenne_markov = prediction_markov.mean()

            # Créer une ligne pour chaque commune
            for index_commune, commune in enumerate(COMMUNES):
                lignes.append({
                    "date": date,
                    "commune": commune,
                    "predict_markov": prediction_markov[index_commune],
                    "total_markov": total_markov,
                    "moyenne_markov": moyenne_markov,
                    "jour_semaine": jour_semaine,
                    "est_weekend": est_weekend,
                    "mois": mois,
                    "cas_reel": cas_reels_suiv[index_commune]
                })

        df_fonctionnalites_suppl = pd.DataFrame(lignes)
        return df_fonctionnalites_suppl
    except Exception as e:
        logger.error(f"[ERREUR enrichir_le_dataset] : {e}", exc_info = True)
        raise


@log_debut_fin_logger_dynamique("logger")
def standardiser_variables(
    df: pd.DataFrame, logger: logging.Logger
    )-> tuple[pd.DataFrame, StandardScaler, StandardScaler, StandardScaler]:
    """
    Applique une mise à l'échelle (standardisation) aux variables Markov du DataFrame.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes à standardiser.
    
    Returns:
        tuple: (df, scaler_predict, scaler_total, scaler_moyenne)
            - df (pd.DataFrame): DataFrame mis à jour avec colonnes standardisées.
            - scaler_predict (StandardScaler): Scaler pour la variable predict_markov.
            - scaler_total (StandardScaler): Scaler pour la variable total_markov.
            - scaler_moyenne (StandardScaler): Scaler pour la variable moyenne_markov.

    Note:
        - Les scalers sont "fit" sur l'ensemble du DataFrame fourni.
        - Les colonnes créées sont :
            - predict_markov_mise_a_l_echelle
            - total_markov_mis_a_l_echelle
            - moyenne_markov_mise_a_l_echelle
        - Nécessite que les trois colonnes Markov existent dans df.

    Example:
        >>> df, s1, s2, s3 = standardiser_variables(df)
        >>> print(df.columns)
        Index(['...', 'predict_markov_mise_a_l_echelle', 'total_markov_mis_a_l_echelle',
               'moyenne_markov_mise_a_l_echelle'], ...)

    Étapes:
        1. Instancie un StandardScaler pour chaque variable Markov.
        2. Ajuste et transforme chaque colonne, ajoute les colonnes standardisées.
        3. Retourne le DataFrame mis à jour et les trois objets scalers.

    Tips:
        - Toujours sauvegarder les scalers pour réutilisation sur les données futures.
        - Le scaler_predict sert aussi pour dénormaliser les prédictions du modèle.

    Utilisation:
        À placer après la création du DataFrame de features, avant le split train/test.
        Indispensable si tu utilises des modèles sensibles à l’échelle (ex : régression).

    Limitation:
        - Échoue si une colonne attendue est absente du DataFrame (voir validation).
        - Ne traite pas les valeurs manquantes automatiquement.
    
    See also:
        - split_train_test pour la séparation en jeu d'entraînement et de test.
        - enrichir_le_dataset pour la création des variables Markov.
        - Documentation scikit-learn StandardScaler.
    """
    try:
        logger.info(f"Applique une mise à l'échelle aux prédictions Markov "
                    f"pour une meilleure performance du modèle de régression.")
        scaler_predict = StandardScaler()
        scaler_total = StandardScaler()
        scaler_moyenne = StandardScaler()

        df["predict_markov_mise_a_l_echelle"] = scaler_predict.fit_transform(
            df[["predict_markov"]])
        df["total_markov_mis_a_l_echelle"] = scaler_total.fit_transform(
            df[["total_markov"]])
        df["moyenne_markov_mise_a_l_echelle"] = scaler_moyenne.fit_transform(
            df[["moyenne_markov"]])

        return df, scaler_predict, scaler_total, scaler_moyenne
    except Exception as e:
        logger.error(f"[ERREUR standardiser_variables] : {e}", exc_info = True)
        raise

@log_debut_fin_logger_dynamique("logger")
def split_train_test(
    df: pd.DataFrame, logger: logging.Logger
    ) -> tuple[
        pd.DataFrame, pd.DataFrame,     # X_train, X_test
        np.ndarray, np.ndarray,         # y_train, y_test
        pd.DataFrame, pd.DataFrame      # df_train_info, df_test_info
    ]:
    """
    Découpe le DataFrame en ensembles d'entraînement et de test, cible + features.
    Applique une transformation log à la cible pour stabilité.

    Args:
        df (pd.DataFrame): DataFrame avec features et variable cible ("cas_reel").

    Returns:
        tuple: (X_train, X_test, y_train, y_test, df_train_info, df_test_info)
            - X_train, X_test (pd.DataFrame): Features pour entraînement/test.
            - y_train, y_test (np.ndarray): Cible transformée (np.log1p).
            - df_train_info, df_test_info (pd.DataFrame): Infos pour post-traitement.

    Note:
        - Split aléatoire 80% train, 20% test, reproductible (random_state=42).
        - La cible (cas_reel) est transformée par np.log1p.

    Example:
        >>> Xtr, Xte, ytr, yte, info_tr, info_te = split_train_test(df)

    Étapes:
        1. Sélectionne les colonnes features et la cible.
        2. Applique np.log1p sur la cible.
        3. Splitte les données via train_test_split.
        4. Retourne tous les jeux de données et les infos d’index.

    Tips:
        - Garder df_test_info pour l’analyse après prédiction.
        - Le random_state permet de refaire toujours le même split.

    Utilisation:
        Juste après la standardisation, avant l’entraînement du modèle.

    Limitation:
        - Ne prend pas en compte le déséquilibre temporel ou spatial.
        - Si df est petit, la taille du test peut être trop faible.
    
    See also:
        - sklearn.model_selection.train_test_split.
        - standardiser_variables pour le prétraitement.
    """
    try:
        logger.info("Divise les données en X (features) et y (cible log-transformée).")
        X = df[[
            "predict_markov_mise_a_l_echelle", "total_markov_mis_a_l_echelle", 
            "moyenne_markov_mise_a_l_echelle", "jour_semaine", "est_weekend", "mois"
        ]]
        y = np.log1p(df["cas_reel"])

        X_train, X_test, y_train, y_test, df_train_info, df_test_info = (
        train_test_split(
            X, y, df[["date", "commune"]], test_size=0.2, random_state=42
        ))
        return X_train, X_test, y_train, y_test, df_train_info, df_test_info
    except Exception as e:
        logger.error(f"[ERREUR split_train_test] : {e}", exc_info = True)
        raise

