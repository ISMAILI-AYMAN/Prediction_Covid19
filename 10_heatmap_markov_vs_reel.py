from datetime import datetime
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from utils_loader import charger_avec_cache
from utils_log import configurer_logging
from constantes import Cles, mois_fr, COMMUNES


NIVEAU_LOG = logging.INFO
logger = configurer_logging(NIVEAU_LOG, "10_heatmap_markov_vs_reel")

donnees_lissees   = charger_avec_cache(Cles.DONNEES_LISSEES, logger)
COMMUNES          = charger_avec_cache(Cles.COMMUNES, logger)
X_MEILLEUR_MODELE = charger_avec_cache(Cles.X_MEILLEUR_MODELE, logger)
DATES_LISSEES     = charger_avec_cache(Cles.DATES_LISSEES, logger)
DATES_COVID       = charger_avec_cache(Cles.DATES_COVID, logger)

"""
Ce script génère des heatmaps des cas prédits et réels de COVID pour les communes mois par mois
sur la période de janvier 2020 à juillet 2023.

Le flux du script est le suivant :
1. Construction d'un dataset avec les prédictions de la matrice Markov et les cas réels.
2. Boucle sur chaque mois de la période cible.
3. Création de matrices (pivot tables) des valeurs prédites, réelles et de leur différence.
4. Visualisation des résultats sous forme de heatmaps côte à côte :
   - prédictions Markov
   - cas réels observés
   - différence entre prédictions et réels
5. Sauvegarde des figures dans un dossier dédié avec une nomenclature claire.
"""



# === Préparer dataset (prédictions Markov + cas réels) ===
lignes = []
for index_date, date in enumerate(DATES_LISSEES[:-1]):
    cas_actuels = np.array([
        donnees_lissees.get(date, {}).get(commune, 0) for commune in COMMUNES
    ]).reshape(-1, 1)
    cas_reels_suiv = np.array([
        donnees_lissees.get(DATES_COVID[index_date + 1], {}).get(commune, 0) for commune in COMMUNES
    ])

    if cas_actuels.shape[0] != X_MEILLEUR_MODELE.shape[1]:
        continue  # skip si incohérence

    markov_predict = (X_MEILLEUR_MODELE @ cas_actuels).flatten()

    for index_commune, commune in enumerate(COMMUNES):
        lignes.append({
            "date": DATES_LISSEES[index_date + 1],
            "commune": commune,
            "markov_pred": markov_predict[index_commune],
            "cas_reel": cas_reels_suiv[index_commune]
        })

df_features = pd.DataFrame(lignes)

# === Fonction utilitaire pour les labels des communes ===
def passer_a_la_ligne_sur_tiret(nom):
    """
    Ajoute un retour à la ligne sur les espaces et tirets pour les labels des heatmaps.
    """
    nom = nom.replace(" ", "\n")
    return nom.replace("-", "\n-")

# === Création du dossier de sauvegarde ===
os.makedirs("visualizations_markov", exist_ok=True)

# === Boucle sur chaque mois de janvier 2020 à juillet 2023 ===
for year in range(2020, 2024):
    for month in range(1, 13):
        if year == 2023 and month > 7:
            break

        temporalite = f"{year}-{month:02d}"
        temporalite_txt = f"{mois_fr[month]} {year}"

        # Filtrage des données pour le mois en cours
        df_mois = df_features[df_features["date"].str.startswith(temporalite)]
        if df_mois.empty:
            continue  # aucun enregistrement pour ce mois

        # Construction des matrices
        pivot_pred = df_mois.pivot_table(index="commune", columns="date", values="markov_pred", aggfunc="mean")
        pivot_reel = df_mois.pivot_table(index="commune", columns="date", values="cas_reel", aggfunc="mean")
        pivot_diff = pivot_pred - pivot_reel

        # Appliquer formatage des noms
        pivot_pred.index = [passer_a_la_ligne_sur_tiret(c) for c in pivot_pred.index]
        pivot_reel.index = [passer_a_la_ligne_sur_tiret(c) for c in pivot_reel.index]
        pivot_diff.index = [passer_a_la_ligne_sur_tiret(c) for c in pivot_diff.index]

        # Génération des heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(30, 10), sharey=True)

        sns.heatmap(pivot_pred, cmap="YlOrRd", annot=False, linewidths=0.5, ax=axes[0])
        axes[0].set_title(f"Prédictions - {temporalite_txt}")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Commune")
        axes[0].tick_params(axis='x', rotation=90)

        sns.heatmap(pivot_reel, cmap="YlOrRd", annot=False, linewidths=0.5, ax=axes[1])
        axes[1].set_title(f"Cas réels - {temporalite_txt}")
        axes[1].set_xlabel("Date")
        axes[1].tick_params(axis='x', rotation=90)

        sns.heatmap(pivot_diff, cmap="coolwarm", center=0, annot=False, linewidths=0.5, ax=axes[2])
        axes[2].set_title("Différence (Prédiction - Réel)")
        axes[2].set_xlabel("Date")
        axes[2].tick_params(axis='x', rotation=90)

        # Ajuster taille des labels Y
        for ax in axes:
            ax.tick_params(axis='y', labelsize=8)

        plt.tight_layout()

        # Sauvegarde
        filename = f"visualizations_markov/predictions_markov_{mois_fr[month]}_{year}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Graphique sauvegardé : {filename}")
