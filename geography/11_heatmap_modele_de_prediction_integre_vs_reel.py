import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from constantes import *

"""
Ce script réalise des prédictions de cas pour chaque commune sur la période de janvier 2020 à juillet 2023
et génère des heatmaps comparant les prédictions aux données réelles.

Étapes principales :
1. Construction d'un dataset avec :
   - les prédictions Markov,
   - les variables dérivées (somme, moyenne),
   - les caractéristiques temporelles (jour de la semaine, mois, weekend).
2. Boucle sur les mois de la période.
3. Transformation des prédicteurs par des scalers pré-entraînés.
4. Prédiction des cas avec un modèle final pré-entraîné.
5. Agrégation sous forme de matrices pour visualisation :
   - prédictions,
   - cas réels,
   - différence entre prédictions et cas réels.
6. Génération et sauvegarde des heatmaps.

Pré-requis :
- Import des constantes et modèles pré-entraînés depuis `utils.constantes`.
"""

# Dictionnaire des mois en français
mois_fr = {
    1: "janvier", 2: "février", 3: "mars", 4: "avril",
    5: "mai", 6: "juin", 7: "juillet", 8: "août",
    9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre"
}

# === Préparer dataset complet (prédictions Markov + features) ===
lignes = []
for index_date, date in enumerate(DATES_LISSEES[:-1]):
    cas_actuels = np.array([ donnees_lissees.get(date, {}).get(commune, 0) for commune in COMMUNES ]).reshape(-1, 1)
    cas_reels_suiv = np.array([ donnees_lissees.get(DATES_COVID[index_date + 1], {}).get(commune, 0) for commune in COMMUNES ])

    if cas_actuels.shape[0] != X_MEILLEUR_MODELE.shape[1]:
        continue  # skip si incohérence

    markov_predict = (X_MEILLEUR_MODELE @ cas_actuels).flatten()
    markov_total = markov_predict.sum()
    markov_moyenne = markov_predict.mean()

    date_obj = datetime.strptime(date, "%Y-%m-%d")
    jour_semaine = date_obj.weekday()
    est_weekend = 1 if jour_semaine >= 5 else 0
    mois = date_obj.month

    for index_commune, commune in enumerate(COMMUNES):
        lignes.append({
            "date": DATES_LISSEES[index_date + 1],
            "commune": commune,
            "markov_pred": markov_predict[index_commune],
            "markov_total": markov_total,
            "markov_mean": markov_moyenne,
            "jour_semaine": jour_semaine,
            "is_weekend": est_weekend,
            "mois": mois,
            "cas_reel": cas_reels_suiv[index_commune]
        })

df = pd.DataFrame(lignes)

# === Fonction pour ajuster les labels des communes ===
def passer_a_la_ligne_sur_tiret(nom):
    nom = nom.replace(" ", "\n")
    return nom.replace("-", "\n-")

# === Créer un dossier de sauvegarde ===
os.makedirs("visualizations_modele_integre", exist_ok=True)

# === Boucle sur les mois de la période ===
for year in range(2020, 2024):
    for month in range(1, 13):
        if year == 2023 and month > 7:
            break

        temporalite = f"{year}-{month:02d}"
        temporalite_txt = f"{mois_fr[month]} {year}"

        # Filtrer les données du mois
        df_dates_recherchees = df[df["date"].str.startswith(temporalite)].copy()
        if df_dates_recherchees.empty:
            continue

        # Générer features
        df_dates_recherchees["markov_pred_scaled"] = SCALER_PREDICT.transform(df_dates_recherchees[["markov_pred"]])
        df_dates_recherchees["markov_total_scaled"] = SCALER_TOTAL.transform(df_dates_recherchees[["markov_total"]])
        df_dates_recherchees["markov_mean_scaled"] = SCALER_MOYENNE.transform(df_dates_recherchees[["markov_mean"]])

        X = df_dates_recherchees[
            ["markov_pred_scaled", "markov_total_scaled", "markov_mean_scaled", "jour_semaine", "is_weekend", "mois"]
        ]

        # Prédiction
        y_pred_log = MODELE_FINAL.predict(X)
        y_pred = np.expm1(y_pred_log)
        df_dates_recherchees["prediction"] = y_pred

        # Pivot tables
        pivot_pred = df_dates_recherchees.pivot_table(index="commune", columns="date", values="prediction", aggfunc="mean")
        pivot_reel = df_dates_recherchees.pivot_table(index="commune", columns="date", values="cas_reel", aggfunc="mean")
        pivot_diff = pivot_pred - pivot_reel

        pivot_pred.index = [passer_a_la_ligne_sur_tiret(c) for c in pivot_pred.index]
        pivot_reel.index = [passer_a_la_ligne_sur_tiret(c) for c in pivot_reel.index]
        pivot_diff.index = [passer_a_la_ligne_sur_tiret(c) for c in pivot_diff.index]

        # Génération des heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(30, 10), sharey=True)

        sns.heatmap(pivot_pred, cmap="YlOrRd", annot=False, linewidths=0.5, ax=axes[0])
        axes[0].set_title(f"Prédictions modèle final - {temporalite_txt}")
        axes[0].set_xlabel("Date")
        axes[0].tick_params(axis='x', rotation=90)

        sns.heatmap(pivot_reel, cmap="YlOrRd", annot=False, linewidths=0.5, ax=axes[1])
        axes[1].set_title(f"Cas réels - {temporalite_txt}")
        axes[1].set_xlabel("Date")
        axes[1].tick_params(axis='x', rotation=90)

        sns.heatmap(pivot_diff, cmap="coolwarm", center=0, annot=False, linewidths=0.5, ax=axes[2])
        axes[2].set_title("Différence (Prédiction - Réel)")
        axes[2].set_xlabel("Date")
        axes[2].tick_params(axis='x', rotation=90)

        plt.tight_layout()

        # Sauvegarde
        filename = f"visualizations_modele_integre/predictions_modele_integre_{mois_fr[month]}_{year}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Graphique sauvegardé : {filename}")
