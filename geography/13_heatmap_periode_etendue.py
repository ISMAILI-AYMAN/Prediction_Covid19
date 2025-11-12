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
Ce script réalise des prédictions de cas pour chaque commune sur la période de mars 2023
et génère des heatmaps comparant les prédictions aux données réelles.

Étapes principales :
1. Construction d'un dataset avec :
   - les prédictions Markov,
   - les variables dérivées (somme, moyenne),
   - les caractéristiques temporelles (jour de la semaine, mois, weekend).
2. Filtrage des données pour mars 2023.
3. Transformation des prédicteurs par des scalers pré-entraînés.
4. Prédiction des cas avec un modèle final pré-entraîné.
5. Agrégation sous forme de matrices pour visualisation :
   - prédictions,
   - cas réels,
   - différence entre prédictions et cas réels.
6. Génération de trois heatmaps côte à côte.

Pré-requis :
- Import des constantes et modèles pré-entraînés depuis `utils.constantes`.
- Données lissées et matrice Markov disponibles.
"""

# Date minimale recherchée
date_min = "2023-01-01"
# Date maximale recherchée
date_max = "2023-05-31"
# Conversion en objets datetime
date_min_dt = datetime.strptime(date_min, "%Y-%m-%d")
date_max_dt = datetime.strptime(date_max, "%Y-%m-%d")
# Dictionnaire des mois en français
mois_fr = {
    1: "janvier", 2: "février", 3: "mars", 4: "avril",
    5: "mai", 6: "juin", 7: "juillet", 8: "août",
    9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre"
}
# Format texte français
date_min_txt = f"{date_min_dt.day:02d} {mois_fr[date_min_dt.month]} {date_min_dt.year}"
date_max_txt = f"{date_max_dt.day:02d} {mois_fr[date_max_dt.month]} {date_max_dt.year}"

# === Préparer dataset ===
lignes = []
for index_date, date in enumerate(DATES_LISSEES[:-1]):
    cas_actuels = np.array([ donnees_lissees.get(date, {}).get(commune, 0) for commune in COMMUNES ]).reshape(-1, 1)
    cas_reels_suiv = np.array([ donnees_lissees.get(DATES_COVID[index_date + 1], {}).get(commune, 0) for commune in COMMUNES ])

    if cas_actuels.shape[0] != X_MEILLEUR_MODELE.shape[1]:
        continue  

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

# === Filtrer Mars 2023 ===
df_periode = df[(df["date"] >= date_min) & (df["date"] <= date_max)].copy()

# === Générer features ===
df_periode["markov_pred_scaled"] = SCALER_PREDICT.transform(df_periode[["markov_pred"]])
df_periode["markov_total_scaled"] = SCALER_TOTAL.transform(df_periode[["markov_total"]])
df_periode["markov_mean_scaled"] = SCALER_MOYENNE.transform(df_periode[["markov_mean"]])

X_mars = df_periode[[ "markov_pred_scaled", "markov_total_scaled", "markov_mean_scaled", "jour_semaine", "is_weekend", "mois"]]

# === Prédire ===
y_pred_log = MODELE_FINAL.predict(X_mars)
y_pred = np.expm1(y_pred_log)
df_periode["prediction"] = y_pred

# === Pivoter ===
pivot_pred = df_periode.pivot_table(index="commune", columns="date", values="prediction", aggfunc="mean")
pivot_reel = df_periode.pivot_table(index="commune", columns="date", values="cas_reel", aggfunc="mean")
pivot_diff = pivot_pred - pivot_reel

# === Heatmaps ===
fig, axes = plt.subplots(1, 3, figsize=(30, 10), sharey=True)

sns.heatmap(pivot_pred, cmap="YlOrRd", annot=False, linewidths=0.5, ax=axes[0])
axes[0].set_title(f"Prédictions entre {date_min_txt} et {date_max_txt}")
axes[0].set_xlabel("Date")
axes[0].tick_params(axis='x', rotation=90)
axes[0].set_xticks(axes[0].get_xticks()[::7])  # Réduit les ticks visibles

sns.heatmap(pivot_reel, cmap="YlOrRd", annot=False, linewidths=0.5, ax=axes[1])
axes[1].set_title(f"Réels entre {date_min_txt} et {date_max_txt}")
axes[1].set_xlabel("Date")
axes[1].tick_params(axis='x', rotation=90)
axes[1].set_xticks(axes[1].get_xticks()[::7])  # Réduit les ticks visibles

sns.heatmap(pivot_diff, cmap="coolwarm", center=0, annot=False, linewidths=0.5, ax=axes[2])
axes[2].set_title("Différence (Prédiction - Réel)")
axes[2].set_xlabel("Date")
axes[2].tick_params(axis='x', rotation=90)
axes[2].set_xticks(axes[2].get_xticks()[::7])  # Réduit les ticks visibles

plt.tight_layout()

# === Créer le dossier de sauvegarde et enregistrer ===
os.makedirs("visualizations_modele_integre", exist_ok=True)
filename = f"visualizations_modele_integre/predictions_modele_integre_{date_min}_{date_max}.png"
plt.savefig(filename, dpi=300, bbox_inches="tight")
plt.close()
print(f"Graphique sauvegardé : {filename}")
plt.show()