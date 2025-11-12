import os 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import matplotlib.dates as mdates

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from constantes import *

"""
Ce script réalise des prédictions de cas pour chaque commune sur la période de janvier à juin 2023
et génère des visualisations comparant les prédictions aux données réelles.

Étapes principales :
1. Construction d'un dataset avec :
   - les prédictions Markov,
   - les variables dérivées (somme, moyenne),
   - les caractéristiques temporelles (jour de la semaine, mois, weekend).
2. Filtrage des données sur la période souhaitée.
3. Transformation des prédicteurs par des scalers pré-entraînés.
4. Prédiction des cas avec un modèle final pré-entraîné.
5. Production de graphes par commune : prédictions + cas réels superposés,
   sauvegardés automatiquement dans un dossier dédié.
"""

# === Période souhaitée ===
# Ici on prend 6 mois : janvier à juin 2023
df_periode = df = None
temporalite_debut = "2023-01-01"
temporalite_fin = "2023-06-30"

# === Préparer dataset ===
lignes = []
for index_date, date in enumerate(DATES_LISSEES[:-1]):
    cas_actuels = np.array([donnees_lissees.get(date, {}).get(commune, 0) for commune in COMMUNES]).reshape(-1, 1)
    cas_reels_suiv = np.array([donnees_lissees.get(DATES_COVID[index_date + 1], {}).get(commune, 0) for commune in COMMUNES])

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

# === Filtrer sur la période ===
df_periode = df[(df["date"] >= temporalite_debut) & (df["date"] <= temporalite_fin)].copy()

# === Générer features ===
df_periode["markov_pred_scaled"] = SCALER_PREDICT.transform(df_periode[["markov_pred"]])
df_periode["markov_total_scaled"] = SCALER_TOTAL.transform(df_periode[["markov_total"]])
df_periode["markov_mean_scaled"] = SCALER_MOYENNE.transform(df_periode[["markov_mean"]])

X_periode = df_periode[[
    "markov_pred_scaled", "markov_total_scaled", "markov_mean_scaled", 
    "jour_semaine", "is_weekend", "mois"
]]

# === Prédire ===
y_pred_log = MODELE_FINAL.predict(X_periode)
y_pred = np.expm1(y_pred_log)
df_periode["prediction"] = y_pred

# === Pivoter les données ===
pivot_pred = df_periode.pivot_table(index="date", columns="commune", values="prediction", aggfunc="mean")
pivot_reel = df_periode.pivot_table(index="date", columns="commune", values="cas_reel", aggfunc="mean")

# Convertir les dates pour l'axe des x
pivot_pred.index = pd.to_datetime(pivot_pred.index)
pivot_reel.index = pd.to_datetime(pivot_reel.index)

# === Créer dossier de sauvegarde ===
output_dir = "visualizations_modele_integre_graphiques"
os.makedirs(output_dir, exist_ok=True)

# === Tracer un graphe par commune : prédictions + cas réels ===
for commune in COMMUNES:
    plt.figure(figsize=(14, 5))  # Largeur standard, hauteur réduite de 15 %

    plt.plot(pivot_pred.index, pivot_pred[commune], label="Prédiction", color="red", marker='o')
    plt.plot(pivot_reel.index, pivot_reel[commune], label="Cas réels", color="blue", marker='x')

    plt.title(f"Évolution des cas COVID-19 - {commune}")
    plt.xlabel("Date")
    plt.ylabel("Nombre de cas")
    plt.legend()

    # Formatage des dates condensé
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    # Sauvegarde automatique
    filename = (
        f"{output_dir}/prediction_modele_integre_{commune.replace(' ', '_').replace('(', '').replace(')', '')}"
        f"_{temporalite_debut}_{temporalite_fin}.png"
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Graphique sauvegardé : {filename}")
