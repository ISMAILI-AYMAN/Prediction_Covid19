import os
import numpy as np
import pandas as pd
import sys
from datetime import datetime
import plotly.express as px

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from constantes import *

# === Période souhaitée ===
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

# === Convertir date ===
df_periode["date"] = pd.to_datetime(df_periode["date"])

# === Générer un graphique par commune ===
for commune in COMMUNES:
    df_c = df_periode[df_periode["commune"] == commune]

    fig = px.line(
        df_c,
        x="date",
        y=["prediction", "cas_reel"],
        labels={"value": "Nombre de cas", "date": "Date", "variable": "Série"},
        title=f"Évolution des cas COVID-19 - {commune}",
        markers=True
    )

    fig.update_layout(
        height=800,  # Hauteur augmentée x2.5 par rapport au précédent 400
        legend_title_text="",
        xaxis=dict(
            tickformat="%Y-%m-%d",
            tickangle=45
        )
    )

    fig.show()
