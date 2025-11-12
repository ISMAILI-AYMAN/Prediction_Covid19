import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from constantes import *

"""
Ce script calcule l'erreur absolue moyenne des prédictions de cas COVID pour chaque commune
durant une période donnée (ici de novembre 2021 sur 6 mois), et génère un graphe orienté représentant les relations entre communes.

Les étapes principales :
1. Pour chaque date de la période :
   - Calcule les prédictions Markov pour chaque commune.
   - Prépare les features et effectue la prédiction finale.
   - Évalue l'erreur absolue par commune.
2. Moyenne les erreurs absolues pour chaque commune sur la période.
3. Crée un graphe orienté :
   - Les nœuds représentent les communes et sont dimensionnés selon l'erreur moyenne.
   - Les arêtes représentent les relations selon la matrice Markov (poids > seuil, pas de boucle).
4. Affiche le graphe avec une mise en page spring layout et une colorbar des erreurs.
5. Sauvegarde automatique du graphe dans un fichier au nom évocateur.

Pré-requis :
- Les constantes (COMMUNES, DATES_LISSEES, DATES_COVID, etc.) et modèles doivent être importés
  depuis `utils.constantes`.
- Les matrices Markov et les scalers doivent être initialisés.
"""

# === Définir la période de 6 mois à partir de novembre 2021 ===
date_debut = datetime.strptime("2021-11-01", "%Y-%m-%d")
date_fin = datetime.strptime("2022-04-30", "%Y-%m-%d")

# === Calculer prédictions et erreurs sur la période ===
erreurs_par_commune = {comm: [] for comm in COMMUNES}

for index_date, date in enumerate(DATES_LISSEES[:-1]):
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    if date_obj < date_debut or date_obj > date_fin:
        continue

    cas_actuels = np.array([donnees_lissees.get(date, {}).get(c, 0) for c in COMMUNES]).reshape(-1, 1)
    cas_reels_suiv = np.array([donnees_lissees.get(DATES_COVID[index_date + 1], {}).get(c, 0) for c in COMMUNES])

    markov_predict = (X_MEILLEUR_MODELE @ cas_actuels).flatten()

    # features
    markov_total = markov_predict.sum()
    markov_moyenne = markov_predict.mean()
    jour_semaine = date_obj.weekday()
    est_weekend = 1 if jour_semaine >= 5 else 0
    mois = date_obj.month

    for index_commune, commune in enumerate(COMMUNES):
        X_sample = np.array([
            SCALER_PREDICT.transform([[markov_predict[index_commune]]])[0][0],
            SCALER_TOTAL.transform([[markov_total]])[0][0],
            SCALER_MOYENNE.transform([[markov_moyenne]])[0][0],
            jour_semaine,
            est_weekend,
            mois
        ]).reshape(1, -1)

        predict = np.expm1(MODELE_FINAL.predict(X_sample)[0])
        erreur = abs(predict - cas_reels_suiv[index_commune])
        erreurs_par_commune[commune].append(erreur)

# Moyenne des erreurs
moyenne_des_erreurs = {comm: np.mean(errors)
                       if errors else 0 for comm, errors in erreurs_par_commune.items()}

# === Graphe ===
G = nx.DiGraph()

for commune in COMMUNES:
    G.add_node(commune, error=moyenne_des_erreurs[commune])

# Arêtes : uniquement si pas boucle sur soi-même et poids > seuil
seuil = 0.05
for index_commune, src in enumerate(COMMUNES):
    for j, tgt in enumerate(COMMUNES):
        if index_commune == j:
            continue  # on retire les boucles
        poids = X_MEILLEUR_MODELE[index_commune, j]
        if poids > seuil:
            G.add_edge(src, tgt, weight=poids)

# Couleurs et taille des nœuds
erreurs = [G.nodes[n]["error"] for n in G.nodes]
norm_errors = (np.array(erreurs) - np.min(erreurs)) / (np.max(erreurs) - np.min(erreurs) + 1e-6)
# Taille des noeuds va grossir selon erreur
taille_noeuds = 800 + norm_errors * 2000  

pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(16, 12))  # graphe élargi
noeuds = nx.draw_networkx_nodes(
    G, pos,
    node_color=norm_errors,
    cmap=plt.cm.YlOrRd,  # Palette plus claire
    node_size=taille_noeuds
)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, alpha=0.3)

plt.colorbar(noeuds, label="Erreur absolue moyenne (nov 2021 à avr 2022)")
plt.title("Graphe des communes - Erreur absolue moyenne des prédictions (nov 2021 à avr 2022)")

# === Sauvegarde ===
output_dir = "visualizations_erreur_graphe"
os.makedirs(output_dir, exist_ok=True)
filename = f"{output_dir}/graphe_erreur_modele_markov_2021-11_2022-04.png"
plt.savefig(filename, dpi=300, bbox_inches="tight")
plt.show()
print(f"Graphe sauvegardé : {filename}")
