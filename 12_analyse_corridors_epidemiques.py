# -*- coding: utf-8 -*-

"""
12_analyse_corridors_epidemiques.py
------------------------------------

Module d'analyse des corridors épidémiques à partir des matrices de distribution GBM.

Ce module identifie les corridors spatiaux et fonctionnels en analysant la persistance
temporelle (au moins N périodes consécutives) des éléments élevés de la matrice de distribution.
Il calcule également les paramètres de bifurcation β et μ (moyenne temporelle) globalement 
et localement.

Corrections v1.1:
    - Persistance basée sur indices temporels successifs (adapté à toute granularité)
    - μ calculé comme moyenne temporelle (pas somme)
    - Visualisation bipartite pour meilleure lisibilité

Auteur   : Harry FRANCIS (2025)
Contact  : github.com/documentHarry
Version  : 1.1.0 (2025-01-27)
License  : Code source ouvert à la consultation.
           Toute redistribution ou usage commercial est soumis à accord écrit préalable.
"""

__version__ = "1.1.0"

# Compatible Python 3.10+ | Typage PEP 484 | PEP 257 | PEP 8 (88 caractères max)

# --- Librairies standards
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Set
from collections import defaultdict

# --- Librairies tiers
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Modules locaux
from utils_loader import charger_avec_cache
from utils_log import configurer_logging, log_debut_fin
from utils_io import charger_json, sauvegarder_json
from constantes import Cles, COMMUNES, ADJACENTES, emplacement_donnees

# ========== CONFIG LOGGING ==========
NIVEAU_LOG: int = logging.INFO
logger: logging.Logger = configurer_logging(
    NIVEAU_LOG, "12_analyse_corridors_epidemiques")

# ========== CHARGEMENT DES DONNÉES ==========
EMPLACEMENT_DISTRIBUTION: str = emplacement_donnees("gbm_proximity_distribution.json")
EMPLACEMENT_SORTIE: str = emplacement_donnees("corridors_epidemiques_analyse.json")
EMPLACEMENT_RESUME_TEXTE: str = emplacement_donnees("corridors_epidemiques_resume.json")
EMPLACEMENT_RESUME_PERSISTANCE: str = emplacement_donnees("corridors_persistance.json")
EMPLACEMENT_RESUME_CORRIDORS: str = emplacement_donnees("corridors_identifies.json")
EMPLACEMENT_RESUME_METRIQUES: str = emplacement_donnees("corridors_metriques.json")
EMPLACEMENT_VISUALISATIONS: str = "visualizations_corridors"

# Constantes d'analyse
SEUIL_PERSISTANCE_PERIODES: int = 3  # Nombre minimum de périodes consécutives
SEUIL_VALEUR_ELEVEE: float = 0.0  # Sera calculé dynamiquement (percentile)


def sont_adjacentes(commune_i: str, commune_j: str, adjacentes: Dict[str, List[str]]) -> bool:
    """
    Vérifie si deux communes sont adjacentes (partagent une frontière).

    Args:
        commune_i (str): Première commune.
        commune_j (str): Deuxième commune.
        adjacentes (Dict[str, List[str]]): Dictionnaire des adjacences.

    Returns:
        bool: True si les communes sont adjacentes, False sinon.
    """
    return commune_j in adjacentes.get(commune_i, [])


@log_debut_fin(logger)
def charger_matrices_distribution(emplacement: str) -> Dict[str, np.ndarray]:
    """
    Charge les matrices de distribution depuis le fichier JSON.

    Args:
        emplacement (str): Chemin vers le fichier JSON.

    Returns:
        Dict[str, np.ndarray]: Dictionnaire {date: matrice_distribution}.

    Note:
        - Convertit les dictionnaires imbriqués en numpy arrays.
        - Les dates sont triées chronologiquement.
    """
    try:
        logger.info(f"Chargement des matrices de distribution depuis {emplacement}")
        donnees = charger_json(emplacement, logger)
        
        if "distributions" not in donnees:
            raise ValueError("Clé 'distributions' absente du fichier JSON")

        matrices = {}
        for date_str, dist_date in donnees["distributions"].items():
            n = len(COMMUNES)
            matrice = np.zeros((n, n))
            
            for i, commune_i in enumerate(COMMUNES):
                if commune_i in dist_date:
                    for j, commune_j in enumerate(COMMUNES):
                        if commune_j in dist_date[commune_i]:
                            matrice[i, j] = dist_date[commune_i][commune_j]
            
            matrices[date_str] = matrice

        logger.info(f"{len(matrices)} matrices chargées")
        return matrices

    except Exception as e:
        logger.error(f"Erreur lors du chargement : {e}", exc_info=True)
        raise


@log_debut_fin(logger)
def analyser_persistance_corridors(
        matrices: Dict[str, np.ndarray],
        communes: List[str],
        seuil_periodes: int = 3,
        percentile_seuil: float = 75.0) -> Dict[str, Any]:
    """
    Identifie les corridors persistants (valeurs élevées pendant au moins N périodes consécutives).

    Args:
        matrices (Dict[str, np.ndarray]): Matrices de distribution par date.
        communes (List[str]): Liste des communes.
        seuil_periodes (int): Nombre minimum de périodes consécutives (défaut: 3).
        percentile_seuil (float): Percentile pour définir "valeur élevée" (défaut: 75.0).

    Returns:
        Dict[str, Any]: Dictionnaire contenant :
            - corridors_persistants: Liste des corridors avec persistance
            - corridors_spatiaux: Corridors entre communes adjacentes
            - corridors_fonctionnels: Corridors entre communes non adjacentes
            - statistiques: Statistiques sur les corridors identifiés
    
    Note:
        - Utilise des indices temporels successifs au lieu de différences calendaires
        - Adapté à tout type de granularité (journalière, mensuelle, etc.)
    """
    try:
        logger.info(f"Analyse de la persistance des corridors (seuil: {seuil_periodes} périodes)")
        
        # Trier les dates chronologiquement
        dates_triees = sorted(matrices.keys())
        n = len(communes)
        
        logger.info(f"Granularité détectée: {len(dates_triees)} périodes temporelles")
        logger.info(f"Première date: {dates_triees[0]}, Dernière date: {dates_triees[-1]}")
        
        # Calculer le seuil de valeur élevée (percentile sur toutes les matrices)
        toutes_valeurs = []
        for matrice in matrices.values():
            toutes_valeurs.extend(matrice.flatten())
        seuil_valeur = np.percentile(toutes_valeurs, percentile_seuil)
        logger.info(f"Seuil de valeur élevée (percentile {percentile_seuil}%): {seuil_valeur:.4f}")

        # Pour chaque paire (i,j), suivre la persistance temporelle par INDICE
        persistance_par_paire: Dict[Tuple[int, int], List[Tuple[int, str, float]]] = defaultdict(list)
        
        for idx_temps, date_str in enumerate(dates_triees):
            matrice = matrices[date_str]
            for i in range(n):
                for j in range(n):
                    valeur = matrice[i, j]
                    if valeur >= seuil_valeur:
                        persistance_par_paire[(i, j)].append((idx_temps, date_str, valeur))

        # Identifier les périodes de persistance consécutive ≥ seuil_periodes
        corridors_persistants = []
        
        for (i, j), valeurs_temporelles in persistance_par_paire.items():
            # Trier par indice temporel
            valeurs_temporelles.sort(key=lambda x: x[0])
            
            # Trouver les séquences consécutives (indices successifs)
            sequence_courante = []
            indice_precedent = None
            
            for idx_temps, date_str, valeur in valeurs_temporelles:
                if indice_precedent is None:
                    # Première occurrence
                    sequence_courante = [(idx_temps, date_str, valeur)]
                else:
                    # Vérifier si l'indice est consécutif
                    ecart_indices = idx_temps - indice_precedent
                    
                    if ecart_indices == 1:  # Période consécutive
                        sequence_courante.append((idx_temps, date_str, valeur))
                    else:  # Rupture dans la séquence
                        # Vérifier si la séquence précédente est assez longue
                        if len(sequence_courante) >= seuil_periodes:
                            corridors_persistants.append({
                                "source": communes[j],
                                "target": communes[i],
                                "periode_debut": sequence_courante[0][1],
                                "periode_fin": sequence_courante[-1][1],
                                "duree_periodes": len(sequence_courante),
                                "valeur_moyenne": float(np.mean([v for _, _, v in sequence_courante])),
                                "valeur_max": float(np.max([v for _, _, v in sequence_courante])),
                                "indice_source": j,
                                "indice_target": i
                            })
                        # Recommencer une nouvelle séquence
                        sequence_courante = [(idx_temps, date_str, valeur)]
                
                indice_precedent = idx_temps
            
            # Vérifier la dernière séquence (si elle n'a pas été interrompue)
            if len(sequence_courante) >= seuil_periodes:
                corridors_persistants.append({
                    "source": communes[j],
                    "target": communes[i],
                    "periode_debut": sequence_courante[0][1],
                    "periode_fin": sequence_courante[-1][1],
                    "duree_periodes": len(sequence_courante),
                    "valeur_moyenne": float(np.mean([v for _, _, v in sequence_courante])),
                    "valeur_max": float(np.max([v for _, _, v in sequence_courante])),
                    "indice_source": j,
                    "indice_target": i
                })

        logger.info(f"{len(corridors_persistants)} corridors persistants identifiés")

        # Séparer corridors spatiaux (adjacents) et fonctionnels (non adjacents)
        corridors_spatiaux = []
        corridors_fonctionnels = []
        
        for corridor in corridors_persistants:
            source = corridor["source"]
            target = corridor["target"]
            
            if sont_adjacentes(target, source, ADJACENTES):
                corridor["type"] = "spatial"
                corridors_spatiaux.append(corridor)
            else:
                corridor["type"] = "fonctionnel"
                corridors_fonctionnels.append(corridor)

        logger.info(f"  - {len(corridors_spatiaux)} corridors spatiaux (adjacents)")
        logger.info(f"  - {len(corridors_fonctionnels)} corridors fonctionnels (non adjacents)")

        # Sélectionner les Top-3 spatiaux et Top-2 fonctionnels (selon durée ou valeur)
        corridors_spatiaux_tries = sorted(
            corridors_spatiaux,
            key=lambda x: (x["duree_periodes"], x["valeur_moyenne"]),
            reverse=True
        )[:3]
        
        corridors_fonctionnels_tries = sorted(
            corridors_fonctionnels,
            key=lambda x: (x["duree_periodes"], x["valeur_moyenne"]),
            reverse=True
        )[:2]

        logger.info(f"Top-3 corridors spatiaux sélectionnés")
        logger.info(f"Top-2 corridors fonctionnels sélectionnés")

        resultat = {
            "corridors_persistants": corridors_persistants,
            "corridors_spatiaux": corridors_spatiaux_tries,
            "corridors_fonctionnels": corridors_fonctionnels_tries,
            "statistiques": {
                "total_corridors": len(corridors_persistants),
                "spatiaux": len(corridors_spatiaux),
                "fonctionnels": len(corridors_fonctionnels),
                "seuil_periodes": seuil_periodes,
                "seuil_valeur": float(seuil_valeur),
                "percentile_utilise": percentile_seuil,
                "nombre_periodes_analysees": len(dates_triees)
            }
        }

        return resultat

    except Exception as e:
        logger.error(f"Erreur lors de l'analyse de persistance : {e}", exc_info=True)
        raise


@log_debut_fin(logger)
def calculer_metriques_bifurcation(
        matrices: Dict[str, np.ndarray],
        communes: List[str],
        corridors: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calcule les métriques de bifurcation μ, β et R globalement et localement.

    Args:
        matrices (Dict[str, np.ndarray]): Matrices de distribution par date.
        communes (List[str]): Liste des communes.
        corridors (Dict[str, Any]): Résultat de l'analyse des corridors.

    Returns:
        Dict[str, Any]: Dictionnaire contenant :
            - metriques_globales: μ, β, R calculés sur toutes les matrices
            - metriques_corridors: μ, β, R calculés localement dans les zones de corridors
            - metriques_par_date: μ, β, R pour chaque date
    """
    try:
        logger.info("Calcul des métriques de bifurcation (μ, β, R)")
        
        dates_triees = sorted(matrices.keys())
        n = len(communes)
        
        # Métriques globales (moyennes sur toutes les dates pour μ, max pour β)
        mu_global = np.zeros(n)
        beta_global = 0.0
        
        # Métriques par date
        metriques_par_date = {}
        
        for date_str in dates_triees:
            matrice = matrices[date_str]
            
            # μ_i = somme ligne de C (contribution totale reçue par commune i)
            mu_date = matrice.sum(axis=1)
            
            # β = maximum des sommes colonnes (contribution totale émise maximale)
            somme_colonnes = matrice.sum(axis=0)
            beta_date = np.max(somme_colonnes)
            
            # R_i = μ_i * β
            R_date = mu_date * beta_date
            
            metriques_par_date[date_str] = {
                "mu": mu_date.tolist(),
                "beta": float(beta_date),
                "R": R_date.tolist(),
                "somme_colonnes": somme_colonnes.tolist()
            }
            
            mu_global += mu_date
            # β global = maximum de tous les β par date (cohérent avec la définition)
            if beta_date > beta_global:
                beta_global = beta_date
        
        # Moyenner μ global
        mu_global /= len(dates_triees)
        # R global calculé avec β global (max)
        R_global = mu_global * beta_global
        
        metriques_globales = {
            "mu": mu_global.tolist(),
            "beta": float(beta_global),
            "R": R_global.tolist(),
            "mu_par_commune": {
                commune: float(mu_global[i])
                for i, commune in enumerate(communes)
            },
            "R_par_commune": {
                commune: float(R_global[i])
                for i, commune in enumerate(communes)
            }
        }
        
        logger.info(f"β global (maximum): {beta_global:.4f}")
        logger.info(f"μ moyen par commune: min={mu_global.min():.4f}, max={mu_global.max():.4f}")

        # Métriques locales (dans les zones de corridors)
        metriques_corridors = {}
        
        # Pour chaque type de corridor (spatial/fonctionnel)
        for type_corridor in ["corridors_spatiaux", "corridors_fonctionnels"]:
            corridors_list = corridors.get(type_corridor, [])
            
            if not corridors_list:
                continue
            
            # Collecter toutes les communes impliquées dans ces corridors
            communes_impliquees = set()
            for corridor in corridors_list:
                communes_impliquees.add(corridor["source"])
                communes_impliquees.add(corridor["target"])
            
            indices_impliques = [i for i, c in enumerate(communes) if c in communes_impliquees]
            
            if not indices_impliques:
                continue
            
            # Calculer les métriques locales (sous-matrice des communes impliquées)
            # Note: Pour les métriques locales, on utilise la sous-matrice correspondant
            # uniquement aux communes impliquées dans les corridors
            mu_local = np.zeros(len(indices_impliques))
            beta_local = 0.0
            
            for date_str in dates_triees:
                matrice = matrices[date_str]
                # Sous-matrice pour les communes impliquées
                sous_matrice = matrice[np.ix_(indices_impliques, indices_impliques)]
                
                # μ_i local = somme ligne de la sous-matrice
                mu_local_date = sous_matrice.sum(axis=1)
                
                # β local = maximum des sommes colonnes de la sous-matrice
                somme_colonnes_local = sous_matrice.sum(axis=0)
                beta_local_date = np.max(somme_colonnes_local)
                
                mu_local += mu_local_date
                # β local = maximum sur toutes les dates
                if beta_local_date > beta_local:
                    beta_local = beta_local_date
            
            # Moyenner μ local sur toutes les dates
            mu_local /= len(dates_triees)
            # R local calculé avec β local (max)
            R_local = mu_local * beta_local
            
            metriques_corridors[type_corridor] = {
                "communes_impliquees": list(communes_impliquees),
                "mu": mu_local.tolist(),
                "beta": float(beta_local),
                "R": R_local.tolist(),
                "mu_par_commune": {
                    communes[indices_impliques[i]]: float(mu_local[i])
                    for i in range(len(indices_impliques))
                },
                "R_par_commune": {
                    communes[indices_impliques[i]]: float(R_local[i])
                    for i in range(len(indices_impliques))
                }
            }
            
            logger.info(f"  {type_corridor}: β={beta_local:.4f}, μ moyen={mu_local.mean():.4f}")

        resultat = {
            "metriques_globales": metriques_globales,
            "metriques_corridors": metriques_corridors,
            "metriques_par_date": metriques_par_date
        }

        return resultat

    except Exception as e:
        logger.error(f"Erreur lors du calcul des métriques : {e}", exc_info=True)
        raise


@log_debut_fin(logger)
def sauvegarder_resultats(
        corridors: Dict[str, Any],
        metriques: Dict[str, Any],
        emplacement: str) -> None:
    """
    Sauvegarde les résultats de l'analyse dans un fichier JSON.

    Args:
        corridors (Dict[str, Any]): Résultats de l'analyse des corridors.
        metriques (Dict[str, Any]): Métriques de bifurcation calculées.
        emplacement (str): Chemin de sauvegarde.

    Note:
        - Structure JSON avec métadonnées, corridors et métriques.
    """
    try:
        logger.info(f"Sauvegarde des résultats dans {emplacement}")
        
        resultat = {
            "metadata": {
                "create_at": datetime.now().strftime("%Y-%m-%d"),
                "method": "analyse_corridors_epidemiques",
                "seuil_persistance_periodes": SEUIL_PERSISTANCE_PERIODES,
                "communes_count": len(COMMUNES)
            },
            "corridors": corridors,
            "metriques": metriques
        }
        
        sauvegarder_json(resultat, emplacement, logger, ecrasement=True)
        logger.info(f"Résultats sauvegardés : {emplacement}")

    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde : {e}", exc_info=True)
        raise


@log_debut_fin(logger)
def generer_resume_persistance(
        corridors: Dict[str, Any]) -> str:
    """
    Génère un résumé textuel de l'analyse de persistance.

    Args:
        corridors (Dict[str, Any]): Résultats de l'analyse des corridors.

    Returns:
        str: Résumé formaté en texte.
    """
    lignes = []
    lignes.append("=" * 80)
    lignes.append("1. ANALYSE DE LA PERSISTANCE DES CORRIDORS")
    lignes.append("=" * 80)
    lignes.append(f"Date d'analyse: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lignes.append(f"Seuil de persistance: {SEUIL_PERSISTANCE_PERIODES} périodes consécutives")
    lignes.append("")
    
    # Statistiques générales
    stats = corridors.get("statistiques", {})
    lignes.append("--- STATISTIQUES GÉNÉRALES ---")
    lignes.append(f"Total de corridors persistants identifiés: {stats.get('total_corridors', 0)}")
    lignes.append(f"  - Corridors spatiaux (adjacents): {stats.get('spatiaux', 0)}")
    lignes.append(f"  - Corridors fonctionnels (non adjacents): {stats.get('fonctionnels', 0)}")
    lignes.append(f"Seuil de valeur élevée (percentile {stats.get('percentile_utilise', 75)}%): "
                  f"{stats.get('seuil_valeur', 0):.4f}")
    lignes.append("")
    
    # Méthodologie
    lignes.append("--- MÉTHODOLOGIE ---")
    lignes.append("1. Calcul du seuil de valeur élevée:")
    lignes.append("   - Utilisation du percentile 75% de toutes les valeurs des matrices")
    lignes.append("   - Seuil calculé: valeurs ≥ ce percentile sont considérées comme 'élevées'")
    lignes.append("")
    lignes.append("2. Détection de la persistance:")
    lignes.append("   - Pour chaque paire de communes (source → target),")
    lignes.append("   - Identification des séquences consécutives de périodes avec valeurs élevées")
    lignes.append("   - Conservation uniquement des séquences ≥ N périodes consécutives")
    lignes.append("   - Utilise les indices temporels successifs (adapté à toute granularité)")
    lignes.append("")
    lignes.append("3. Classification:")
    lignes.append("   - Corridors spatiaux: communes adjacentes (partagent une frontière)")
    lignes.append("   - Corridors fonctionnels: communes non adjacentes")
    lignes.append("")
    
    # Liste complète des corridors persistants
    lignes.append("--- LISTE COMPLÈTE DES CORRIDORS PERSISTANTS ---")
    corridors_persistants = corridors.get("corridors_persistants", [])
    if corridors_persistants:
        lignes.append(f"Total: {len(corridors_persistants)} corridors")
        lignes.append("")
        for idx, corridor in enumerate(corridors_persistants, 1):
            lignes.append(f"{idx}. {corridor['source']} → {corridor['target']} "
                         f"[{corridor.get('type', 'N/A')}]")
            lignes.append(f"   Durée: {corridor['duree_periodes']} périodes consécutives")
            lignes.append(f"   Période: {corridor['periode_debut']} à {corridor['periode_fin']}")
            lignes.append(f"   Valeur moyenne: {corridor['valeur_moyenne']:.6f}")
            lignes.append(f"   Valeur maximale: {corridor['valeur_max']:.6f}")
            lignes.append("")
    else:
        lignes.append("Aucun corridor persistant identifié.")
    
    lignes.append("=" * 80)
    lignes.append("FIN DU RÉSUMÉ - ANALYSE DE PERSISTANCE")
    lignes.append("=" * 80)
    
    return "\n".join(lignes)


@log_debut_fin(logger)
def generer_resume_persistance_json(
        corridors: Dict[str, Any]) -> Dict[str, Any]:
    """
    Génère un résumé JSON de l'analyse de persistance.

    Args:
        corridors (Dict[str, Any]): Résultats de l'analyse des corridors.

    Returns:
        Dict[str, Any]: Résumé structuré en JSON.
    """
    stats = corridors.get("statistiques", {})
    corridors_persistants = corridors.get("corridors_persistants", [])
    
    return {
        "metadata": {
            "titre": "Analyse de persistance des corridors épidémiques",
            "date_analyse": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seuil_persistance_periodes": SEUIL_PERSISTANCE_PERIODES,
            "methodologie": {
                "seuil_valeur": {
                    "description": "Percentile 75% de toutes les valeurs des matrices",
                    "valeur": stats.get("seuil_valeur", 0.0),
                    "percentile_utilise": stats.get("percentile_utilise", 75.0)
                },
                "detection_persistance": {
                    "description": "Identification des séquences consécutives de périodes avec valeurs élevées (indices temporels)",
                    "critere": f"≥ {SEUIL_PERSISTANCE_PERIODES} périodes consécutives"
                },
                "classification": {
                    "corridors_spatiaux": "Communes adjacentes (partagent une frontière)",
                    "corridors_fonctionnels": "Communes non adjacentes"
                }
            }
        },
        "statistiques_generales": {
            "total_corridors_persistants": stats.get("total_corridors", 0),
            "corridors_spatiaux": stats.get("spatiaux", 0),
            "corridors_fonctionnels": stats.get("fonctionnels", 0),
            "seuil_valeur": float(stats.get("seuil_valeur", 0.0)),
            "percentile_utilise": stats.get("percentile_utilise", 75.0)
        },
        "corridors_persistants": corridors_persistants
    }


@log_debut_fin(logger)
def generer_resume_corridors(
        corridors: Dict[str, Any]) -> str:
    """
    Génère un résumé textuel de l'identification des corridors.

    Args:
        corridors (Dict[str, Any]): Résultats de l'analyse des corridors.

    Returns:
        str: Résumé formaté en texte.
    """
    lignes = []
    lignes.append("=" * 80)
    lignes.append("2. IDENTIFICATION DES CORRIDORS ÉPIDÉMIQUES")
    lignes.append("=" * 80)
    lignes.append(f"Date d'analyse: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lignes.append("")
    
    # Corridors spatiaux
    lignes.append("=" * 80)
    lignes.append("TOP-3 CORRIDORS SPATIAUX (Adjacents - Communes Frontalières)")
    lignes.append("=" * 80)
    corridors_spatiaux = corridors.get("corridors_spatiaux", [])
    if corridors_spatiaux:
        for idx, corridor in enumerate(corridors_spatiaux, 1):
            lignes.append(f"\n{idx}. {corridor['source']} → {corridor['target']}")
            lignes.append(f"   Durée de persistance: {corridor['duree_periodes']} périodes")
            lignes.append(f"   Période: {corridor['periode_debut']} à {corridor['periode_fin']}")
            lignes.append(f"   Valeur moyenne: {corridor['valeur_moyenne']:.6f}")
            lignes.append(f"   Valeur maximale: {corridor['valeur_max']:.6f}")
            lignes.append(f"   Type: Spatial (communes adjacentes)")
    else:
        lignes.append("Aucun corridor spatial identifié.")
    lignes.append("")
    
    # Corridors fonctionnels
    lignes.append("=" * 80)
    lignes.append("TOP-2 CORRIDORS FONCTIONNELS (Non Adjacents)")
    lignes.append("=" * 80)
    corridors_fonctionnels = corridors.get("corridors_fonctionnels", [])
    if corridors_fonctionnels:
        for idx, corridor in enumerate(corridors_fonctionnels, 1):
            lignes.append(f"\n{idx}. {corridor['source']} → {corridor['target']}")
            lignes.append(f"   Durée de persistance: {corridor['duree_periodes']} périodes")
            lignes.append(f"   Période: {corridor['periode_debut']} à {corridor['periode_fin']}")
            lignes.append(f"   Valeur moyenne: {corridor['valeur_moyenne']:.6f}")
            lignes.append(f"   Valeur maximale: {corridor['valeur_max']:.6f}")
            lignes.append(f"   Type: Fonctionnel (communes non adjacentes)")
    else:
        lignes.append("Aucun corridor fonctionnel identifié.")
    lignes.append("")
    
    # Critères de sélection
    lignes.append("--- CRITÈRES DE SÉLECTION ---")
    lignes.append("Les corridors sont triés par:")
    lignes.append("  1. Durée de persistance (périodes consécutives)")
    lignes.append("  2. Valeur moyenne (en cas d'égalité de durée)")
    lignes.append("")
    lignes.append("Top-3 spatiaux: Les 3 corridors adjacents avec la plus longue persistance")
    lignes.append("Top-2 fonctionnels: Les 2 corridors non adjacents avec la plus longue persistance")
    lignes.append("")
    
    lignes.append("=" * 80)
    lignes.append("FIN DU RÉSUMÉ - IDENTIFICATION DES CORRIDORS")
    lignes.append("=" * 80)
    
    return "\n".join(lignes)


@log_debut_fin(logger)
def generer_resume_corridors_json(
        corridors: Dict[str, Any]) -> Dict[str, Any]:
    """
    Génère un résumé JSON de l'identification des corridors.

    Args:
        corridors (Dict[str, Any]): Résultats de l'analyse des corridors.

    Returns:
        Dict[str, Any]: Résumé structuré en JSON.
    """
    corridors_spatiaux = corridors.get("corridors_spatiaux", [])
    corridors_fonctionnels = corridors.get("corridors_fonctionnels", [])
    
    return {
        "metadata": {
            "titre": "Identification des corridors épidémiques",
            "date_analyse": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "criteres_selection": {
                "tri": [
                    "Durée de persistance (périodes consécutives)",
                    "Valeur moyenne (en cas d'égalité de durée)"
                ],
                "top_spatiaux": "Les 3 corridors adjacents avec la plus longue persistance",
                "top_fonctionnels": "Les 2 corridors non adjacents avec la plus longue persistance"
            }
        },
        "top_3_corridors_spatiaux": [
            {
                "rang": idx + 1,
                "source": c["source"],
                "target": c["target"],
                "type": "spatial",
                "duree_periodes": c["duree_periodes"],
                "periode_debut": c["periode_debut"],
                "periode_fin": c["periode_fin"],
                "valeur_moyenne": float(c["valeur_moyenne"]),
                "valeur_max": float(c["valeur_max"]),
                "indice_source": c.get("indice_source", -1),
                "indice_target": c.get("indice_target", -1)
            }
            for idx, c in enumerate(corridors_spatiaux)
        ],
        "top_2_corridors_fonctionnels": [
            {
                "rang": idx + 1,
                "source": c["source"],
                "target": c["target"],
                "type": "fonctionnel",
                "duree_periodes": c["duree_periodes"],
                "periode_debut": c["periode_debut"],
                "periode_fin": c["periode_fin"],
                "valeur_moyenne": float(c["valeur_moyenne"]),
                "valeur_max": float(c["valeur_max"]),
                "indice_source": c.get("indice_source", -1),
                "indice_target": c.get("indice_target", -1)
            }
            for idx, c in enumerate(corridors_fonctionnels)
        ]
    }


@log_debut_fin(logger)
def generer_resume_metriques(
        metriques: Dict[str, Any],
        communes: List[str]) -> str:
    """
    Génère un résumé textuel des métriques de bifurcation.

    Args:
        metriques (Dict[str, Any]): Métriques de bifurcation.
        communes (List[str]): Liste des communes.

    Returns:
        str: Résumé formaté en texte.
    """
    lignes = []
    lignes.append("=" * 80)
    lignes.append("3. CALCUL DES MÉTRIQUES DE BIFURCATION (μ, β, R)")
    lignes.append("=" * 80)
    lignes.append(f"Date d'analyse: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lignes.append("")
    
    # Définitions
    lignes.append("--- DÉFINITIONS DES MÉTRIQUES ---")
    lignes.append("μ_i (mu): Contribution totale reçue par la commune i")
    lignes.append("         = Somme des lignes de la matrice C (moyenne sur toutes les dates)")
    lignes.append("")
    lignes.append("β (beta): Maximum des contributions totales émises")
    lignes.append("         = Maximum des sommes de colonnes (max sur toutes les dates)")
    lignes.append("")
    lignes.append("R_i (ratio): μ_i × β")
    lignes.append("            = Indicateur de bifurcation pour la commune i")
    lignes.append("")
    
    # Métriques globales
    lignes.append("=" * 80)
    lignes.append("MÉTRIQUES GLOBALES")
    lignes.append("=" * 80)
    mg = metriques.get("metriques_globales", {})
    lignes.append(f"\nβ global (maximum des sommes colonnes): {mg.get('beta', 0):.6f}")
    lignes.append("\nTop-10 communes par μ (contribution totale reçue):")
    lignes.append("-" * 80)
    lignes.append(f"{'Commune':<30} {'μ (moyenne)':<15} {'R (μ × β)':<15}")
    lignes.append("-" * 80)
    mu_par_commune = mg.get("mu_par_commune", {})
    R_par_commune = mg.get("R_par_commune", {})
    communes_triees_mu = sorted(
        mu_par_commune.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for commune, mu_val in communes_triees_mu:
        R_val = R_par_commune.get(commune, 0)
        lignes.append(f"{commune:<30} {mu_val:<15.6f} {R_val:<15.6f}")
    lignes.append("")
    
    lignes.append("\nTop-10 communes par R (ratio μ × β):")
    lignes.append("-" * 80)
    lignes.append(f"{'Commune':<30} {'R (μ × β)':<15} {'μ (moyenne)':<15}")
    lignes.append("-" * 80)
    communes_triees_R = sorted(
        R_par_commune.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for commune, R_val in communes_triees_R:
        mu_val = mu_par_commune.get(commune, 0)
        lignes.append(f"{commune:<30} {R_val:<15.6f} {mu_val:<15.6f}")
    lignes.append("")
    
    # Métriques locales
    lignes.append("=" * 80)
    lignes.append("MÉTRIQUES LOCALES (Zones de Corridors)")
    lignes.append("=" * 80)
    metriques_cor = metriques.get("metriques_corridors", {})
    
    for type_corridor, metr_local in metriques_cor.items():
        type_nom = "SPATIAUX" if "spatiaux" in type_corridor else "FONCTIONNELS"
        lignes.append(f"\n--- Corridors {type_nom} ---")
        lignes.append(f"β local: {metr_local.get('beta', 0):.6f}")
        communes_impl = metr_local.get("communes_impliquees", [])
        lignes.append(f"Communes impliquées ({len(communes_impl)}): {', '.join(communes_impl)}")
        lignes.append("\nMétriques par commune dans cette zone:")
        lignes.append("-" * 80)
        lignes.append(f"{'Commune':<30} {'μ local':<15} {'R local':<15}")
        lignes.append("-" * 80)
        mu_local = metr_local.get("mu_par_commune", {})
        R_local = metr_local.get("R_par_commune", {})
        for commune in communes_impl:
            mu_val = mu_local.get(commune, 0)
            R_val = R_local.get(commune, 0)
            lignes.append(f"{commune:<30} {mu_val:<15.6f} {R_val:<15.6f}")
        lignes.append("")
    
    lignes.append("=" * 80)
    lignes.append("FIN DU RÉSUMÉ - MÉTRIQUES DE BIFURCATION")
    lignes.append("=" * 80)
    
    return "\n".join(lignes)


@log_debut_fin(logger)
def generer_resume_metriques_json(
        metriques: Dict[str, Any],
        communes: List[str]) -> Dict[str, Any]:
    """
    Génère un résumé JSON des métriques de bifurcation.

    Args:
        metriques (Dict[str, Any]): Métriques de bifurcation.
        communes (List[str]): Liste des communes.

    Returns:
        Dict[str, Any]: Résumé structuré en JSON.
    """
    mg = metriques.get("metriques_globales", {})
    mu_par_commune = mg.get("mu_par_commune", {})
    R_par_commune = mg.get("R_par_commune", {})
    
    # Top-10 par μ
    communes_triees_mu = sorted(
        mu_par_commune.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    # Top-10 par R
    communes_triees_R = sorted(
        R_par_commune.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    # Métriques locales
    metriques_cor = metriques.get("metriques_corridors", {})
    metriques_locales = {}
    
    for type_corridor, metr_local in metriques_cor.items():
        type_nom = "spatiaux" if "spatiaux" in type_corridor else "fonctionnels"
        communes_impl = metr_local.get("communes_impliquees", [])
        mu_local = metr_local.get("mu_par_commune", {})
        R_local = metr_local.get("R_par_commune", {})
        
        metriques_locales[f"corridors_{type_nom}"] = {
            "beta_local": float(metr_local.get("beta", 0)),
            "communes_impliquees": communes_impl,
            "metriques_par_commune": [
                {
                    "commune": commune,
                    "mu_local": float(mu_local.get(commune, 0)),
                    "R_local": float(R_local.get(commune, 0))
                }
                for commune in communes_impl
            ]
        }
    
    return {
        "metadata": {
            "titre": "Métriques de bifurcation (μ, β, R)",
            "date_analyse": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "definitions": {
                "mu": "Contribution totale reçue par la commune i (somme des lignes, moyenne sur toutes les dates)",
                "beta": "Maximum des contributions totales émises (maximum des sommes de colonnes sur toutes les dates)",
                "R": "μ_i × β (indicateur de bifurcation pour la commune i)"
            }
        },
        "metriques_globales": {
            "beta_global": float(mg.get("beta", 0)),
            "top_10_par_mu": [
                {
                    "rang": idx + 1,
                    "commune": commune,
                    "mu": float(mu_val),
                    "R": float(R_par_commune.get(commune, 0))
                }
                for idx, (commune, mu_val) in enumerate(communes_triees_mu)
            ],
            "top_10_par_R": [
                {
                    "rang": idx + 1,
                    "commune": commune,
                    "R": float(R_val),
                    "mu": float(mu_par_commune.get(commune, 0))
                }
                for idx, (commune, R_val) in enumerate(communes_triees_R)
            ],
            "toutes_communes": {
                commune: {
                    "mu": float(mu_par_commune.get(commune, 0)),
                    "R": float(R_par_commune.get(commune, 0))
                }
                for commune in communes
            }
        },
        "metriques_locales": metriques_locales
    }


@log_debut_fin(logger)
def generer_resume_texte(
        corridors: Dict[str, Any],
        metriques: Dict[str, Any],
        communes: List[str]) -> str:
    """
    Génère un résumé textuel formaté de l'analyse des corridors et métriques.

    Args:
        corridors (Dict[str, Any]): Résultats de l'analyse des corridors.
        metriques (Dict[str, Any]): Métriques de bifurcation.
        communes (List[str]): Liste des communes.

    Returns:
        str: Résumé formaté en texte.
    """
    lignes = []
    lignes.append("=" * 80)
    lignes.append("RÉSUMÉ DE L'ANALYSE DES CORRIDORS ÉPIDÉMIQUES")
    lignes.append("=" * 80)
    lignes.append(f"Date d'analyse: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lignes.append(f"Seuil de persistance: {SEUIL_PERSISTANCE_PERIODES} périodes consécutives")
    lignes.append("")
    
    # Statistiques générales
    stats = corridors.get("statistiques", {})
    lignes.append("--- STATISTIQUES GÉNÉRALES ---")
    lignes.append(f"Total de corridors persistants identifiés: {stats.get('total_corridors', 0)}")
    lignes.append(f"  - Corridors spatiaux (adjacents): {stats.get('spatiaux', 0)}")
    lignes.append(f"  - Corridors fonctionnels (non adjacents): {stats.get('fonctionnels', 0)}")
    lignes.append(f"Seuil de valeur élevée (percentile {stats.get('percentile_utilise', 75)}%): "
                  f"{stats.get('seuil_valeur', 0):.4f}")
    lignes.append("")
    
    # Corridors spatiaux
    lignes.append("=" * 80)
    lignes.append("TOP-3 CORRIDORS SPATIAUX (Adjacents - Communes Frontalières)")
    lignes.append("=" * 80)
    corridors_spatiaux = corridors.get("corridors_spatiaux", [])
    if corridors_spatiaux:
        for idx, corridor in enumerate(corridors_spatiaux, 1):
            lignes.append(f"\n{idx}. {corridor['source']} → {corridor['target']}")
            lignes.append(f"   Durée de persistance: {corridor['duree_periodes']} périodes")
            lignes.append(f"   Période: {corridor['periode_debut']} à {corridor['periode_fin']}")
            lignes.append(f"   Valeur moyenne: {corridor['valeur_moyenne']:.4f}")
            lignes.append(f"   Valeur maximale: {corridor['valeur_max']:.4f}")
    else:
        lignes.append("Aucun corridor spatial identifié.")
    lignes.append("")
    
    # Corridors fonctionnels
    lignes.append("=" * 80)
    lignes.append("TOP-2 CORRIDORS FONCTIONNELS (Non Adjacents)")
    lignes.append("=" * 80)
    corridors_fonctionnels = corridors.get("corridors_fonctionnels", [])
    if corridors_fonctionnels:
        for idx, corridor in enumerate(corridors_fonctionnels, 1):
            lignes.append(f"\n{idx}. {corridor['source']} → {corridor['target']}")
            lignes.append(f"   Durée de persistance: {corridor['duree_periodes']} périodes")
            lignes.append(f"   Période: {corridor['periode_debut']} à {corridor['periode_fin']}")
            lignes.append(f"   Valeur moyenne: {corridor['valeur_moyenne']:.4f}")
            lignes.append(f"   Valeur maximale: {corridor['valeur_max']:.4f}")
    else:
        lignes.append("Aucun corridor fonctionnel identifié.")
    lignes.append("")
    
    # Métriques globales
    lignes.append("=" * 80)
    lignes.append("MÉTRIQUES GLOBALES DE BIFURCATION (μ, β, R)")
    lignes.append("=" * 80)
    mg = metriques.get("metriques_globales", {})
    lignes.append(f"\nβ (maximum des sommes colonnes): {mg.get('beta', 0):.6f}")
    lignes.append("\nTop-10 communes par μ (contribution totale reçue):")
    lignes.append("-" * 80)
    lignes.append(f"{'Commune':<30} {'μ (moyenne)':<15} {'R (μ × β)':<15}")
    lignes.append("-" * 80)
    mu_par_commune = mg.get("mu_par_commune", {})
    R_par_commune = mg.get("R_par_commune", {})
    communes_triees_mu = sorted(
        mu_par_commune.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for commune, mu_val in communes_triees_mu:
        R_val = R_par_commune.get(commune, 0)
        lignes.append(f"{commune:<30} {mu_val:<15.6f} {R_val:<15.6f}")
    lignes.append("")
    
    lignes.append("\nTop-10 communes par R (ratio μ × β):")
    lignes.append("-" * 80)
    lignes.append(f"{'Commune':<30} {'R (μ × β)':<15} {'μ (moyenne)':<15}")
    lignes.append("-" * 80)
    communes_triees_R = sorted(
        R_par_commune.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for commune, R_val in communes_triees_R:
        mu_val = mu_par_commune.get(commune, 0)
        lignes.append(f"{commune:<30} {R_val:<15.6f} {mu_val:<15.6f}")
    lignes.append("")
    
    # Métriques locales
    lignes.append("=" * 80)
    lignes.append("MÉTRIQUES LOCALES (Zones de Corridors)")
    lignes.append("=" * 80)
    metriques_cor = metriques.get("metriques_corridors", {})
    
    for type_corridor, metr_local in metriques_cor.items():
        type_nom = "SPATIAUX" if "spatiaux" in type_corridor else "FONCTIONNELS"
        lignes.append(f"\n--- Corridors {type_nom} ---")
        lignes.append(f"β local: {metr_local.get('beta', 0):.6f}")
        communes_impl = metr_local.get("communes_impliquees", [])
        lignes.append(f"Communes impliquées ({len(communes_impl)}): {', '.join(communes_impl)}")
        lignes.append("\nMétriques par commune dans cette zone:")
        lignes.append("-" * 80)
        lignes.append(f"{'Commune':<30} {'μ local':<15} {'R local':<15}")
        lignes.append("-" * 80)
        mu_local = metr_local.get("mu_par_commune", {})
        R_local = metr_local.get("R_par_commune", {})
        for commune in communes_impl:
            mu_val = mu_local.get(commune, 0)
            R_val = R_local.get(commune, 0)
            lignes.append(f"{commune:<30} {mu_val:<15.6f} {R_val:<15.6f}")
        lignes.append("")
    
    lignes.append("=" * 80)
    lignes.append("FIN DU RÉSUMÉ")
    lignes.append("=" * 80)
    
    return "\n".join(lignes)


@log_debut_fin(logger)
def generer_resume_complet_json(
        corridors: Dict[str, Any],
        metriques: Dict[str, Any],
        communes: List[str]) -> Dict[str, Any]:
    """
    Génère un résumé JSON complet combinant toutes les sections.

    Args:
        corridors (Dict[str, Any]): Résultats de l'analyse des corridors.
        metriques (Dict[str, Any]): Métriques de bifurcation.
        communes (List[str]): Liste des communes.

    Returns:
        Dict[str, Any]: Résumé complet structuré en JSON.
    """
    return {
        "metadata": {
            "titre": "Résumé complet de l'analyse des corridors épidémiques",
            "date_analyse": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seuil_persistance_periodes": SEUIL_PERSISTANCE_PERIODES,
            "nombre_communes": len(communes)
        },
        "persistance": generer_resume_persistance_json(corridors),
        "corridors_identifies": generer_resume_corridors_json(corridors),
        "metriques": generer_resume_metriques_json(metriques, communes)
    }


@log_debut_fin(logger)
def sauvegarder_resumes_separes(
        corridors: Dict[str, Any],
        metriques: Dict[str, Any],
        communes: List[str]) -> None:
    """
    Génère et sauvegarde les résumés dans des fichiers séparés pour chaque section.

    Args:
        corridors (Dict[str, Any]): Résultats de l'analyse des corridors.
        metriques (Dict[str, Any]): Métriques de bifurcation.
        communes (List[str]): Liste des communes.
    """
    try:
        # 1. Résumé de persistance (JSON)
        resume_persistance = generer_resume_persistance_json(corridors)
        sauvegarder_json(resume_persistance, EMPLACEMENT_RESUME_PERSISTANCE, logger, ecrasement=True)
        logger.info(f"Résumé persistance sauvegardé (JSON) : {EMPLACEMENT_RESUME_PERSISTANCE}")
        
        # 2. Résumé des corridors identifiés (JSON)
        resume_corridors = generer_resume_corridors_json(corridors)
        sauvegarder_json(resume_corridors, EMPLACEMENT_RESUME_CORRIDORS, logger, ecrasement=True)
        logger.info(f"Résumé corridors sauvegardé (JSON) : {EMPLACEMENT_RESUME_CORRIDORS}")
        
        # 3. Résumé des métriques (JSON)
        resume_metriques = generer_resume_metriques_json(metriques, communes)
        sauvegarder_json(resume_metriques, EMPLACEMENT_RESUME_METRIQUES, logger, ecrasement=True)
        logger.info(f"Résumé métriques sauvegardé (JSON) : {EMPLACEMENT_RESUME_METRIQUES}")
        
        # 4. Résumé complet (JSON)
        resume_complet = generer_resume_complet_json(corridors, metriques, communes)
        sauvegarder_json(resume_complet, EMPLACEMENT_RESUME_TEXTE, logger, ecrasement=True)
        logger.info(f"Résumé complet sauvegardé (JSON) : {EMPLACEMENT_RESUME_TEXTE}")

    except Exception as e:
        logger.error(f"Erreur lors de la génération/sauvegarde des résumés : {e}", exc_info=True)
        raise


@log_debut_fin(logger)
def visualiser_corridors_sankey(
        corridors: Dict[str, Any],
        date_str: str = None,
        emplacement_dossier: str = None) -> None:
    """
    Génère un diagramme Sankey pour les corridors identifiés.
    
    Args:
        corridors (Dict[str, Any]): Résultats de l'analyse des corridors.
        date_str (str, optional): Date pour le nom du fichier.
        emplacement_dossier (str, optional): Dossier de sauvegarde.
    
    Note:
        - Sauvegarde au format HTML (interactif) et PNG (statique).
        - Les corridors spatiaux sont en rouge, fonctionnels en bleu.
    """
    try:
        if emplacement_dossier is None:
            emplacement_dossier = EMPLACEMENT_VISUALISATIONS
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        os.makedirs(emplacement_dossier, exist_ok=True)
        
        # Préparer les données
        corridors_spatiaux = corridors.get("corridors_spatiaux", [])
        corridors_fonctionnels = corridors.get("corridors_fonctionnels", [])
        tous_corridors = corridors_spatiaux + corridors_fonctionnels
        
        if not tous_corridors:
            logger.warning("Aucun corridor à visualiser")
            return
        
        # Créer les labels uniques
        communes_set = set()
        for cor in tous_corridors:
            communes_set.add(cor["source"])
            communes_set.add(cor["target"])
        labels = sorted(list(communes_set))
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        
        # Préparer les liens
        source_indices = []
        target_indices = []
        values = []
        link_colors = []
        
        for cor in tous_corridors:
            source_idx = label_to_idx[cor["source"]]
            target_idx = label_to_idx[cor["target"]]
            valeur = cor["valeur_moyenne"]
            
            source_indices.append(source_idx)
            target_indices.append(target_idx)
            values.append(valeur)
            
            # Couleur selon le type
            if cor.get("type") == "spatial":
                link_colors.append("rgba(255, 100, 100, 0.7)")  # Rouge pour spatial
            else:
                link_colors.append("rgba(100, 100, 255, 0.7)")  # Bleu pour fonctionnel
        
        # Créer le diagramme Sankey
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color="lightblue"
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color=link_colors
            )
        )])
        
        fig.update_layout(
            title_text=f"Corridors Épidémiques - {date_str}<br>"
                      f"<sub>Rouge: Spatiaux | Bleu: Fonctionnels</sub>",
            font_size=12,
            height=900,
            width=1400
        )
        
        # Sauvegarder HTML
        filename_html = os.path.join(emplacement_dossier, f"corridors_sankey_{date_str.replace('-', '_')}.html")
        fig.write_html(filename_html)
        logger.info(f"Diagramme Sankey corridors (HTML) sauvegardé : {filename_html}")
        
        # Sauvegarder PNG
        try:
            import kaleido
            filename_png = os.path.join(emplacement_dossier, f"corridors_sankey_{date_str.replace('-', '_')}.png")
            fig.write_image(filename_png, width=1400, height=900, scale=2)
            logger.info(f"Diagramme Sankey corridors (PNG) sauvegardé : {filename_png}")
        except ImportError:
            logger.error(
                "Le package 'kaleido' n'est pas installé. "
                "Installez-le avec: pip install kaleido"
            )
            raise
        except Exception as e:
            logger.error(
                f"Impossible de sauvegarder le Sankey en PNG : {e}. "
                f"Le fichier HTML est toujours disponible : {filename_html}"
            )
            raise
            
    except Exception as e:
        logger.error(f"Erreur lors de la génération du diagramme Sankey : {e}", exc_info=True)
        raise


@log_debut_fin(logger)
def visualiser_corridors_bipartite(
        corridors: Dict[str, Any],
        date_str: str = None,
        emplacement_dossier: str = None) -> None:
    """
    Génère un diagramme bipartite pour les corridors identifiés.
    
    Args:
        corridors (Dict[str, Any]): Résultats de l'analyse des corridors.
        date_str (str, optional): Date pour le nom du fichier.
        emplacement_dossier (str, optional): Dossier de sauvegarde.
    
    Note:
        - Représentation bipartite: sources à gauche, cibles à droite
        - Sauvegarde au format HTML (interactif) et PNG (statique).
        - Les corridors spatiaux sont en rouge, fonctionnels en bleu.
        - Épaisseur des liens proportionnelle à l'intensité du flux.
    """
    try:
        if emplacement_dossier is None:
            emplacement_dossier = EMPLACEMENT_VISUALISATIONS
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        os.makedirs(emplacement_dossier, exist_ok=True)
        
        # Préparer les données
        corridors_spatiaux = corridors.get("corridors_spatiaux", [])
        corridors_fonctionnels = corridors.get("corridors_fonctionnels", [])
        tous_corridors = corridors_spatiaux + corridors_fonctionnels
        
        if not tous_corridors:
            logger.warning("Aucun corridor à visualiser")
            return
        
        # Créer des ensembles de communes sources et cibles
        communes_sources = set()
        communes_cibles = set()
        for cor in tous_corridors:
            communes_sources.add(cor["source"])
            communes_cibles.add(cor["target"])
        
        # Créer des listes ordonnées
        sources_list = sorted(list(communes_sources))
        cibles_list = sorted(list(communes_cibles))
        
        n_sources = len(sources_list)
        n_cibles = len(cibles_list)
        
        # Palette de couleurs pour les communes (cohérente)
        import matplotlib.colors as mcolors
        colors_communes = list(mcolors.TABLEAU_COLORS.values())
        commune_to_color = {}
        toutes_communes = sorted(list(set(sources_list + cibles_list)))
        for idx, commune in enumerate(toutes_communes):
            commune_to_color[commune] = colors_communes[idx % len(colors_communes)]
        
        # Positions: sources à gauche (x=0), cibles à droite (x=2)
        # Espacées verticalement
        y_sources = np.linspace(0, n_sources - 1, n_sources)
        y_cibles = np.linspace(0, n_cibles - 1, n_cibles)
        
        # Normaliser les y pour centrer
        if n_sources > n_cibles:
            offset = (n_sources - n_cibles) / 2
            y_cibles = y_cibles + offset
        elif n_cibles > n_sources:
            offset = (n_cibles - n_sources) / 2
            y_sources = y_sources + offset
        
        source_to_y = {src: y_sources[idx] for idx, src in enumerate(sources_list)}
        cible_to_y = {cib: y_cibles[idx] for idx, cib in enumerate(cibles_list)}
        
        fig = go.Figure()
        
        # Ajouter les liens (flux) en premier (pour qu'ils soient en arrière-plan)
        max_val = max([cor["valeur_moyenne"] for cor in tous_corridors]) if tous_corridors else 1.0
        
        for cor in tous_corridors:
            source = cor["source"]
            target = cor["target"]
            valeur = cor["valeur_moyenne"]
            
            x0, y0 = 0, source_to_y[source]
            x1, y1 = 2, cible_to_y[target]
            
            # Créer une courbe de Bézier pour les liens
            t = np.linspace(0, 1, 100)
            # Point de contrôle au milieu
            mid_x = 1.0
            mid_y = (y0 + y1) / 2
            
            # Courbe quadratique de Bézier
            bezier_x = (1-t)**2 * x0 + 2*(1-t)*t * mid_x + t**2 * x1
            bezier_y = (1-t)**2 * y0 + 2*(1-t)*t * mid_y + t**2 * y1
            
            # Épaisseur proportionnelle à la valeur
            line_width = max(1, int(valeur * 20 / max_val))
            
            # Couleur selon le type
            if cor.get("type") == "spatial":
                color = "rgba(255, 80, 80, 0.6)"  # Rouge pour spatial
            else:
                color = "rgba(80, 80, 255, 0.6)"  # Bleu pour fonctionnel
            
            fig.add_trace(go.Scatter(
                x=bezier_x,
                y=bezier_y,
                mode='lines',
                line=dict(width=line_width, color=color),
                hoverinfo='text',
                text=f"{source} → {target}<br>Valeur: {valeur:.6f}<br>Type: {cor.get('type', 'N/A')}<br>Durée: {cor.get('duree_periodes', 'N/A')} périodes",
                showlegend=False
            ))
        
        # Ajouter les nœuds sources (gauche)
        for idx, source in enumerate(sources_list):
            fig.add_trace(go.Scatter(
                x=[0],
                y=[y_sources[idx]],
                mode='markers+text',
                marker=dict(
                    size=25,
                    color=commune_to_color[source],
                    line=dict(width=2, color='black')
                ),
                text=source,
                textposition="middle left",
                textfont=dict(size=9, color='black'),
                name=source,
                showlegend=False,
                hoverinfo='text',
                hovertext=f"Source: {source}"
            ))
        
        # Ajouter les nœuds cibles (droite)
        for idx, cible in enumerate(cibles_list):
            fig.add_trace(go.Scatter(
                x=[2],
                y=[y_cibles[idx]],
                mode='markers+text',
                marker=dict(
                    size=25,
                    color=commune_to_color[cible],
                    line=dict(width=2, color='black')
                ),
                text=cible,
                textposition="middle right",
                textfont=dict(size=9, color='black'),
                name=cible,
                showlegend=False,
                hoverinfo='text',
                hovertext=f"Cible: {cible}"
            ))
        
        fig.update_layout(
            title_text=f"Diagramme Bipartite - Corridors Épidémiques - {date_str}<br>"
                      f"<sub>Rouge: Spatiaux | Bleu: Fonctionnels</sub>",
            font_size=12,
            height=max(800, n_sources * 40, n_cibles * 40),
            width=1400,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-0.5, 2.5]
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-1, max(max(y_sources) if len(y_sources) > 0 else 0, 
                              max(y_cibles) if len(y_cibles) > 0 else 0) + 1]
            ),
            plot_bgcolor='white',
            showlegend=False,
            hovermode='closest'
        )
        
        # Sauvegarder HTML
        filename_html = os.path.join(emplacement_dossier, f"corridors_bipartite_{date_str.replace('-', '_')}.html")
        fig.write_html(filename_html)
        logger.info(f"Diagramme Bipartite corridors (HTML) sauvegardé : {filename_html}")
        
        # Sauvegarder PNG
        try:
            import kaleido
            filename_png = os.path.join(emplacement_dossier, f"corridors_bipartite_{date_str.replace('-', '_')}.png")
            fig.write_image(
                filename_png,
                width=1400,
                height=max(800, n_sources * 40, n_cibles * 40),
                scale=2
            )
            logger.info(f"Diagramme Bipartite corridors (PNG) sauvegardé : {filename_png}")
        except ImportError:
            logger.error(
                "Le package 'kaleido' n'est pas installé. "
                "Installez-le avec: pip install kaleido"
            )
            raise
        except Exception as e:
            logger.error(
                f"Impossible de sauvegarder le Bipartite en PNG : {e}. "
                f"Le fichier HTML est toujours disponible : {filename_html}"
            )
            raise
            
    except Exception as e:
        logger.error(f"Erreur lors de la génération du diagramme Bipartite : {e}", exc_info=True)
        raise


@log_debut_fin(logger)
def main() -> None:
    """
    Fonction principale orchestrant l'analyse complète des corridors épidémiques.

    Étapes:
        1. Charge les matrices de distribution depuis le JSON.
        2. Analyse la persistance des éléments élevés (≥ N périodes consécutives).
        3. Identifie les corridors spatiaux (Top-3) et fonctionnels (Top-2).
        4. Calcule les métriques μ (moyenne), β et R globalement et localement.
        5. Sauvegarde les résultats en JSON.
        6. Génère les visualisations Sankey et Bipartite.
        7. Affiche un résumé de l'analyse.
    """
    try:
        logger.info("==== DÉMARRAGE 12_analyse_corridors_epidemiques ====")

        # 1. Charger les matrices de distribution
        matrices = charger_matrices_distribution(EMPLACEMENT_DISTRIBUTION)

        # 2. Analyser la persistance et identifier les corridors
        corridors = analyser_persistance_corridors(
            matrices, COMMUNES, seuil_periodes=SEUIL_PERSISTANCE_PERIODES
        )

        # 3. Calculer les métriques de bifurcation
        metriques = calculer_metriques_bifurcation(matrices, COMMUNES, corridors)

        # 4. Sauvegarder les résultats
        sauvegarder_resultats(corridors, metriques, EMPLACEMENT_SORTIE)

        # 5. Générer et sauvegarder les résumés dans des fichiers séparés
        sauvegarder_resumes_separes(corridors, metriques, COMMUNES)

        # 6. Générer les visualisations Sankey et Bipartite
        logger.info("Génération des visualisations Sankey et Bipartite pour les corridors")
        try:
            visualiser_corridors_sankey(corridors)
            visualiser_corridors_bipartite(corridors)
        except Exception as e:
            logger.warning(
                f"Erreur lors de la génération des visualisations: {e}. "
                f"Les visualisations ne sont pas critiques pour l'analyse."
            )
            # Ne pas arrêter le script si les visualisations échouent

        logger.info("==== Fin du traitement 12_analyse_corridors_epidemiques ====")

    except Exception as e:
        logger.critical(f"[FATAL ERROR - main()] : {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

