# -*- coding: utf-8 -*-

"""
02_data_loader.py
-----------------
Script de nettoyage et de structuration des données brutes COVID-19 par commune pour la
région de Bruxelles-Capitale. Organise, filtre, corrige et sauvegarde les séries de cas.

Ce module fournit :
    - Le chargement robuste d’un fichier JSON de données COVID brutes.
    - La restructuration sous forme {date: {commune: nombre de cas}}, avec correction des cas anonymisés.
    - Un contrôle automatique des communes présentes/absentes et du format des enregistrements.
    - La sauvegarde sécurisée du fichier nettoyé, avec gestion de l’écrasement et logs détaillés.

Objectif :
    - Nettoyer, homogénéiser et fiabiliser les données brutes COVID pour toute analyse ultérieure.
    - Garantir une traçabilité complète des étapes et des éventuelles anomalies via le logging.

Fonctionnalités principales :
    - Validation de chaque ligne de donnée (présence des champs essentiels, format des cas).
    - Remplacement automatique des valeurs "<5" (anonymisées) par 1, gestion des valeurs invalides.
    - Filtrage optionnel sur une plage de dates, adaptable à d’autres régions ou périodes.
    - Sauvegarde sous forme JSON structurée, adaptée aux scripts d’analyse ou de visualisation.
    - Logging pédagogique de toutes les étapes, erreurs et conseils de maintenance.

Pré-requis :
    - Un fichier d’entrée (JSON brut COVID) à l’emplacement "Data/C0VID19BE_CASES_MUNI.json".
    - Un dossier "Data/" accessible en écriture pour sauvegarder le fichier nettoyé.
    - Python 3.9+, dépendances standard (logging, os, collections…).

Philosophie :
    - Toutes les transformations “critiques” sont explicitées et loguées pour faciliter l’audit.
    - Aucune variable globale cachée : tout paramètre est clairement défini ou passé en argument.

Utilisation typique :
    >>> $ python 02_data_loader.py
    >>> # ou : importer organiser_donnees_par_commune pour réutiliser la logique ailleurs

Best Practice :
    - Utiliser les logs pour détecter toute anomalie dans les données (ex : commune manquante).
    - Adaptez le paramètre “ecraser_la_sauvegarde_existante” selon votre pipeline de travail.

Conseils :
    - Pour d’autres jeux de données, adaptez la liste des communes ou les règles de correction.
    - En cas de modification du format source, mettez à jour la fonction de validation.

Limitations :
    - Le module ne traite que la Région Bruxelles-Capitale (19 communes, noms exacts attendus).
    - Ne fait pas d’analyse temporelle ou géographique : réservé au prétraitement de la donnée brute.

Maintenance :
    - Ajoutez toute correction ou nouveau champ ici pour garantir la cohérence du projet.
    - Suivez les logs pour toute évolution future ou intégration dans un workflow automatisé.

Documentation :
    - Toutes les fonctions sont documentées (format Google), avec exemples pour l’apprentissage.
    - Pour la structure des données source, se référer à la documentation officielle Sciensano.

Auteur   : Harry FRANCIS (2025)
Contact  : github.com/documentHarry
Version  : 1.0.0 (2025-07-23)
License  : Code source ouvert à la consultation.
           Toute redistribution ou usage commercial est soumis à accord écrit préalable.
"""


__version__ = "1.0.0"

# Compatible Python 3.9+  | Typage PEP 484 | Type PEP257

# --- Librairies standards
import logging
import os
from collections import OrderedDict
from typing import Any, Optional

# --- Modules locaux
from utils_io import charger_json, sauvegarder_json, creer_dossier_si_absent
from utils_log import configurer_logging, log_debut_fin
from utils_dates import filtrer_liste_par_date, str_vers_datetime
from constantes import (
    COMMUNES, EMPLACEMENT_DONNEES_BRUTES_COVID, EMPLACEMENT_DONNEES_COVID, 
    EnregistrementCovidBrut
)

# === Paramètres de configuration ===
#: Active le mode détaillé pour afficher les logs/statistiques sur le traitement.
MODE_DETAILLE: bool = True
#: Niveau du logging (INFO, DEBUG, WARNING...).
NIVEAU_LOG: int = logging.INFO
logger: logging.Logger = configurer_logging(NIVEAU_LOG, "02_data_loader")
ecraser_la_sauvegarde_existante: bool = True
LIGNE_PARAM_ECRASEMENT = 63


def valider_enregistrement(enregistrement: dict ) -> Optional[tuple[str, str, Any]]:
    """
    Vérifie qu’un enregistrement brut contient les trois champs essentiels (DATE, commune, cas).
    Retourne (date, commune, cas) si tous sont présents, sinon None.

    Args:
        enregistrement (dict): Dictionnaire à valider (issu du JSON brut).

    Returns:
        Optional[tuple[str, str, Any]]: tuple (date, commune, cas) si tous les champs sont là.
        Retourne None si l’enregistrement est incomplet ou mal formé.

    Note:
        - Un warning est loggué si le format n’est pas respecté (utile pour déboguer).
        - Ne lève jamais d’exception, retourne juste None si invalide.

    Example:
        >>> valider_enregistrement({"DATE": "2023-03-01", "TX_DESCR_FR": "Bruxelles", "CASES": 8})
        ('2023-03-01', 'Bruxelles', 8)
        >>> valider_enregistrement({"DATE": "2023-03-01", "CASES": 8})
        None
        >>> valider_enregistrement("bidule")
        None

    Étapes:
        1. Vérifie le type (doit être dict).
        2. Vérifie la présence des clés "DATE", "TX_DESCR_FR", "CASES".
        3. Retourne le tuple si tout est présent, sinon None.

    Tips:
        - À appliquer sur chaque ligne du JSON brut avant toute structuration.
        - Permet de sécuriser le pipeline de nettoyage.

    Utilisation:
        À utiliser dans toute boucle de nettoyage/contrôle sur données COVID brutes.

    Limitation:
        - Ne vérifie pas le contenu (valeur numérique, format date, etc.).
        - Toute ligne non conforme est ignorée (pas d’exception).
    """
    if not isinstance(enregistrement, dict):
        logger.warning(f"Élément ignoré (non dict) : {enregistrement}")
        return None
    if ("DATE" not in enregistrement 
        or "TX_DESCR_FR" not in enregistrement 
        or "CASES" not in enregistrement):
        logger.warning(f"Ligne ignorée (champ manquant) : {enregistrement}")
        return None
    return (enregistrement["DATE"], enregistrement["TX_DESCR_FR"], 
            enregistrement["CASES"])



def convertir_cases_en_int(cas: Any, commune: str, date: str ) -> int:
    """
    Transforme le champ CASES en entier. Gère les valeurs anonymisées "<5" et les erreurs de type.

    Args:
        cas (Any): Valeur du champ CASES (int attendu, ou str "<5", ou autre).
        commune (str): Nom de la commune (pour le log en cas d’erreur).
        date (str): Date de l’enregistrement (pour le log en cas d’erreur).

    Returns:
        int: Nombre de cas, 1 si "<5", 0 si la valeur est invalide/non convertible.

    Note:
        - Les valeurs "<5" (anonymisation officielle) sont transformées en 1.
        - Si la conversion en int échoue, un warning est loggué, valeur 0 retournée.
        - Jamais d’exception levée.

    Example:
        >>> convertir_cases_en_int("<5", "Bruxelles", "2023-03-01")
        1
        >>> convertir_cases_en_int(7, "Ixelles", "2023-03-01")
        7
        >>> convertir_cases_en_int("abc", "Jette", "2023-03-01")
        0

    Étapes:
        1. Si cas == "<5", retourne 1 (politique d’anonymisation).
        2. Tente la conversion en int, retourne la valeur si succès.
        3. Si erreur (ValueError ou TypeError), log un warning et retourne 0.

    Tips:
        - Cette fonction garantit qu’aucun champ CASES n’est absent du dictionnaire final.
        - Centralise la gestion de l’anonymisation et des erreurs de saisie.

    Utilisation:
        À appeler sur chaque valeur CASES lors de la structuration de la donnée brute.

    Limitation:
        - La règle "<5" → 1 peut être adaptée selon la politique de données.
        - Ne distingue pas entre 0 réel et erreur de saisie.

    See also:
        - organiser_donnees_par_commune
    """
    if cas == "<5":
        return 1
    try:
        return int(cas)
    except (ValueError, TypeError):
        logger.warning(f"Valeur invalide pour {commune} le {date}: "
                       f"{cas} (remplacé par 0)")
        return 0


@log_debut_fin(logger)
def organiser_donnees_par_commune(
        liste_enregistrements_bruts: list[EnregistrementCovidBrut],
        liste_communes: list[str] = COMMUNES, logger = logger,
        mode_detaille: bool = True, date_debut: Optional[str] = None, 
        date_fin: Optional[str] = None) -> dict[str, dict[str, int]]:
    """
    Structure les données brutes en dict {date: {commune: cas}}, avec corrections et filtrages.

    Args:
        liste_enregistrements_bruts (list[EnregistrementCovidBrut]): Liste d'enregistrements bruts.
        liste_communes (list[str], optionnel): Communes à garder. Par défaut, toutes.
        mode_detaille (bool): Affiche logs/statistiques si True (pour le suivi du traitement).
        date_debut (str, optionnel): Date de début incluse (format 'YYYY-MM-DD'), ou None.
        date_fin (str, optionnel): Date de fin incluse (format 'YYYY-MM-DD'), ou None.

    Returns:
        dict[str, dict[str, int]]: Dictionnaire final, clés=dates (str),
        valeurs=dictionnaires {commune: nombre de cas (int)}.

    Note:
        - Log toutes les étapes et anomalies (ex : commune absente).
        - Peut être adaptée à d'autres régions (adapter liste_communes).

    Example:
        >>> organiser_donnees_par_commune(
        ...     [{"DATE": "2023-03-01", "TX_DESCR_FR": "Bruxelles", "CASES": "<5"},
        ...      {"DATE": "2023-03-01", "TX_DESCR_FR": "Ixelles", "CASES": 7}],
        ...     liste_communes=["Bruxelles", "Ixelles"], mode_detaille=False)
        {'2023-03-01': {'Bruxelles': 1, 'Ixelles': 7}}
        # Cas vide : liste sans entrée
        >>> organiser_donnees_par_commune([], liste_communes=["Bruxelles", "Ixelles"], mode_detaille=False)
        {}

        # Cas tout invalide : aucun enregistrement correct
        >>> organiser_donnees_par_commune([
        ...     {"DATE": "2023-03-01"},  # manquant 'TX_DESCR_FR' et 'CASES'
        ...     {"foo": 42}
        ... ], liste_communes=["Bruxelles"], mode_detaille=False)
        {}

    Étapes:
        1. (Optionnel) Filtre la liste brute selon les dates demandées.
        2. Valide chaque enregistrement avec valider_enregistrement.
        3. Pour chaque ligne correcte, transforme la valeur CASES en int (anonymisation et erreurs gérées).
        4. Structure sous forme {date: {commune: cas}}.
        5. Trie les dates et logue les anomalies/statistiques.

    Tips:
        - Les logs signalent les communes manquantes ou les cas problématiques.
        - Le mode_detaille=True aide à auditer les problèmes lors de l’intégration des données.

    Utilisation:
        Fonction centrale pour préparer la donnée avant toute analyse ou visualisation.

    Limitation:
        - Ne gère que les communes spécifiées dans la liste passée en paramètre.
        - Ne traite pas les doublons de date/commune (les derniers cas remplacent les précédents).
        - Ne fait pas d’imputation temporelle ou de validation croisée.

    See also:
        - valider_enregistrement, convertir_cases_en_int
        - filtrer_liste_par_date (importée depuis utils)
    """   
    # === Filtrage par date si demandé ===
    if date_debut or date_fin:
        liste_enregistrements_bruts, stats_filtrage = filtrer_liste_par_date(
            liste_enregistrements_bruts, logger = logger,
            date_debut = date_debut, date_fin = date_fin
        )

    if mode_detaille:
        logger.info(f"Traitement des {len(liste_communes)} communes de Bruxelles...")

    donnees_organisees: dict[str, dict[str, int]] = {}
    nbr_total_entrees: int = 0
    nbr_entrees_traitees: int = 0
    communes_trouvees: set[str] = set()

    for enregistrement in liste_enregistrements_bruts:
        nbr_total_entrees += 1
        valide = valider_enregistrement(enregistrement)
        if not valide:
            continue
        date, commune, cas = valide

        if commune in liste_communes:
            nbr_entrees_traitees += 1
            communes_trouvees.add(commune)
            nbr_cas = convertir_cases_en_int(cas, commune, date)
            if date not in donnees_organisees:
                donnees_organisees[date] = {}
            donnees_organisees[date][commune] = nbr_cas

    donnees_organisees = OrderedDict(sorted(
        donnees_organisees.items(),
        key=lambda x: str_vers_datetime(x[0], logger)
    ))

    if mode_detaille:
        logger.info(f"Entrées traitées: {nbr_entrees_traitees}/{nbr_total_entrees}")
        logger.info(f"Communes trouvées: {sorted(communes_trouvees)}/{liste_communes}")
        logger.info(f"Dates uniques: {len(donnees_organisees)}")

        communes_manquantes = set(liste_communes) - communes_trouvees
        if communes_manquantes:
            logger.warning("Communes manquantes dans les données :")
            for commune in sorted(communes_manquantes):
                logger.warning(f"    - {commune}")

    return donnees_organisees


@log_debut_fin(logger)
def main() -> None:
    """
    Point d'entrée du script de nettoyage/structuration des données COVID-19.

    Cette fonction lance l’ensemble du pipeline : chargement, organisation, sauvegarde, 
    avec journalisation détaillée de toutes les étapes et erreurs.

    Args:
        Aucun (fonction sans argument).

    Returns:
        None

    Note:
        - Log toutes les étapes, succès et erreurs critiques.
        - Affiche des conseils pour la gestion de l’écrasement du fichier de sortie.
        - Peut être exécutée en script ou appelée via __main__.

    Example:
        >>> main()
        (Affiche dans le terminal les logs détaillés et l'emplacement du fichier de sortie.)

    Étapes:
        1. Charge le fichier brut (JSON) via charger_json.
        2. (Optionnel) Affiche la période couverte.
        3. Appelle organiser_donnees_par_commune pour créer le dictionnaire propre.
        4. (Optionnel) Log de la période si filtrage par date.
        5. Sauvegarde le dictionnaire propre dans le fichier cible, avec gestion de l’écrasement.
        6. Log de succès ou d’erreur selon le résultat.

    Tips:
        - Pour forcer la réécriture du fichier, modifiez “ecraser_la_sauvegarde_existante”.
        - Peut être utilisé tel quel ou adapté pour d’autres datasets régionaux.

    Utilisation:
        Fonction à appeler pour lancer tout le pipeline de prétraitement sur un jeu COVID.

    Limitation:
        - Aucun retour explicite (fonction de script), résultats accessibles par le log et le fichier.
        - Ne gère pas de rollback ou d’annulation en cas d’erreur sur la sauvegarde.

    See also:
        - organiser_donnees_par_commune
        - sauvegarder_json, charger_json (utils)
    """

    logger.info("==== DÉMARRAGE 02_data_loader ====\n")
    try:
        # --- CHARGEMENT des données brutes ---
        liste_enregistrements_bruts = charger_json(
            EMPLACEMENT_DONNEES_BRUTES_COVID, logger
        )



        # --- AFFICHAGE (optionnel) de la période couverte ---
        # afficher_periode_liste(
        #       liste_enregistrements_bruts, logger=logger, champ_date="DATE")

        # --- PARAMÈTRES de plage de dates (désactivés par défaut) ---
        # date_debut = "2023-01-01"     # Décommenter pour filtrer
        # date_fin   = "2023-12-31"     # Décommenter pour filtrer

        # --- TRAITEMENT ---
        donnees_organisees: dict[str, dict[str, int]] = organiser_donnees_par_commune(
            liste_enregistrements_bruts,
            liste_communes = COMMUNES,
            logger = logger,
            mode_detaille = MODE_DETAILLE,
            # date_debut=date_debut,
            # date_fin=date_fin

        )

        # --- LOG : Affiche la période filtrée si filtrage activé ---
        # if "date_debut" in locals() or "date_fin" in locals():
        #     logging.info(f"Plage de dates filtrée : {date_debut or '...'} à {date_fin or '...'}")

        # --- SAUVEGARDE ---
        if donnees_organisees:
            # Ne crée le dossier cible seulement s'il est précisé
            dir_cible = os.path.dirname(EMPLACEMENT_DONNEES_COVID)
            if dir_cible:
                creer_dossier_si_absent(dir_cible, logger)

            emplacement_final = sauvegarder_json(
                donnees_organisees, EMPLACEMENT_DONNEES_COVID,
                ecrasement=ecraser_la_sauvegarde_existante,
                format_date_d_enregistrement="", logger = logger
            )

            if ecraser_la_sauvegarde_existante:
                action = "a été écrasé/sauvegardé avec succès."
                consigne_action = "empêcher"
                bool_val = "False"
            else:
                action = "était déjà sauvegardé (pas d'écrasement du fichier existant)."
                consigne_action = "forcer"
                bool_val = "True"

            conseil = (
                f"Pour {consigne_action} l'écrasement automatique, "
                f"mettez à la ligne {LIGNE_PARAM_ECRASEMENT}:\n"
                f"    ecraser_la_sauvegarde_existante: bool = {bool_val}"
            )

            logger.info(f"Le fichier {emplacement_final} {action}\n{conseil}")

        else:
            logger.error("Aucune donnée valide à sauvegarder.", exc_info = True)
    except Exception as exception:
        logger.error(f"Erreur critique : {exception}", exc_info=True)
    
    logger.info("==== Fin du traitement 02_data_loader ====\n")


if __name__ == "__main__":
    main()

