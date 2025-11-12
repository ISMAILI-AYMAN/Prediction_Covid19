# -*- coding: utf-8 -*-

"""
01_data_downloader.py
---------------------
Téléchargement, filtrage et sauvegarde des données COVID-19 par commune pour la Belgique.

Ce module fournit :
    - Le téléchargement robuste (avec gestion des réessais) du jeu de données COVID-19 par commune
      via l’API OpenDataSoft officielle.
    - L’extraction des champs essentiels : DATE, TX_DESCR_FR (nom de la commune), CASES (nombre de cas).
    - La sauvegarde locale au format JSON structuré, compatible avec les scripts de prétraitement.

Objectif :
    - Automatiser et fiabiliser la récupération des données brutes COVID-19 pour la région cible.
    - Garantir une base de données propre, filtrée, prête pour l’analyse ou le nettoyage ultérieur.

Fonctionnalités principales :
    - Fonction `telecharger_donnees` : télécharge le JSON depuis l’API avec gestion des coupures réseau.
    - Fonction `filtrer_donnees` : isole les champs utiles et ignore les lignes incomplètes.
    - Fonction `telechargement_donnees_covid` : pipeline complet : télécharge, filtre, sauvegarde.
    - Logger détaillé pour le suivi de toutes les étapes et erreurs.

Pré-requis :
    - Connexion Internet stable (l’API OpenDataSoft doit être accessible).
    - Le dossier cible `Data/` doit être disponible (créé automatiquement si absent).
    - Python 3.9+, dépendances standard (`requests`, `os`, `logging`...).

Philosophie :
    - Centralise le téléchargement des données sources pour une traçabilité et une reproductibilité totales.
    - Sépare clairement les étapes réseau, filtrage et sauvegarde (bonnes pratiques pour l’automatisation).

Utilisation typique :
    >>> $ python 01_data_downloader.py
    >>> # Ou importer filtrer_donnees pour réutilisation ailleurs.

Best Practice :
    - Toujours vérifier dans les logs la réussite du téléchargement et la taille du fichier obtenu.
    - Adaptez la variable `forcer_le_telechargement` selon que vous voulez écraser les anciennes données.

Conseils :
    - Pour automatiser sur serveur ou en crontab, activez le mode proxy si besoin (`USE_PROXY = True`).
    - Gardez le script synchronisé avec la structure de l’API : adaptez la clé des champs si l’API évolue.

Limitations :
    - Le script ne gère que les données COVID-19 "cas par commune", pas d’autres indicateurs (hospitalisations…).
    - Ne vérifie pas la cohérence des données : pour la correction/structuration, voir le module de cleaning.

Maintenance :
    - Toute évolution de l’API (changement de structure, champs, pagination) doit être reflétée ici.
    - Les logs permettent d’auditer toutes les étapes en cas d’incident réseau ou de modification source.

Documentation :
    - Chaque fonction est documentée (format Google/PEP 257), avec exemples pour l’autoformation.
    - Voir l’API officielle OpenDataSoft pour le détail des champs disponibles : 
      https://public.opendatasoft.com/explore/dataset/covid-19-pandemic-belgium-cases-municipality/information/

Auteur   : Harry FRANCIS (2025)
Contact  : github.com/documentHarry
Version  : 1.0.0 (2025-07-23)
License  : Code source ouvert à la consultation.
           Toute redistribution ou usage commercial est soumis à accord écrit préalable.
"""


__version__ = "1.0.0"

# Compatible Python 3.9+  | Typage PEP 484 | Type PEP257

# --- Librairies standards
import datetime
import logging
import os
from time import sleep
from typing import TypedDict, Any, Optional, Union

# --- Librairies tiers
import requests

# --- Modules locaux
from utils_io import creer_dossier_si_absent, sauvegarder_json
from utils_log import log_debut_fin, configurer_logging
from constantes import EMPLACEMENT_DONNEES_COVID, PROXIES

# === Paramètres de configuration ===
url_api: str = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "covid-19-pandemic-belgium-cases-municipality/exports/json"
)
forcer_le_telechargement: bool = False
nbr_reessais: int = 3
pause_reessai: int = 5  # secondes
#: Niveau du logging (INFO, DEBUG, WARNING...).
NIVEAU_LOG: int = logging.INFO
logger: logging.Logger = configurer_logging(NIVEAU_LOG, "01_data_downloader")




class EnregistrementCovid(TypedDict):
    """
    Enregistrement typé pour un cas COVID-19 par commune.

    Attributs :
        DATE (str) : Date du relevé, au format 'YYYY-MM-DD'.
        TX_DESCR_FR (str) : Nom de la commune (en français).
        CASES (int | str) : Nombre de cas (int ou "<5" en str si anonymisé).
    """
    DATE: str
    TX_DESCR_FR: str
    CASES: Union[int, str] # int ou "<5"


@log_debut_fin(logger)
def telecharger_donnees(
        export_url:str, nbr_reessais: int, pause_reessai: int
        ) -> Optional[list[dict[str, Any]]]:
    """
    Télécharge les données JSON depuis l’API, avec gestion de réessais réseau et logging.

    Args:
        export_url (str): URL complète du service web/API à interroger pour récupérer le JSON.
        nbr_reessais (int): Nombre maximum de tentatives si le téléchargement échoue.
        pause_reessai (int): Pause (en secondes) entre deux essais en cas d’échec.

    Returns:
        Optional[list[dict[str, Any]]]: Liste de dictionnaires (enregistrements JSON), ou None
        si le téléchargement échoue après tous les essais.

    Raises:
        requests.exceptions.RequestException: Si une erreur réseau inattendue survient
        (exception relancée seulement si non gérée dans la boucle de réessais).

    Note:
        - Utilise un logger dédié pour tracer chaque étape et chaque tentative.
        - Si un proxy est activé (voir USE_PROXY), il est utilisé pour la connexion.
        - Timeout fixé à 180 secondes pour éviter les blocages réseau longs.
        - Abandonne définitivement après `nbr_reessais` tentatives échouées.

    Example:
        >>> donnees = telecharger_donnees(url_api, 3, 5, logger)
        >>> if donnees is None:
        ...     logger.error("Erreur de téléchargement")
        >>> else:
        ...     logger.info(f"Premier enregistrement : {donnees[0]}")

    Étapes:
        1. Effectue jusqu’à `nbr_reessais` tentatives de requête HTTP (GET) vers l’API.
        2. Si succès, convertit la réponse JSON en liste Python et la retourne.
        3. Si échec, attend `pause_reessai` secondes puis recommence.
        4. Log chaque tentative, succès, ou erreur (détail accessible dans le terminal).
        5. Retourne None si tous les essais échouent.

    Tips:
        - Adapter le nombre de tentatives/pause si la connexion réseau est lente ou instable.
        - Les erreurs du serveur distant sont également capturées.
        - Attention à la consommation API si vous augmentez trop le nombre de réessais.

    Utilisation:
        Utilisé en amont de tout pipeline d’ETL, dans un script ou un notebook de collecte.
        Intégrable dans un script automatisé (cron, pipeline) ou pour du monitoring.

    Limitation:
        - Ne gère que les requêtes HTTP GET et le format JSON natif de l’API OpenDataSoft.
        - Pas de gestion avancée d’authentification, pagination, ou quota API.
        - Si la structure du JSON de l’API change, adaptation manuelle requise.

    See also:
        - requests.get (https://docs.python-requests.org/)
        - filtrer_donnees (pour nettoyer la sortie brute)
        - le logger local pour le suivi en production
    """
    for tentative in range(1, nbr_reessais + 1):
        try:
            logger.info(f"Tentative {tentative}/{nbr_reessais} – "
                        f"Téléchargement du jeu de données complet...")
            reponse = requests.get(export_url, proxies=PROXIES, timeout=180)
            reponse.raise_for_status()
            donnees = reponse.json()
            logger.info(f"{len(donnees)} enregistrements téléchargés")
            return donnees
        except Exception as exception:
            logger.warning(
                f"Erreur lors du téléchargement (tentative "
                f"{tentative}/{nbr_reessais}) : {exception}"
            )
            if tentative < nbr_reessais:
                logger.info(f"Nouvel essai dans {pause_reessai} secondes...")
                sleep(pause_reessai)
            else:
                logger.error(f"Nombre maximal de tentatives atteint. "
                             f"Abandon du téléchargement.", exc_info = True)
                return None


@log_debut_fin(logger)
def filtrer_donnees(donnees: list[dict[str, Any]]) -> list[EnregistrementCovid]:
    """
    Filtre la liste d’enregistrements pour ne garder que DATE, TX_DESCR_FR et CASES.

    Args:
        donnees (list[dict]): Liste d’enregistrements bruts (un dictionnaire par ligne JSON).

    Returns:
        list[EnregistrementCovid]: Liste nettoyée avec seulement les champs DATE (str),
        TX_DESCR_FR (str) et CASES (int ou str), structure conforme à l’analyse en aval.

    Note:
        - Ignore toute ligne qui n’a pas de date ou de nom de commune (filtrage strict).
        - La structure retournée correspond à la classe TypedDict `EnregistrementCovid`.
        - Les erreurs de structure sont simplement ignorées (pas d’exception levée).
        - Loggue la progression et les cas particuliers.

    Example:
        >>> brutes = [{"date": "2022-01-01", "tx_descr_fr": "Bruxelles", "cases": 5}]
        >>> filtrer_donnees(brutes)
        [{'DATE': '2022-01-01', 'TX_DESCR_FR': 'Bruxelles', 'CASES': 5}]
        >>> filtrer_donnees([{"foo": 123}])
        []
        >>> filtrer_donnees([])
        []

    Étapes:
        1. Parcourt chaque ligne de la liste brute JSON.
        2. Récupère les champs "date", "tx_descr_fr", "cases" si présents.
        3. Si "DATE" et "TX_DESCR_FR" existent, ajoute à la liste résultat.
        4. Retourne la liste des enregistrements valides.

    Tips:
        - Toujours utiliser ce filtre avant sauvegarde ou analyse pour garantir la cohérence.
        - Compatible avec l’export JSON/CSV ou le passage à des outils pandas/numpy.

    Utilisation:
        À insérer juste après le téléchargement, avant toute étape de nettoyage,
        d’analyse, ou d’export.

    Limitation:
        - N’inclut pas de validation du type de CASES (peut rester une chaîne "<5").
        - Les champs additionnels (ex : code commune) sont perdus (filtrage minimal).

    See also:
        - telecharger_donnees
        - telechargement_donnees_covid
    """
    donnees_converties: list[EnregistrementCovid] = []
    for donnee in donnees:
        ligne: EnregistrementCovid = {
            "DATE": donnee.get("date"),
            "TX_DESCR_FR": donnee.get("tx_descr_fr"),
            "CASES": donnee.get("cases")
        }
        if ligne["DATE"] and ligne["TX_DESCR_FR"]:
            donnees_converties.append(ligne)
    return donnees_converties


@log_debut_fin(logger)
def telechargement_donnees_covid() -> Optional[str]:
    """
    Pipeline complet : télécharge, filtre et sauvegarde les données COVID-19 par commune.

    Returns:
        Optional[str]: Emplacement du fichier JSON créé si le pipeline réussit.
        Retourne None si une étape du pipeline échoue (téléchargement ou sauvegarde).

    Note:
        - Toutes les étapes sont loguées, chaque incident ou succès est notifié dans le terminal.
        - S’arrête dès qu’une étape critique échoue (ex : téléchargement impossible).
        - Utilise le paramètre global `forcer_le_telechargement` pour décider de retélécharger ou non.
        - Le fichier est sauvegardé dans le dossier `Data/` (créé si besoin).

    Example:
        >>> emplacement = telechargement_donnees_covid(logger)
        >>> if emplacement:
        ...     logger.info(f"Fichier sauvegardé : {emplacement}")
        >>> else:
        ...     logger.error("Erreur lors du téléchargement ou de la sauvegarde")

    Étapes:
        1. Vérifie si un fichier existe déjà ; s’il existe et `forcer_le_telechargement` est False,
           retourne l'emplacement sans retélécharger.
        2. Si téléchargement nécessaire : appelle `telecharger_donnees` pour récupérer les données brutes.
        3. Applique `filtrer_donnees` pour nettoyer les champs inutiles.
        4. Sauvegarde le résultat filtré dans un fichier JSON, log l’action.
        5. Retourne l'emplacement complet du fichier si tout a réussi, sinon None.

    Tips:
        - Pour forcer un téléchargement frais, positionner `forcer_le_telechargement = True`.
        - Peut s’intégrer dans une routine automatique (batch/cron) ou être appelé à la main.

    Utilisation:
        Utilisé comme point d’entrée du pipeline, dans `main()` ou en import dans un notebook.

    Limitation:
        - Ne gère pas le contrôle de version ou l’archivage des anciens fichiers.
        - Ne valide pas le contenu sémantique du JSON (ex : doublons, valeurs aberrantes).

    See also:
        - filtrer_donnees
        - telecharger_donnees
        - sauvegarder_json (utilitaire d’export, voir utils.py)
    """
    if not forcer_le_telechargement:
        if os.path.exists(EMPLACEMENT_DONNEES_COVID):
            logger.info(f"Le fichier {EMPLACEMENT_DONNEES_COVID} existe déjà. "
                        f"Pas de téléchargement.\n"
                        f"Pour forcer le téléchargement, modifiez la ligne 48:\n"
                        f"    forcer_le_telechargement: bool = True"
            )
            return EMPLACEMENT_DONNEES_COVID
        else:
            logger.info(f"forcer_le_telechargement est à False et "
                        f"le fichier n'existe pas. Aucun téléchargement n'est lancé.")
            return None
    
    # Ici, on force le téléchargement :
    temps_debut = datetime.datetime.now()
    donnees = telecharger_donnees(url_api, nbr_reessais, pause_reessai)

    if donnees is None:
        return None

    donnees_filtrees: list[EnregistrementCovid] = filtrer_donnees(donnees)

    try:
        dir_cible = os.path.dirname(EMPLACEMENT_DONNEES_COVID)
        if dir_cible:
             creer_dossier_si_absent(dir_cible, logger)
        emplacement_sauvegarde = sauvegarder_json(
            donnees_filtrees, EMPLACEMENT_DONNEES_COVID, ecrasement = True,
            logger = logger
        )
        logger.info(f"{len(donnees_filtrees)} enregistrements sauvegardés "
                    f"dans {emplacement_sauvegarde}")
    except Exception as exception:
        logger.error(f"Erreur lors de la sauvegarde du fichier : {exception}", 
                     exc_info = True)
        return None

    temps_fin = datetime.datetime.now()
    duree = temps_fin - temps_debut
    logger.info(f"Durée totale : {duree}")
    return emplacement_sauvegarde


@log_debut_fin(logger)
def main() -> None:
    """
    Point d’entrée du script de téléchargement/sauvegarde COVID-19.

    Lance le pipeline complet de récupération, nettoyage, et sauvegarde des données, 
    avec journalisation détaillée.

    Args:
        None

    Returns:
        None

    Note:
        - Toutes les actions et erreurs sont loguées.
        - Fournit un message clair de fin de traitement ou d’erreur.

    Example:
        >>> main()
        (Affiche les logs du pipeline, sauvegarde le fichier si tout va bien.)

    Étapes:
        1. Démarre et loggue le début du traitement.
        2. Appelle `telechargement_donnees_covid` pour lancer le pipeline principal.
        3. Loggue la réussite ou l’échec du traitement.
        4. Affiche le chemin absolu du fichier s’il a été créé.

    Tips:
        - À appeler seulement en mode script (protection if __name__ == "__main__").
        - Pour usage en notebook, utiliser plutôt les fonctions séparément.

    Utilisation:
        Script principal pour lancer toute la chaîne ETL COVID-19 (collecte brute).

    Limitation:
        - Ne retourne aucune valeur (fonction “procedurale” pure).
        - Toute erreur bloquante loggée mais non remontée à l’utilisateur final.

    See also:
        - telechargement_donnees_covid
        - le logger du module (paramétrable dans utils)
    """
    logger.info("==== DÉMARRAGE 01_data_downloader ====")
    try:
        emplacement = telechargement_donnees_covid()
        if emplacement:
            logger.info(f"Fin de traitement. Fichier obtenu : {emplacement}")
            logger.info(f"Le fichier final est disponible ici :\n"
                        f"{os.path.abspath(emplacement)}")
        else:
            logger.warning("Aucun fichier n'a été téléchargé/sauvegardé.")
    except Exception as exception:
        logger.critical(f"Erreur inattendue : {exception}", exc_info=True)
    logger.info("==== Fin du traitement 01_data_downloader ====")


if __name__ == "__main__":
    main()
