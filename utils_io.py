# -*- coding: utf-8 -*-

"""
utils_io.py
-----------
Module d’outils pour la gestion des fichiers et des entrées/sorties dans le projet.  
Conçu pour faciliter le traitement, la sauvegarde, et la récupération des données dans des formats simples.

Ce module fournit...
    - Des décorateurs pour la gestion centralisée des erreurs lors de la lecture/écriture ou du téléchargement.
    - Des fonctions pour charger et sauvegarder des fichiers JSON en toute sécurité.
    - Des utilitaires pour créer des dossiers et manipuler les chemins de fichiers de façon portable.

Objectif :
    - Mettre à disposition des fonctions/classes/outils réutilisables pour la gestion robuste des fichiers,
      des dossiers, et du réseau.
    - Permettre une centralisation, une maintenance aisée et une utilisation pédagogique de toutes les
      opérations d’entrées/sorties du projet.

Fonctionnalités principales :
    - log_io_exceptions : décorateur pour journaliser et relancer les erreurs disque.
    - log_http_exceptions : décorateur pour journaliser les erreurs réseau et HTTP.
    - charger_json, sauvegarder_json : lecture/écriture sécurisées des fichiers JSON.
    - creer_dossier_si_absent : création de dossiers parents si besoin.
    - telecharger_json_depuis_url : récupération de données JSON distantes avec gestion d’erreurs.
    - obtenir_emplacement_donnees : obtention fiable du chemin d’un fichier de données local.

Prérequis :
    - Données au format JSON (pour lecture/écriture).
    - Un logger Python initialisé (pour journaliser erreurs et succès).
    - Bibliothèques externes : requests (pour les fonctions réseau).

Philosophie :
    - Centralise le paramétrage et les opérations d’E/S pour faciliter la maintenance.
    - Garantit des logs cohérents pour tout traitement, utile pour le debug et l’audit.
    - Facilite la modification et garantit la cohérence des traitements sur fichiers et dossiers.

Utilisation typique :
    >>> from utils_io import charger_json, sauvegarder_json, telecharger_json_depuis_url
    >>> donnees = charger_json("data/mon_fichier.json", logger)
    >>> sauvegarder_json(donnees, "data/export.json", logger)
    >>> nouvelle_donnees = telecharger_json_depuis_url("https://...", logger)

Best Practice :
    - Toujours utiliser un logger explicite pour tracer les erreurs et les étapes importantes.
    - Appeler creer_dossier_si_absent avant toute écriture de fichier dans un nouveau dossier.

Conseils :
    - Pour personnaliser les formats de dates ou l’emplacement des dossiers, modifier les constantes dans le module.
    - Intégrer ces fonctions dans un script principal pour automatiser les traitements.

Limitations :
    - Ce module ne gère pas l’export CSV, Excel, ou les fichiers binaires.
    - Ne vérifie pas la compatibilité totale des objets passés à json.dump : prévoir une validation au besoin.
    - Ne prend pas en charge la gestion des bases de données ou la connexion à des APIs complexes.

Maintenance :
    - Toute évolution de la logique d’E/S doit se faire dans ce fichier pour garder le projet cohérent.
    - Les logs produits par chaque fonction facilitent le débogage et la traçabilité des traitements.

Documentation :
    - Les fonctions et classes sont documentées selon le format Google/PEP 257, avec exemples concrets.
    - Pour en savoir plus sur la gestion des fichiers ou sur requests : https://docs.python.org/3/library/json.html
      et https://docs.python-requests.org/

Auteur   : Harry FRANCIS (2025)
Contact  : github.com/documentHarry
Version  : 1.0.0 (2025-07-23)
License  : Code source ouvert à la consultation.
           Toute redistribution ou usage commercial est soumis à accord écrit préalable.
"""

__version__ = "1.0.0"

# Compatible Python 3.10+ | Typage PEP 484 | PEP 257 | PEP 8 (88 caractères max)

# --- Librairies standards
# 'functools' fournit des outils avancés pour manipuler ou décorer des fonctions (ex: @wraps, partial).
import functools
# 'json' permet de lire et écrire des données au format JSON (échange de données, configuration, etc.).
import json
# 'logging' gère la journalisation : création de logs pour enregistrer erreurs, infos, ou déboguer un programme.
import logging
# 'os' offre des fonctions pour interagir avec le système d’exploitation 
# (fichiers, dossiers, chemins, variables d'env.).
import os
# Importe 'datetime' du module 'datetime' pour manipuler facilement les dates et heures (ex : obtenir la date
# courante, formater des timestamps, calculer des différences entre dates).
from datetime import datetime
# 'typing' : outils pour typer les fonctions/classes :
from typing import Any

# --- Librairies tiers
# Importe la bibliothèque 'requests', utilisée pour envoyer des requêtes HTTP, récupérer des données
# depuis des APIs web et intercepter les exceptions spécifiques aux erreurs réseau (timeout, proxy, etc.).
import joblib
import requests

# --- Modules locaux
# Importe la constante EMPLACEMENT_DONNEES : elle représente le chemin absolu du dossier 'Data' du projet,
# utilisé pour lire et enregistrer tous les fichiers de données de l’application de façon centralisée.
from constantes import EMPLACEMENT_DONNEES, EMPLACEMENT_MODELE_FINAL


def log_http_exceptions(fonction):
    """
    Décorateur qui capture, loggue et relance les exceptions HTTP ou JSON,
    pour toute fonction de téléchargement ou d’appel réseau utilisant requests.

    Args:
        fonction (callable): Fonction cible à décorer. Doit accepter un logger
            comme paramètre, en positionnel ou nommé.

    Returns:
        callable: Fonction décorée. Relance toutes les exceptions après les avoir
            logguées avec le logger fourni.

    Raises:
        requests.exceptions.Timeout: Temps d’attente dépassé lors de la requête réseau.
        requests.exceptions.ConnectionError: Erreur de connexion réseau (hôte
            injoignable, DNS, etc.).
        requests.exceptions.ProxyError: Erreur liée à la configuration du proxy.
        requests.exceptions.TooManyRedirects: Trop de redirections HTTP.
        requests.exceptions.HTTPError: Réponse HTTP avec code d’erreur (4xx, 5xx...).
        requests.exceptions.RequestException: Erreur générique requests non
            listée plus haut (ex : chunked encoding, etc.).
        json.JSONDecodeError: Réponse reçue non décodable en JSON valide.
        Exception: Toute autre erreur inattendue (réseau, parsing ou autre).

    Note:
        - Le logger doit être passé à la fonction décorée (position ou nom).
        - Toutes les erreurs sont systématiquement logguées avant d’être relancées.
        - N’altère pas la signature ni les variables de la fonction décorée.
        - Compatible Python 3.10+ et requests 2.x.

    Example:
        >>> @log_http_exceptions
        ... def telecharger_json(url, logger):
        ...     r = requests.get(url, timeout=5)
        ...     r.raise_for_status()
        ...     return r.json()
        >>> telecharger_json("http://exemple.com/api", logger)
        # Loggue l’erreur en cas de timeout, HTTPError ou JSONDecodeError.

    Étapes:
        1. Recherche un logger dans les paramètres.
        2. Exécute la fonction cible normalement.
        3. Si exception requests ou JSON, loggue un message explicite.
        4. Relance l’exception pour gestion externe.

    Tips:
        - À utiliser pour toute fonction qui appelle une API ou télécharge
          via requests.
        - Facilite le debug réseau grâce aux logs complets, même en production.
        - Prévoyez un logger bien configuré avant l’appel.

    Utilisation:
        À appliquer sur les fonctions qui font du réseau (HTTP, API, téléchargement)
        pour obtenir un historique d’erreurs dans les logs du projet.

    Limitation:
        - Ne gère pas les erreurs d’écriture disque après téléchargement.
        - Le logger doit être passé et doit être du type logging.Logger.

    See also:
        - requests, requests.exceptions, logging.Logger
        - log_io_exceptions (pour les erreurs disque)
    """
    @functools.wraps(fonction)
    def fonction_decoree(*args, **kwargs):
        # Recherche du logger dans les kwargs ou args
        logger = kwargs.get("logger", None)
        if logger is None:
            for arg in reversed(args):
                if isinstance(arg, logging.Logger):
                    logger = arg
                    break
        try:
            return fonction(*args, **kwargs)
        except requests.exceptions.Timeout as timeout_erreur:
            logger.error(f"Timeout réseau : {timeout_erreur}", exc_info = True)
            raise
        except requests.exceptions.ConnectionError as connexion_erreur:
            logger.error(f"Erreur de connexion réseau : {connexion_erreur}", 
                         exc_info = True)
            raise
        except requests.exceptions.ProxyError as proxy_erreur:
            logger.error(f"Erreur de proxy : {proxy_erreur}", exc_info = True)
            raise
        except requests.exceptions.TooManyRedirects as redirection_erreur:
            logger.error(f"Trop de redirections : {redirection_erreur}", 
                         exc_info = True)
            raise
        except requests.exceptions.HTTPError as http_erreur:
            logger.error(f"Erreur HTTP : {http_erreur}", exc_info = True)
            raise
        except requests.exceptions.RequestException as requete_erreur:
            logger.error(f"Erreur générale requests : {requete_erreur}", 
                         exc_info = True)
            raise
        except json.JSONDecodeError as json_erreur:
            logger.error(f"Erreur de décodage JSON : {json_erreur}", exc_info = True)
            raise
        except Exception as exception:
            logger.error(f"Erreur inattendue (réseau ou parsing) : {exception}", 
                         exc_info = True)
            raise
    return fonction_decoree


@log_http_exceptions
def telecharger_json_depuis_url(
        url: str, logger: logging.Logger, nbr_reessais: int = 3,
        pause_reessai: int = 5, timeout: int = 60, proxies: dict | None = None
        ) -> dict[str, Any] | None:
    """
    Télécharge un fichier JSON (ou GeoJSON) depuis une URL, avec gestion des erreurs,
    des délais d’attente et des logs.

    Args:
        url (str): Adresse internet du fichier à télécharger (ex: "https://...").
        logger (logging.Logger): Logger utilisé pour afficher les étapes, erreurs,
            et infos du téléchargement.
        nbr_reessais (int): Nombre de tentatives en cas d’échec (défaut 3).
        pause_reessai (int): Pause en secondes entre chaque tentative (défaut 5).
        timeout (int): Durée maximale d’attente de réponse HTTP (secondes, défaut 60).
        proxies (dict | None): Dictionnaire de proxy HTTP(s) si besoin, ou None.

    Returns:
        dict[str, Any] | None: Le contenu JSON téléchargé sous forme de dictionnaire,
            ou None si le téléchargement échoue.

    Note:
        - Toutes les étapes (début, succès, erreur) sont loguées avec le logger fourni.
        - Si toutes les tentatives échouent, la fonction retourne None.
        - La fonction ne bloque jamais définitivement le script (maximum = nbr_reessais).
        - Fonction compatible Python 3.10+ (usage | None et dict[...]).

    Example:
        >>> logger = logging.getLogger("test")
        >>> data = telecharger_json_depuis_url(
        ...     "https://data.geojson", logger, nbr_reessais=2)
        >>> print(type(data))
        <class 'dict'>  # ou None si échec

    Étapes:
        1. Effectue jusqu'à nbr_reessais téléchargements de l’URL donnée.
        2. Après chaque échec, affiche un message et attend pause_reessai secondes.
        3. Si succès, retourne le JSON décodé ; sinon, retourne None.

    Tips:
        - Utilise un logger configuré pour voir les messages détaillés.
        - Pour les jeux de données publics, proxies peut rester à None.

    Utilisation:
        Pour automatiser le chargement de données à jour sur internet
        dans un pipeline data, ou valider la disponibilité d’une source.

    Limitation:
        - Ne gère que le format JSON/GeoJSON (pas CSV/XML).
        - Retourne None silencieusement si toutes les tentatives échouent.

    See also:
        - requests.get, logger, charger_json
    """
    for tentative in range(1, nbr_reessais + 1):        
        logger.info(f"Tentative {tentative}/{nbr_reessais} – "
                    f"Téléchargement : {url}")
        reponse = requests.get(url, proxies=proxies, timeout=timeout)
        reponse.raise_for_status()
        donnees = reponse.json()   # Sera capturé si JSON mal formé
        logger.info(f"{len(donnees) if hasattr(donnees, '__len__') else '...'} "
                    f"éléments téléchargés.")
        return donnees
        # Gestion de la pause hors try/except pour laisser remonter l'erreur

        # Si on veut continuer les tentatives sur les erreurs, 
        # il faut mettre tout sauf la pause dans le try.
    # Si toutes les tentatives échouent, raise ou return None selon politique
    logger.error("Nombre maximal de tentatives atteint. Abandon du téléchargement.")
    return None


def log_io_exceptions(fonction):
    """
    Décorateur pour fonction. Capture et loggue les exceptions liées aux entrées/sorties
    (I/O), puis relance chaque erreur après l’avoir enregistrée avec le logger fourni.

    Args:
        fonction (callable): Fonction cible à décorer. Doit accepter un argument
            logger ou avoir un logger passé parmi ses paramètres.

    Returns:
        callable: Une nouvelle fonction décorée, qui loggue et relance les erreurs
            I/O courantes, sans modifier la signature.

    Raises:
        FileNotFoundError: Si un fichier à lire/écrire est introuvable.
        PermissionError: Si les droits d’accès au fichier/dossier sont insuffisants.
        IsADirectoryError: Un dossier est fourni à la place d’un fichier.
        NotADirectoryError: Une partie du chemin attendue comme dossier est un fichier.
        FileExistsError: Le fichier existe déjà lors d’une opération exclusive.
        OSError: Problème système (ex : disque plein, erreur matérielle, chemin non
            valide).
        UnicodeDecodeError: Problème d’encodage ou de décodage (UTF-8, etc.).
        ValueError: Erreur de valeur inattendue dans le traitement du fichier.
        Exception: Toute autre erreur inattendue survenue pendant l’exécution.

    Note:
        - Le logger doit être passé à la fonction décorée, soit en argument nommé,
          soit comme dernier paramètre.
        - Les exceptions sont toujours relancées après avoir été logguées.
        - Chaque type d’erreur reçoit un message de log explicite (voir code source).
        - Aucun attribut de la fonction n’est modifié.
        - Compatible Python 3.10+.

    Example:
        >>> @log_io_exceptions
        ... def ouvrir_fichier(chemin, logger):
        ...     with open(chemin) as f:
        ...         return f.read()
        >>> ouvrir_fichier("inexistant.txt", logger)
        # Log d’erreur : "Fichier introuvable : inexistant.txt"
        # Lève FileNotFoundError

    Étapes:
        1. Recherche un logger dans les arguments.
        2. Exécute la fonction décorée normalement.
        3. Si exception, loggue le message détaillé via logger.
        4. Relance systématiquement l’exception d’origine.

    Tips:
        - Utiliser ce décorateur pour toutes les fonctions de lecture/écriture
          de fichiers ou dossiers.
        - Le logger doit être correctement initialisé avant l’appel.
        - Permet de centraliser la gestion des erreurs I/O pour le debug.

    Utilisation:
        Décorez vos fonctions manipulant le système de fichiers pour avoir des logs
        clairs en cas de bug, sans modifier leur interface.

    Limitation:
        - Ne capture pas les erreurs si aucun logger n’est passé à la fonction
          décorée.
        - Le logger doit être du type logging.Logger (standard Python).

    See also:
        - logging.Logger (standard Python)
        - os, open, shutil, pathlib
    """

    @functools.wraps(fonction)
    def fonction_decoree(*args, **kwargs):
        # Recherche du logger dans les arguments positionnels ou nommés
        logger = None
        # Cherche dans les kwargs d'abord
        if "logger" in kwargs:
            logger = kwargs["logger"]
        # Sinon cherche dans les args par convention (logger dernier ou avant dernier argument)
        else:
            for arg in reversed(args):
                if isinstance(arg, logging.Logger):
                    logger = arg
                    break
        try:
            return fonction(*args, **kwargs)
        except FileNotFoundError as fichier_non_trouve:
            logger.error(
                f"Fichier introuvable : {fichier_non_trouve}", exc_info = True)
            raise
        except PermissionError as permission_erreur:
            logger.error(
                f"Droits insuffisants : {permission_erreur}", exc_info = True)
            raise
        except IsADirectoryError as repertoire_erreur:
            logger.error(
                f"Dossier fourni à la place d’un fichier : "
                f"{repertoire_erreur}", exc_info = True)
            raise
        except NotADirectoryError as n_est_pas_un_repertoire:
            logger.error(
                f"Une partie du chemin est un fichier, pas un dossier : "
                f"{n_est_pas_un_repertoire}", exc_info = True)
            raise
        except FileExistsError as fichier_existant:
            logger.error(
                f"Conflit fichier existant : {fichier_existant}", exc_info = True)
            raise
        except OSError as os_erreur:
            if "No space left" in str(os_erreur):
                logger.error(
                    f"Espace disque insuffisant : {os_erreur}", exc_info = True)
            elif "Input/output error" in str(os_erreur):
                logger.error(
                    f"Erreur matérielle d'entrée/sortie : {os_erreur}",
                    exc_info = True)
            else:
                logger.error(f"Erreur système de fichiers : {os_erreur}",
                    exc_info = True)
            raise
        except UnicodeDecodeError as encodage_erreur:
            logger.error(
                f"Erreur d'encodage/UTF-8 : {encodage_erreur}", exc_info = True)
            raise
        except ValueError as valeur_erreur:
            logger.error(f"Valeur inattendue : {valeur_erreur}", exc_info = True)
            raise
        except Exception as exception:
            logger.error(f"Erreur inattendue : {exception}", exc_info = True)
            raise
    return fonction_decoree


@log_io_exceptions
def charger_json(
    emplacement: str, logger: logging.Logger, cache: dict[str, Any] = None) -> Any:
    """
    Charge un fichier JSON local et retourne son contenu décodé, en journalisant chaque
    étape avec le logger fourni.

    Args:
        emplacement (str): Chemin du fichier JSON à lire (relatif ou absolu).
        logger (logging.Logger): Objet logger pour journaliser informations et erreurs.

    Returns:
        Any: Structure Python extraite du JSON (dict, list…), selon le contenu lu.

    Raises:
        FileNotFoundError: Fichier inexistant à l’emplacement fourni.
        PermissionError: Accès refusé au fichier (droits insuffisants).
        IsADirectoryError: Un dossier a été donné au lieu d’un fichier.
        NotADirectoryError: Un élément du chemin attendu comme dossier est un fichier.
        FileExistsError: Conflit de fichier déjà existant lors d’une opération annexe.
        OSError: Problème système (disque plein, erreur matérielle, etc.).
        UnicodeDecodeError: Fichier non décodable en UTF-8.
        json.JSONDecodeError: Contenu du fichier non conforme au format JSON.
        ValueError: Autre erreur de valeur liée à la lecture/décodage.
        Exception: Toute autre erreur inattendue pendant la lecture ou le décodage.

    Note:
        - Toutes les erreurs d’E/S et de décodage sont journalisées par le logger.
        - Le retour dépend du contenu JSON : peut être un dict, une list, etc.
        - Fonction compatible Python 3.10+.
        - Utilise le décorateur log_io_exceptions pour la gestion centralisée des erreurs.

    Example:
        >>> logger = logging.getLogger("test")
        >>> resultat = charger_json("exemple.json", logger)
        >>> type(resultat)
        <class 'dict'>  # ou <class 'list'>, selon le fichier

    Étapes:
        1. Ouvre le fichier en lecture UTF-8.
        2. Loggue le début du chargement.
        3. Décode le JSON en objet Python.
        4. Retourne l’objet obtenu ou lève une exception si problème.

    Tips:
        - Idéal pour charger configs, jeux de données, résultats exportés.
        - Logger conseillé pour diagnostiquer facilement les erreurs disque ou format.

    Utilisation:
        Utilisable dans tout pipeline ou script lisant des données JSON locales.

    Limitation:
        - Ne gère que le format JSON.
        - Exceptions levées si fichier corrompu, accès refusé ou format illisible.

    See also:
        - sauvegarder_json : pour l’écriture.
        - telecharger_json_depuis_url : chargement distant.
        - json.load : standard lib utilisée.
    """
    if cache is not None and emplacement in cache:
        if logger: logger.debug(f"Lecture JSON depuis cache : {emplacement}")
        return cache[emplacement]
    with open(emplacement, "r", encoding="utf8") as fichier:
        logger.info(f"Chargement du fichier JSON : {emplacement}")
        return json.load(fichier)


@log_io_exceptions
def sauvegarder_json(
        donnees: Any, emplacement: str, logger: logging.Logger,
        ecrasement: bool = False,
        format_date_d_enregistrement: str = "%Y-%m-%d_%Hh%Mm%Ss", indent: int = 2,
        ensure_ascii: bool = False, ) -> str:
    """
    Sauvegarde un objet Python (dict, list, etc.) au format JSON sur disque, avec 
    gestion robuste des erreurs, journalisation détaillée et création automatique 
    du dossier cible si besoin.

    Args:
        donnees (Any): Données à sauvegarder (dict, list ou tout objet JSON-sérialisable).
        emplacement (str): Chemin cible du fichier à créer ou remplacer.
        logger (logging.Logger): Logger utilisé pour journaliser toutes les étapes.
        ecrasement (bool): Si True, écrase le fichier existant (défaut False).
        format_date_d_enregistrement (str): Format du timestamp ajouté en cas de 
            sauvegarde sans écrasement (ex: '%Y-%m-%d_%Hh%Mm%Ss').
        indent (int): Niveau d’indentation du fichier JSON (défaut 2 pour la lisibilité).
        ensure_ascii (bool): Si True, encode le JSON en ASCII pur (défaut False, UTF-8).

    Returns:
        str: Chemin du fichier effectivement sauvegardé (avec suffixe si besoin).

    Raises:
        PermissionError: L’utilisateur n’a pas les droits d’écriture sur le dossier/fichier.
        FileNotFoundError: Dossier parent absent ou supprimé avant écriture.
        FileExistsError: Un fichier bloque la création du nouveau fichier à cet endroit.
        IsADirectoryError: Le chemin cible pointe sur un dossier et non sur un fichier.
        NotADirectoryError: Une partie du chemin attendu comme dossier est un fichier.
        ValueError: Chemin ou nom de fichier invalide (caractères interdits, path trop long).
        TypeError: Données non sérialisables en JSON (ex: objet complexe sans conversion).
        MemoryError: Problème mémoire lors de la sérialisation (cas extrême).
        UnicodeEncodeError: Erreur d’encodage sur un caractère non UTF-8.
        OSError: Autre problème lié au disque ou au système de fichiers (ex: espace plein).
        Exception: Toute erreur imprévue lors de la sauvegarde.

    Note:
        - Le dossier cible est créé automatiquement s’il n’existe pas.
        - Si le fichier existe déjà et que l’écrasement est False, un suffixe daté est ajouté.
        - Chaque erreur et chaque étape (succès ou échec) est logguée.
        - Compatible Python 3.10+ (utilisation des types modernes et match d’exceptions).

    Example:
        >>> sauvegarder_json({'a': 1}, 'data/result.json', logger)
        'data/result.json'  # ou 'data/result_2025-07-23_15h23m11s.json' si déjà existant

    Étapes:
        1. Vérifie et crée le dossier parent du fichier si besoin.
        2. Détermine le nom de fichier final en fonction du mode écrasement.
        3. Ouvre le fichier en écriture texte (UTF-8).
        4. Sérialise et écrit les données avec json.dump.
        5. Log le succès ou les détails de toute erreur rencontrée.

    Tips:
        - Utiliser indent=2 pour obtenir un fichier facilement lisible à la main.
        - Penser à désactiver ensure_ascii pour garder les accents et caractères spéciaux.
        - Surveiller les logs pour diagnostiquer rapidement un problème d’accès disque.

    Utilisation:
        Idéal pour sauvegarder des résultats d’analyses, des logs structurés, 
        ou exporter des objets complexes (dictionnaires de données, listes de résultats, etc.).

    Limitation:
        - N’enregistre pas les fichiers CSV, binaires ou autres formats non JSON.
        - Ne valide pas la structure des données avant la sérialisation : si un objet 
          n’est pas JSON-compatible, une exception sera levée.

    See also:
        - charger_json (pour la lecture JSON sécurisée)
        - creer_dossier_si_absent (pour la gestion des dossiers)
        - json.dump (fonction native de sérialisation)
        - os.path.splitext, os.makedirs (pour la gestion des chemins)
    """
    emplacement_final = emplacement
    dossier = os.path.dirname(emplacement_final)
    creer_dossier_si_absent(dossier, logger)
    if os.path.exists(emplacement) and not ecrasement:
        base, ext = os.path.splitext(emplacement)
        if format_date_d_enregistrement:
            suffixe = "_" + datetime.now().strftime(format_date_d_enregistrement)
        else:
            suffixe = ""
        emplacement_final = f"{base}{suffixe}{ext}"
        logger.warning(f"Fichier existant, sauvegarde sous {emplacement_final}")
    with open(emplacement_final, "w", encoding="utf8") as fichier:
        json.dump(donnees, fichier, indent=indent, ensure_ascii=ensure_ascii)
    logger.info(f"JSON sauvegardé : {emplacement_final}")
    return emplacement_final


@log_io_exceptions
def creer_dossier_si_absent(
        dossier: str, logger: logging.Logger) -> None:
    """
    Crée le dossier (et ses parents) spécifié si absent, journalise toutes les étapes.

    Args:
        dossier (str): Chemin du dossier à créer (ou None/"" pour ignorer).
        logger (logging.Logger): Logger pour journaliser succès ou erreurs.

    Returns:
        None

    Raises:
        PermissionError: Si l’utilisateur n’a pas les droits pour créer ce dossier.
        FileNotFoundError: Si une partie du chemin n’existe pas (dossier parent
            supprimé pendant l’exécution, etc.).
        NotADirectoryError: Si un composant du chemin censé être un dossier est
            en réalité un fichier.
        IsADirectoryError: Si un fichier existe à l’emplacement où un dossier
            doit être créé.
        FileExistsError: Si un fichier existe déjà à l’endroit d’un des dossiers
            à créer.
        ValueError: Si le nom du dossier contient des caractères non valides ou
            est trop long pour le système.
        OSError: Pour tout autre problème système (disque plein, chemin trop long,
            erreur matérielle d'entrée/sortie, encodage du FS…).
        Exception: Pour toute erreur inattendue non listée ci-dessus.

    Note:
        - La fonction **ne fait rien** si le chemin est vide ou déjà existant.
        - Journalise chaque action et toutes les erreurs détectées.
        - Couvre tous les cas d’erreurs courants connus sous Windows/Linux/Mac.
        - Compatible Python 3.10+.

    Example:
        >>> creer_dossier_si_absent("Data/results", logger)
        # Log: "Dossier créé : Data/results" ou rien si déjà existant

    Étapes:
        1. Vérifie l’existence du dossier.
        2. Si absent, tente de le créer (avec tous ses parents).
        3. Loggue le succès ou, en cas d’échec, la nature exacte de l’erreur
           (droits, espace disque, conflit avec fichier, etc.).
        4. Relance systématiquement l’exception pour gestion externe.

    Tips:
        - Appeler cette fonction avant toute sauvegarde (JSON, CSV, images, logs).
        - Utile pour s’assurer de la robustesse des scripts batch ou automatisés.
        - Consultez les logs pour tout problème inattendu : l’erreur y sera
          explicitement tracée.

    Utilisation:
        À placer systématiquement avant toute ouverture/écriture de fichier
        dans un chemin qui pourrait ne pas exister.

    Limitation:
        - Ne “force” jamais la création en supprimant un éventuel fichier
          bloquant. Une intervention humaine reste nécessaire dans ce cas.
        - La gestion d’erreurs dépend du comportement de l’OS : certains
          systèmes de fichiers exotiques peuvent retourner d’autres codes d’erreur.
        - Peut lever une exception si le système bloque la création du dossier
          pour toute raison non prévue (voir logs pour le détail).

    See also:
        - os.makedirs (fonction système utilisée pour la création)
        - sauvegarder_json, open, os.path.dirname
    """
    if dossier and not os.path.exists(dossier):
        os.makedirs(dossier, exist_ok = True)
        logger.info(f"Dossier créé : {dossier}")


@log_io_exceptions
def obtenir_emplacement_donnees(
        nom_fichier: str, logger: logging.Logger) -> str:
    """
    Retourne le chemin absolu vers un fichier du dossier de données du projet.

    Args:
        nom_fichier (str): Nom ou sous-dossier/nom du fichier à retourner.
        logger (logging.Logger | None): Logger pour signaler anomalies.

    Returns:
        str: Chemin absolu calculé à partir du dossier racine de données.

    Raises:
        TypeError: Si nom_fichier n’est pas une chaîne de caractères.
        ValueError: Si le nom de fichier est vide ou None.
        OSError: Problème système lors de la manipulation du chemin (ex : trop long).
        PermissionError: Accès refusé à une composante du chemin (rare).
        NotADirectoryError: Composant du chemin censé être un dossier, mais qui ne l’est pas.
        IsADirectoryError: Fichier trouvé là où un dossier est attendu.
        Exception: Pour toute erreur inattendue capturée par le décorateur.

    Note:
        - Si nom_fichier est déjà absolu, il est retourné sans modification.
        - Sinon, il est concaténé à EMPLACEMENT_DONNEES.
        - Compatible Python 3.10+.

    Example:
        >>> obtenir_emplacement_donnees("data.json", logger)
        '.../dossier_data/data.json'

    Étapes:
        1. Vérifie le type et le contenu du nom de fichier.
        2. Si absolu, retourne directement.
        3. Sinon, combine avec EMPLACEMENT_DONNEES.
        4. Normalise le chemin.

    Tips:
        - Toujours utiliser ce helper pour rester indépendant de la structure
          disque du projet.

    Utilisation:
        Appeler avant toute lecture/écriture sur les données du projet.


    Limitation:
        - Ne vérifie pas l’existence effective du fichier sur le disque.
        - Exceptions levées en cas de problème système lors du traitement du chemin.

    See also:
        - os.path.join, EMPLACEMENT_DONNEES
    """
    if not isinstance(nom_fichier, str):
        logger.error("nom_fichier doit être une chaîne.", exc_info = True)
        raise TypeError("nom_fichier doit être une chaîne.")
    if not nom_fichier.strip():
        logger.error("nom_fichier est vide.", exc_info = True)
        raise ValueError("nom_fichier ne doit pas être vide.")

    if os.path.isabs(nom_fichier):
        logger.warning(
            f"nom_fichier est un chemin absolu : "
            f"{nom_fichier} (EMPLACEMENT_DONNEES ignoré)")
        return nom_fichier
    chemin = os.path.join(EMPLACEMENT_DONNEES, nom_fichier)
    return os.path.normpath(chemin)


@log_io_exceptions
def sauvegarder_lignes_texte(
        lignes: list[str], emplacement: str, logger: logging.Logger) -> None:
    """
    Sauvegarde une liste de lignes de texte dans un fichier, journalise chaque étape.

    Args:
        lignes (list[str]): Texte à écrire, chaque chaîne devient une ligne.
        emplacement (str): Chemin du fichier à écrire (remplace ou crée).
        logger (logging.Logger): Logger pour journaliser succès/erreurs.

    Returns:
        None

    Raises:
        PermissionError: Si les droits d’écriture sur le dossier ou fichier sont insuffisants.
        OSError: Si un problème disque survient (espace plein, chemin non valide, etc).
        Exception: Pour toute autre erreur inattendue lors de l’écriture.

    Note:
        - Le fichier est créé si besoin, ou écrasé si déjà existant.
        - Le logger indique le nombre de lignes et le chemin final.
        - Vérifie et crée le dossier parent si nécessaire avant d’écrire.
        - Loggue tout problème rencontré.
        - Compatible Python 3.10+.

    Example:
        >>> sauvegarder_lignes_texte(["ligne1", "ligne2"], "log.txt", logger)
        # Log: "2 lignes sauvegardées dans export.txt"

        
    Étapes:
        1. Vérifie et crée le dossier parent du fichier, si absent.
        2. Ouvre le fichier en mode écriture ("w", UTF-8).
        3. Écrit chaque chaîne de la liste `lignes` dans le fichier.
        4. Log le succès avec le nombre de lignes, ou log l’erreur rencontrée.

    Tips:
        - Idéal pour sauvegarder des rapports simples, des journaux d’analyse ou des exports bruts.
        - Utiliser des chaînes déjà formatées, car chaque élément sera écrit tel quel.
        - En cas de problème disque ou de droits, consultez les logs pour le diagnostic précis.

    Utilisation:
        Recommandé pour automatiser l’export de résultats, générer des fichiers de log
        personnalisés, ou produire des fichiers texte pour la relecture humaine.

    Limitation:
        - Le mode "append" (ajout à la fin) n’est pas disponible : le fichier est toujours écrasé.
        - Peut lever une erreur si l’espace disque est saturé ou si l’utilisateur n’a pas les droits.
        - Ne fait pas de validation sur les contenus des chaînes : elles sont écrites telles quelles.

    See also:
       - creer_dossier_si_absent (pour la gestion des dossiers)
        - open (fonction standard d’ouverture de fichiers)
        - os.path.dirname (pour manipuler les chemins de fichiers)
    """
    if not isinstance(lignes, list) or not all(isinstance(l, str) for l in lignes):
        if logger:
            logger.error("lignes doit être une liste de chaînes de caractères.")
        raise TypeError("lignes doit être une liste de chaînes de caractères.")
    creer_dossier_si_absent(os.path.dirname(emplacement), logger)
    with open(emplacement, "w", encoding="utf-8") as f:
        f.writelines(lignes)
    if logger:
        logger.info(f"{len(lignes)} lignes sauvegardées dans {emplacement}")


@log_io_exceptions
def fichiers_identiques(emplacement_fichier_1, emplacement_fichier_2):
    """
    Compare le contenu binaire de deux fichiers pour vérifier s'ils sont identiques.

    Args:
        emplacement_fichier_1 (str): Chemin du premier fichier à comparer.
        emplacement_fichier_2 (str): Chemin du second fichier à comparer.

    Returns:
        bool: True si les fichiers sont exactement identiques (octet par octet),
            False sinon.

    Note:
        - Lit les fichiers par blocs de 8192 octets pour efficacité mémoire.
        - Le test s’arrête dès la première différence détectée.

    Example:
        >>> fichiers_identiques("a.bin", "b.bin")
        True
        >>> fichiers_identiques("file1.txt", "file2.txt")
        False

    Étapes:
        1. Ouvre les deux fichiers en mode binaire ('rb').
        2. Lit les fichiers par blocs et compare chaque bloc.
        3. Retourne False dès qu'une différence est trouvée.
        4. Retourne True si la fin des deux fichiers est atteinte sans écart.

    Tips:
        - Idéal pour vérifier l'intégrité après transfert ou sauvegarde.
        - Utilise peu de mémoire, convient aux fichiers volumineux.

    Utilisation:
        Utile avant d'écraser, synchroniser ou sauvegarder un fichier pour
        éviter les doublons.

    Limitation:
        - Si les fichiers sont volumineux et sur un support lent, la comparaison
          peut prendre du temps.
        - Exceptions Python levées si un chemin n’existe pas ou accès refusé.

    See also:
        - hashlib.md5 ou sha256 pour une alternative basée sur le hash.
        - os.path.exists pour vérifier l'existence avant appel.
    """
    with open(emplacement_fichier_1, 'rb') as file1:
        with open(emplacement_fichier_2, 'rb') as file2:
            while True:
                byte_1 = file1.read(8192)
                byte2 = file2.read(8192)
                if byte_1 != byte2:
                    return False
                if not byte_1:
                    return True
'''
print(fichiers_identiques(
        EMPLACEMENT_MODELE_FINAL,
        r"c:\\Users\\harry\\Backup\\Math\\Math\\final_gb_model.joblib"))
'''
