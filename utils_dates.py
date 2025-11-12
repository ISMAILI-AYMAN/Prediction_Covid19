# -*- coding: utf-8 -*-

"""
utils_dates.py
--------------
Outils robustes pour l’analyse, le filtrage et la validation de listes de
dictionnaires datés (ex : cas COVID-19), avec gestion détaillée des erreurs
et reporting standardisé dans les logs.

Ce module fournit :
    - Des structures typées (TypedDict, NamedTuple) pour la manipulation de données COVID.
    - Des fonctions utilitaires pour parser, filtrer et résumer des listes de dicts datés.
    - Une gestion complète du logging (info, warning, erreur) sur chaque étape de traitement.

Objectif :
    - Garantir la robustesse et la traçabilité lors de l’analyse de séries temporelles (ex : COVID).
    - Faciliter le nettoyage, le filtrage par période, et la détection d’anomalies dans les données.
    - Offrir des fonctions pédagogiques : chaque étape est logguée et documentée pour l’apprentissage.

Fonctionnalités principales :
    - `str_vers_datetime` : conversion robuste de chaîne vers objet datetime, log automatique.
    - `filtrer_liste_par_date` : filtre une liste de dicts selon une plage de dates, avec stats d’erreurs.
    - `afficher_periode_liste` : affiche et loggue la période couverte par une liste, avec reporting d’erreurs.
    - `log_stats_filtrage`, `log_resumer_erreurs` : synthèse chiffrée et explicite des erreurs/filtrages.

Prérequis :
    - Données : liste de dicts (au minimum chaque dict contient une clé de date, ex : "DATE").
    - Python 3.10+ (usage de TypedDict, pattern matching, typage).
    - Le logger Python (`logging.Logger`) doit être configuré dans le script appelant.

Philosophie :
    - Centraliser la logique de parsing / qualité / reporting sur les données COVID ou séries temporelles.
    - Faciliter la maintenance et l’audit (logs structurés, reporting automatique des anomalies).
    - Rendre le code accessible et sûr même pour des débutants Python.

Utilisation typique :
    >>> from covid_filtrage_outils import filtrer_liste_par_date, afficher_periode_liste
    >>> filtres, stats = filtrer_liste_par_date(liste, logger, date_debut="2024-01-01", date_fin="2024-03-01")
    >>> afficher_periode_liste(filtres, logger)

Best Practice :
    - Utiliser systématiquement les fonctions de ce module AVANT toute analyse/agrégation.
    - Toujours passer un logger explicite pour assurer un suivi des erreurs.
    - Inspecter les logs pour vérifier la structure des données avant de continuer.

Conseils :
    - Personnaliser la clé de date (`cle_date`) si vos dictionnaires utilisent un autre champ.
    - Pour du batch, chaîner : filtrer → afficher période → exporter → logs.
    - Complétez les logs par une sauvegarde du dictionnaire de stats pour du reporting.

Limitations :
    - Ne gère pas la visualisation, l’export de fichiers, ni la connexion à des bases de données.
    - N’effectue aucun “auto-fix” des dates : les entrées invalides sont ignorées, pas corrigées.
    - Ne traite qu’une clé de date à la fois : en cas de structure complexe, pré-nettoyer en amont.

Maintenance :
    - Toute évolution de la logique de filtrage, parsing ou reporting doit être centralisée ici.
    - Modifier la structure du `StatsErreursDict` si de nouveaux types d’erreurs doivent être suivis.

Documentation :
    - Toutes les fonctions et classes suivent la convention Google/PEP 257, avec exemples d’usage.
    - Pour en savoir plus : voir la doc Python logging, datetime, typing.TypedDict.
    - Suivre les exemples d’utilisation fournis pour une intégration rapide.


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
# 'logging' gère la journalisation : création de logs pour enregistrer erreurs, infos, ou déboguer un programme.
import logging
# 'typing' : outils pour typer les fonctions/classes :
from typing import Any, Optional, Sequence, TypedDict
# Importe 'datetime' du module 'datetime' pour manipuler facilement les dates et heures (ex : obtenir la date
# courante, formater des timestamps, calculer des différences entre dates).
from datetime import datetime, timedelta

# --- Modules locaux
# EnregistrementCovidBrut : structure typée (NamedTuple/Tuple/TypedDict)
# décrivant un enregistrement COVID-19 pour une commune :
# - DATE (str) : date au format 'YYYY-MM-DD'
# - TX_DESCR_FR (str) : nom de la commune
# - CASES (int | str | None) : nombre de cas (entier, "<5", ou None)
from constantes import EnregistrementCovidBrut


class StatsErreursDict(TypedDict):
    """
    Dictionnaire typé pour résumer les statistiques d’erreurs rencontrées lors
    du filtrage ou du traitement d’une liste de dictionnaires datés.

    Clés :
        dict_non_conformes (int) :
            Nombre d’éléments ignorés car non conformes (pas un dict).
        dates_absentes (int) :
            Nombre d’éléments dont la clé de date est absente ou invalide.
        dates_invalides (int) :
            Nombre de dates présentes mais non convertibles (erreur parsing).
        total_initial (int) :
            Nombre total d’éléments dans la liste de départ.
        total_restant (int) :
            Nombre total d’éléments valides restant après filtrage.

    Note :
        - Utilisé comme second élément retourné par filtrer_liste_par_date.
        - Garantit la présence et le type de chaque clé, ce qui facilite
          l’autocomplétion et la vérification statique (type checking).
        - Permet une documentation claire et standardisée du format de retour.
        - Peut être complété ou étendu si de nouveaux types d’erreurs sont suivis.

    Example :
        >>> stats: StatsErreursDict = {
        ...     "dict_non_conformes": 1,
        ...     "dates_absentes": 2,
        ...     "dates_invalides": 1,
        ...     "total_initial": 50,
        ...     "total_restant": 46,
        ... }
        >>> print(stats["total_restant"])
        46

    Utilisation :
        - À utiliser pour tous les retours de fonctions de filtrage de listes
          ou d’analyse de données où un comptage précis des erreurs est utile.
        - Facilite le reporting, le debug, et la traçabilité des traitements.
    """
    dict_non_conformes: int
    dates_absentes: int
    dates_invalides: int
    total_initial: int
    total_restant: int


def log_datetime_exceptions(fonction):
    """
    Décorateur qui capture et loggue toutes les exceptions courantes lors de la
    conversion de chaînes en objets datetime dans les fonctions décorées.

    Args:
        fonction (Callable): Fonction à décorer, attendue avec un argument logger
            (soit positionnel, soit nommé).

    Returns:
        Callable: Version décorée de la fonction, qui retourne None en cas
            d'exception, après avoir loggué le problème avec le logger fourni.

    Note:
        - Gère automatiquement les erreurs ValueError, TypeError, OverflowError,
          AttributeError et toute autre exception inattendue.
        - Recherche le logger d'abord dans les kwargs ('logger'), puis dans les
          arguments positionnels (par type).
        - Utilise le pattern matching (match/case), disponible à partir de Python
          3.10.
        - Les messages sont adaptés à chaque type d'erreur courante.
        - Le logger doit être fourni à la fonction décorée pour logguer les erreurs.

    Example:
        >>> @log_datetime_exceptions
        ... def str_vers_datetime(date_str, logger):
        ...     return datetime.strptime(date_str, "%Y-%m-%d")
        >>> # Si date_str invalide, loggue le problème et retourne None

    Étapes:
        1. Recherche du logger dans kwargs ou args.
        2. Appelle la fonction cible, capture toute exception.
        3. Loggue un message précis selon le type d'erreur.
        4. Retourne None si une exception a été capturée.

    Tips:
        - Idéal pour sécuriser les fonctions de parsing/conversion où le format des
          données peut varier.
        - Facilite le debug : chaque échec de conversion laisse une trace dans les logs.

    Utilisation:
        À appliquer sur toute fonction de conversion de date/heure pour renforcer la
        robustesse et la lisibilité des logs.

    Limitation:
        - Le logger doit être passé à la fonction décorée pour permettre la journalisation.
        - Retourne toujours None en cas d'échec, quelle que soit l'erreur.
        - Ne lève plus d'exception : à ne pas utiliser si on veut détecter les erreurs par try/except.

    See also:
        - str_vers_datetime : conversion robuste de string vers datetime.
        - dates_str_vers_obj : conversion en lot, s’appuie sur ce décorateur.
        - Docs Python 3.10+ sur match/case.
    """
    @functools.wraps(fonction)
    def fonction_decoree(*args, **kwargs):
        logger = kwargs.get("logger", None)
        if logger is None:
            for arg in reversed(args):
                if isinstance(arg, logging.Logger):
                    logger = arg
                    break
        try:
            return fonction(*args, **kwargs)
        except Exception as exception:
            match exception:
                case ValueError() as valeur_erreur:
                    if logger:
                        logger.warning(
                            "Date invalide (valeur impossible ou format incorrect) : "
                            f"{valeur_erreur}"
                        )
                case TypeError() as type_erreur:
                    if logger:
                        logger.warning(
                            "Mauvais type (TypeError) – attendu str, reçu "
                            f"{type(args[0]).__name__} ({type_erreur})"
                        )
                case OverflowError() as depasse_limite_erreur:
                    if logger:
                        logger.warning(
                            "Valeur numérique hors limite (OverflowError) : "
                            f"{depasse_limite_erreur}"
                        )
                case AttributeError() as attribut_erreur:
                    if logger:
                        logger.warning(
                            "Objet sans attribut attendu (AttributeError) : "
                            f"{attribut_erreur}")
                case _:
                    if logger:
                        logger.error(
                            f"Erreur inattendue lors de la conversion : {exception}",
                            exc_info = True)
            return None
    return fonction_decoree


@log_datetime_exceptions
def str_vers_datetime(
        date_str: str, logger: logging.Logger, format_date: str = "%Y-%m-%d"
        ) -> Optional[datetime]:
    """
    Convertit une chaîne de caractères en objet datetime (format 'YYYY-MM-DD').
    Toutes les erreurs sont logguées explicitement via le logger fourni.

    Args:
        date_str (str): Chaîne à convertir, au format 'YYYY-MM-DD' (ou selon format_date).
        logger (logging.Logger): Logger pour enregistrer tout problème rencontré.
        format_date (str, optionnel): Format attendu pour la date (défaut : '%Y-%m-%d').

    Returns:
        datetime | None: Objet datetime si succès, sinon None (avec log détaillé).

    Note:
        - Les erreurs courantes (format, type, valeur) sont capturées et logguées.
        - Le logger est obligatoire : chaque tentative ratée sera signalée précisément.
        - Fonction décorée par log_datetime_exceptions pour robustesse maximale.

    Example:
        >>> dt = str_vers_datetime("2024-05-10", logger)
        >>> # Renvoie datetime(2024, 5, 10) ou loggue une erreur et renvoie None

    Étapes:
        1. Tente de convertir la chaîne en datetime selon format_date.
        2. Si erreur : loggue l'explication précise et retourne None.

    Tips:
        - Utiliser logger.setLevel(logging.DEBUG) pour voir tous les avertissements.
        - Bien choisir le format_date si tu traites d'autres formats.

    Utilisation:
        À utiliser pour parser des dates venant de fichiers, de l'utilisateur ou d'APIs.

    Limitation:
        - N'accepte que les chaînes correspondant au format donné.
        - Renvoie None en cas d'erreur : vérifier le retour dans le code appelant.

    See also:
        - dates_str_vers_obj (conversion de listes complètes)
        - datetime.strptime (fonction native Python)
    """
    return datetime.strptime(date_str, format_date)


def dates_str_vers_obj(
        date_sequence: Sequence, logger: logging.Logger,
        format_date: str = "%Y-%m-%d") -> list[datetime]:
    """
    Convertit une séquence de chaînes en objets datetime robustes.
    Chaque erreur est logguée via str_vers_datetime et le logger.

    Args:
        date_sequence (Sequence): Séquence de chaînes (dates à convertir).
        logger (logging.Logger): Logger pour consigner toute anomalie.
        format_date (str, optionnel): Format attendu de chaque date (défaut : '%Y-%m-%d').

    Returns:
        list[datetime]: Liste d'objets datetime convertis avec succès.

    Note:
        - Les éléments non convertibles sont ignorés (avec log d'erreur pour chacun).
        - Chaque appel utilise str_vers_datetime pour garantir la robustesse.
        - Le logger enregistre la nature et l'indice de chaque anomalie.

    Example:
        >>> logger = logging.getLogger("test")
        >>> dates = ["2023-01-01", "2023-02-30", "notadate"]
        >>> lst = dates_str_vers_obj(dates, logger)
        >>> # lst == [datetime(2023, 1, 1)]
        >>> # Les erreurs "2023-02-30" et "notadate" sont logguées

    Étapes:
        1. Pour chaque élément de la séquence : 
            a. Tente la conversion via str_vers_datetime.
            b. Si réussite, ajoute à la liste résultat.
            c. Sinon, loggue l'échec (message déjà géré).
        2. Retourne la liste finale (aucune exception non gérée).

    Tips:
        - Toujours inspecter la taille de la liste retournée vs l'input.
        - Pratique pour nettoyage de données historiques ou CSV.

    Utilisation:
        Pour importer des listes de dates depuis fichiers, APIs, saisies, etc.

    Limitation:
        - Ignore silencieusement (avec log) les entrées invalides.
        - Retourne une liste vide si aucun élément valide.

    See also:
        - str_vers_datetime (utilisé en interne)
        - datetime.strptime (Python standard)
    """
    datetime_list: list[datetime] = []
    for element_sequence in date_sequence:
        date_datetime = str_vers_datetime(
            element_sequence, logger = logger, format_date = format_date)
        if date_datetime is not None:
            datetime_list.append(date_datetime)
    return datetime_list


def filtrer_liste_par_date(
        liste: list[dict], logger: logging.Logger, cle_date: str = "DATE", 
        date_debut: Optional[str] = None, date_fin: Optional[str] = None, 
        formattage: str = "%Y-%m-%d"
        ) -> tuple[
            list[dict[str, Any]] | list[EnregistrementCovidBrut], StatsErreursDict
        ]:
    """
    Filtre une liste de dictionnaires pour ne garder que ceux dont la clé de date
    est comprise entre deux bornes optionnelles. Loggue toute erreur rencontrée,
    sans jamais lever d’exception.

    Args:
        liste (list[dict]): Liste de dictionnaires à filtrer selon la période.
        logger (logging.Logger): Logger utilisé pour consigner tous les problèmes,
            avertissements et résumés.
        cle_date (str, optionnel): Nom de la clé où trouver la date dans chaque
            dict (défaut : "DATE").
        date_debut (str, optionnel): Date minimale incluse pour filtrage,
            format selon `formattage`.
        date_fin (str, optionnel): Date maximale incluse pour filtrage,
            format selon `formattage`.
        formattage (str, optionnel): Format attendu pour les chaînes de date
            (défaut "%Y-%m-%d").

    Returns:
        tuple[
            list[dict[str, Any]] | list[EnregistrementCovidBrut],
            StatsErreursDict
        ]: 
            - La liste filtrée des éléments dont la date respecte les bornes
              (type et structure conservés).
            - Un dict typé (StatsErreursDict) résumant : nb d’éléments initiaux,
              restants, et types d’erreurs rencontrés
              (clés : "dict_non_conformes", "dates_absentes", "dates_invalides",
              "total_initial", "total_restant").

    Note:
        - Le résumé d’erreurs est structuré grâce au TypedDict StatsErreursDict,
          ce qui permet une autocomplétion précise et une documentation fiable.
        - Ne s’arrête jamais sur une erreur : chaque anomalie est logguée.
        - Les dates sont parsées par `str_vers_datetime` (robuste : décorateur
          de gestion d’exceptions).
        - Si aucune borne n’est fournie, retourne la liste telle quelle,
          avec statistiques (aucun filtrage).
        - Tous les problèmes de structure, de type ou de parsing sont inclus
          dans le résumé d’erreurs loggué à la fin.

    Example:
        >>> filtres, stats = filtrer_liste_par_date(
        ...     lst, logger, date_debut="2023-01-01", date_fin="2023-06-01")
        >>> print(filtres)
        # [dicts compris entre le 01/01 et le 01/06]
        >>> print(stats)
        # {'dict_non_conformes': 1, 'dates_absentes': 2, ...}

    Étapes:
        1. Initialise les compteurs d’erreurs.
        2. Parse les bornes si fournies (avec gestion d’erreurs/log).
        3. Parcourt chaque dict :
            a. Vérifie type, présence et format de la clé date.
            b. Tente de parser la date, loggue si erreur.
            c. Ne conserve que si dans les bornes.
        4. Loggue un résumé des erreurs, retourne la liste filtrée et les stats.

    Tips:
        - Utile pour nettoyer ou sélectionner une période précise sur des données
          brutes importées.
        - Pour toute analyse série temporelle, toujours valider le nb d’éléments filtrés.
        - Les erreurs sont affichées dans les logs, jamais dans le retour Python.

    Utilisation:
        À appliquer juste après lecture d’un fichier de données datées pour
        travailler sur une période, ou avant tout calcul/agrégation.

    Limitation:
        - Les dates invalides ou manquantes sont simplement ignorées (avec log).
        - Le filtrage ne modifie jamais la liste d’origine.
        - En cas de mauvaise structure de dict (ou None, ou liste vide), tout est loggué
          et la fonction retourne une liste filtrée (souvent vide).

    See also:
        - str_vers_datetime (conversion robuste des dates)
        - log_resumer_erreurs (pour résumer les erreurs rencontrées)
        - afficher_periode_liste (pour valider la période sur la liste filtrée)
    """
    # --- Initialisation des stats : Compteurs d'erreurs ---
    statistiques_erreurs: StatsErreursDict = {
        "dict_non_conformes": 0,
        "dates_absentes": 0,
        "dates_invalides": 0,
        "total_initial": len(liste),
        "total_restant": 0,
    }

   # --- Préparation des bornes ---
    if not date_debut and not date_fin:
        statistiques_erreurs["total_restant"] = len(liste)
        return liste, statistiques_erreurs
    
    try:
        borne_debut = str_vers_datetime(
            date_debut, logger, formattage) if date_debut else None
        borne_fin = str_vers_datetime(
            date_fin, logger, formattage) if date_fin else None
    except Exception as exception:
        logger.error(f"Date de borne invalide : {exception}", exc_info = True)
        statistiques_erreurs["total_restant"] = 0
        log_resumer_erreurs(logger, **statistiques_erreurs)
        return [], statistiques_erreurs

    resultat_filtre_par_date = []
    for index_element, element_de_la_liste in enumerate(liste):
        try:
            # Cas 1 : Non-dict
            if not isinstance(element_de_la_liste, dict):
                logger.warning(
                    f"L’élément d’indice {index_element} ignoré : ce n’est pas "
                    f"un dict {element_de_la_liste!r}")
                statistiques_erreurs["dict_non_conformes"] += 1
                continue 
            date_str = element_de_la_liste.get(cle_date)
            # Cas 2 : Clé absente/invalide
            if not isinstance(date_str, str) or not date_str.strip():
                logger.warning(
                    f"Élément à l’indice {index_element} ignoré : clé '{cle_date}' "
                    f"absente ou valeur de date invalide ({element_de_la_liste!r})")
                statistiques_erreurs["dates_absentes"] += 1
                continue
            try:
                # Cas 3 : Parsing raté
                date_datetime = str_vers_datetime(date_str, logger, formattage)
            except Exception as exception:
                logger.warning(
                    f"Élément à l’indice {index_element} ignoré : date '{date_str}' "
                    f"non convertible ({exception})"
                )
                statistiques_erreurs["dates_invalides"] += 1
                continue
            # --- Filtrage sur bornes ---
            if borne_debut and date_datetime < borne_debut:
                continue
            if borne_fin and date_datetime > borne_fin:
                continue
            resultat_filtre_par_date.append(element_de_la_liste)

        except Exception as exception:
            logger.error(
                f"Erreur inattendue pour l'élément à indice {index_element}: "
                f"{element_de_la_liste!r} — {exception}", exc_info = True)
            statistiques_erreurs["dict_non_conformes"] += 1

    # --- Résumé des erreurs ---
    statistiques_erreurs["total_restant"] = len(resultat_filtre_par_date)
    log_resumer_erreurs(logger, **statistiques_erreurs)
    return resultat_filtre_par_date, statistiques_erreurs


def afficher_periode_liste(
        liste_de_dictionnaire: list[dict], logger: logging.Logger, 
        cle_date: str = "DATE", formattage: str = "%Y-%m-%d") -> None:
    """
    Affiche dans les logs la période (date min et max) couverte par une liste de
    dictionnaires contenant une clé de date. Gère toutes les erreurs de structure,
    de valeur ou de parsing, et utilise un dictionnaire typé StatsErreursDict pour
    comptabiliser précisément les différents types d’erreurs rencontrées.

    Args:
        liste_de_dictionnaire (list[dict]): Liste à analyser, chaque élément doit
            être un dictionnaire comportant une clé de date.
        logger (logging.Logger): Logger utilisé pour consigner tous les messages
            (infos, avertissements, erreurs).
        cle_date (str, optionnel): Nom de la clé où trouver la date (défaut "DATE").
        formattage (str, optionnel): Format attendu de la chaîne de date
            (défaut "%Y-%m-%d").

    Returns:
        None: Aucun retour direct, tout est consigné dans les logs.

    Note:
        - Utilise un dictionnaire typé StatsErreursDict pour stocker et reporter
          les compteurs d’erreurs (voir la définition pour le détail des clés).
        - Chaque erreur rencontrée (élément non dict, clé absente, date invalide)
          est comptabilisée et logguée, sans interrompre l’exécution.
        - La période affichée correspond uniquement aux dates correctement parsées.
        - Un résumé détaillé des erreurs est affiché à la fin dans les logs.

    Example:
        >>> afficher_periode_liste([
        ...     {"DATE": "2023-01-01"},
        ...     {"DATE": "notadate"},
        ...     {"AUTRE": "valeur"},
        ...     "pas_un_dict"
        ... ], logger)
        # Logs :
        # "Clé 'DATE' absente ou invalide..." (pour {"AUTRE": ...})
        # "Date 'notadate' invalide..." (pour {"DATE": "notadate"})
        # "Le dict à l'index 3 est ignoré..." (pour "pas_un_dict")
        # "Période couverte : 2023-01-01 → 2023-01-01"
        # "Dont : 1 dict(s) non conformes ; 1 dict(s) sans date valide ;
        #         1 date(s) non convertibles ignoré(s)."

    Étapes:
        1. Initialise un dictionnaire d’erreurs typé (StatsErreursDict).
        2. Parcourt chaque élément de la liste et vérifie sa structure.
        3. Cherche la clé de date et tente de parser la date.
        4. Comptabilise chaque erreur par type et loggue un avertissement.
        5. Si au moins une date valide : calcule min et max, affiche la période.
        6. Affiche un résumé détaillé des erreurs rencontrées dans les logs.

    Tips:
        - Pratique pour vérifier rapidement la couverture temporelle d’un jeu
          de données avant traitement ou export.
        - Changer la clé cle_date si tes dictionnaires utilisent un autre nom.

    Utilisation:
        Appeler juste après import ou nettoyage d’une liste de données datées,
        pour valider leur période et repérer d’éventuels soucis de structure.

    Limitation:
        - La fonction ne retourne rien et ne lève jamais d’exception.
        - Les erreurs sont seulement logguées, pas accessibles par le code appelant.
        - Les dates mal formatées ou absentes sont ignorées dans le calcul.

    See also:
        - StatsErreursDict (définition des compteurs d’erreurs utilisés)
        - str_vers_datetime (conversion robuste des chaînes de dates)
        - filtrer_liste_par_date (filtrage des listes selon une période)
    """
    liste_dates_datetime = []

    # --- Stats structurées (TypedDict) ---
    stats: StatsErreursDict = {
        "dict_non_conformes": 0,
        "dates_absentes": 0,
        "dates_invalides": 0,
        "total_initial": len(liste_de_dictionnaire),
        "total_restant": 0,
    }

    for index_dico, dico in enumerate(liste_de_dictionnaire):
        try:
            # Cas 1 : l'élément n'est pas un dict
            if not isinstance(dico, dict):
                logger.warning(
                    f"Le dict à l'index {index_dico} est ignoré "
                    f"(pas un dict) : {dico!r}")
                stats["dict_non_conformes"] += 1
                continue
            date_str = dico.get(cle_date)
            # Cas 2 : la clé de date n'est pas trouvée ou n'est pas une str non vide
            if not isinstance(date_str, str) or not date_str.strip():
                logger.warning(
                    f"Clé '{cle_date}' absente ou invalide dans le dictionnaire "
                    f"à l'index {index_dico} : {dico!r}")
                stats["dates_absentes"] += 1
                continue
            try:
                # Cas 3 : parsing de date échoue
                date_datetime = str_vers_datetime(date_str, logger, formattage)
                liste_dates_datetime.append(date_datetime)
            except Exception as exception:
                logger.warning(
                    f"Date '{date_str}' invalide (dict {index_dico}), ignorée "
                    f"({exception})")
                stats["dates_invalides"] += 1
        except Exception as exception:
            logger.error(
                f"Erreur inattendue lors du traitement du dict {index_dico} : "
                f"{dico!r} — {exception}", exc_info=True)
            stats["dict_non_conformes"] += 1
            continue

    # Affichage de la période si au moins une date valide
    if not liste_dates_datetime:
        logger.info("Aucune date valide trouvée dans la liste.")
        stats["total_restant"] = 0
    else:
        try:
            debut_datetime, fin_datetime = (min(liste_dates_datetime),
                                            max(liste_dates_datetime))
            logger.info(
                f"Période couverte : {debut_datetime.date()} → {fin_datetime.date()}")
            stats["total_restant"] = len(liste_dates_datetime)
            if stats["dates_invalides"] > 0:
                logger.info(
                    f"{stats['dates_invalides']} date(s) ignorée(s) "
                    f"(format ou clé invalide).")
        except Exception as exception:
            logger.error(
                f"Erreur lors du calcul min/max des dates : {exception}", 
                exc_info=True)
            stats["total_restant"] = 0

    # Résumé des erreurs via StatsErreursDict
    log_resumer_erreurs(logger, stats)


def log_stats_filtrage(logger: logging.Logger, stats: StatsErreursDict) -> None:
    """
    Affiche dans les logs un résumé synthétique et détaillé du filtrage d’une liste
    en utilisant un dictionnaire typé StatsErreursDict pour toutes les statistiques.
    Les pourcentages et les erreurs détectées sont clairement reportés dans les logs.

    Args:
        logger (logging.Logger): Logger à utiliser pour afficher le résumé.
        stats (StatsErreursDict): Dictionnaire typé résumant les statistiques de
            filtrage, avec les clés suivantes :
                - "total_initial" : Nombre d’éléments avant filtrage
                - "total_restant" : Nombre d’éléments restants
                - "dict_non_conformes" : Dictionnaires mal formés/ignorés
                - "dates_absentes" : Dictionnaires sans clé de date valide
                - "dates_invalides" : Dictionnaires avec date non convertible

    Returns:
        None

    Note:
        - Tous les compteurs sont garantis présents par l’usage du TypedDict.
        - Les lignes de log sont toujours cohérentes avec la structure des stats.
        - Si "total_initial" vaut zéro, les pourcentages sont affichés à 0 %.

    Example:
        >>> stats = StatsErreursDict(
        ...     total_initial=100,
        ...     total_restant=95,
        ...     dict_non_conformes=2,
        ...     dates_absentes=2,
        ...     dates_invalides=1
        ... )
        >>> log_stats_filtrage(logger, stats)
        # Affiche le résumé du filtrage avec détails des erreurs.

    Étapes:
        1. Calcule le nombre total, restant, filtré, et les pourcentages associés.
        2. Affiche le résumé global dans les logs.
        3. Affiche un détail des erreurs rencontrées, si présentes.

    Tips:
        - À utiliser après tout filtrage de liste de dicts datés.
        - Les erreurs doivent être renseignées dans le StatsErreursDict fourni.

    See also:
        - StatsErreursDict (définition du type)
        - filtrer_liste_par_date (qui génère ce dict)
        - log_resumer_erreurs (pour d’autres types de résumés)
    """
    total = stats["total_initial"]
    restant = stats["total_restant"]
    filtres = total - restant
    percent_ok = (restant / total * 100) if total else 0
    percent_filtre = (filtres / total * 100) if total else 0

    logger.info(
        f"Filtrage effectué : {restant} ({percent_ok:.1f}%) éléments valides "
        f"sur {total} ({filtres} ({percent_filtre:.1f}%) filtrés)."
    )
    resume = []
    if stats["dict_non_conformes"]:
        resume.append(f"{stats['dict_non_conformes']} dict(s) non conformes")
    if stats["dates_absentes"]:
        resume.append(f"{stats['dates_absentes']} dict(s) sans date valide")
    if stats["dates_invalides"]:
        resume.append(f"{stats['dates_invalides']} date(s) non convertibles")
    if resume:
        logger.info("Dont : " + " ; ".join(resume) + " ignoré(s).")


def log_resumer_erreurs(
    logger: logging.Logger,
    stats: StatsErreursDict = None,
    **autres,
) -> None:
    """
    Loggue un résumé clair et standardisé des erreurs rencontrées lors du
    traitement d'une liste de dictionnaires, comme après un filtrage ou une
    conversion de dates. Prend un StatsErreursDict complet (peut être None)
    et accepte d'autres compteurs via des arguments nommés.

    Args:
        logger (logging.Logger): Logger utilisé pour afficher le résumé.
        stats (StatsErreursDict, optionnel): Dictionnaire typé des stats
            principales (par défaut None, peut être partiel).
        **autres: Autres compteurs d'erreurs éventuels à afficher, passés
            en mots-clés supplémentaires.

    Returns:
        None

    Note:
        - Combine les stats du TypedDict et celles passées en mots-clés.
        - Ignore tous les compteurs à zéro ou absents.
        - Affiche la ligne uniquement s’il y a au moins une erreur.

    Example:
        >>> stats = StatsErreursDict(
        ...     dict_non_conformes=2, dates_absentes=0, dates_invalides=1,
        ...     total_initial=10, total_restant=7
        ... )
        >>> log_resumer_erreurs(logger, stats)
        # Log : "Dont : 2 dict(s) non conformes ; 1 date(s) non convertibles ignoré(s)."

        >>> log_resumer_erreurs(logger, autre_erreur=3)
        # Log : "Dont : 3 autre erreur ignoré(s)."

    Étapes:
        1. Prend tous les compteurs d’erreurs dans le TypedDict et les kwargs.
        2. Construit la liste des messages pour tous les compteurs non nuls.
        3. Loggue la ligne de résumé si la liste n’est pas vide.

    Tips:
        - À appeler après filtrage ou conversion pour avoir un log centralisé.
        - Combine stats structurées et personnalisées facilement.

    Limitation:
        - Ne lève jamais d’exception.
        - Si tous les compteurs sont à zéro, n’affiche rien.

    See also:
        - log_stats_filtrage (pour le résumé chiffré complet)
        - filtrer_liste_par_date (qui génère un StatsErreursDict)
    """
    resume_erreurs = []
    if stats is not None:
        if stats.get("dict_non_conformes", 0):
            resume_erreurs.append(
                f"{stats['dict_non_conformes']} dict(s) non conformes"
            )
        if stats.get("dates_absentes", 0):
            resume_erreurs.append(
                f"{stats['dates_absentes']} dict(s) sans date valide"
            )
        if stats.get("dates_invalides", 0):
            resume_erreurs.append(
                f"{stats['dates_invalides']} date(s) non convertibles"
            )
    for cle, valeur in autres.items():
        if valeur:
            resume_erreurs.append(f"{valeur} {cle.replace('_', ' ')}")
    if resume_erreurs:
        logger.info("Dont : " + " ; ".join(resume_erreurs) + " ignoré(s).")


def suffixe_periode(
    periode: Optional[tuple[Optional[str], Optional[str]]]) -> str:
    """
    Génère un suffixe de nom de fichier indiquant la période analysée,
    au format '_YYYY-MM-DD_to_YYYY-MM-DD', ou retourne une chaîne vide si absent.

    Args:
        periode (tuple[Optional[str], Optional[str]]): Période à encoder, sous
            forme de tuple (date_debut, date_fin), chaque élément pouvant être
            une chaîne (format 'YYYY-MM-DD') ou None.

    Returns:
        str: Suffixe à ajouter au nom de fichier, par exemple
            '_2020-01-01_to_2020-12-31', ou '' si la période est vide ou nulle.

    Note:
        - Gère les cas où une seule borne est renseignée (début ou fin).
        - Remplace chaque valeur absente (None) par 'xxxx-xx-xx' dans le suffixe.
        - Ne vérifie pas le format exact des dates (suppose des chaînes valides).

    Example:
        >>> suffixe_periode(("2020-01-01", "2020-12-31"))
        '_2020-01-01_to_2020-12-31'
        >>> suffixe_periode((None, "2022-03-15"))
        '_xxxx-xx-xx_to_2022-03-15'
        >>> suffixe_periode(None)
        ''
        >>> suffixe_periode((None, None))
        ''

    Étapes:
        1. Vérifie que le tuple de période est fourni et qu’au moins une borne est présente.
        2. Remplace toute valeur None par 'xxxx-xx-xx' pour signaler l’absence de borne.
        3. Retourne le suffixe sous la forme '_date_debut_to_date_fin'.
        4. Si la période est absente, retourne une chaîne vide.

    Tips:
        - Idéal pour nommer des fichiers exportés selon la période de données traitée.
        - Permet de retrouver facilement la période analysée dans l’arborescence de fichiers.

    Utilisation:
        À utiliser lors de la sauvegarde de fichiers pour des exports batch ou
        des rapports filtrés par période, afin d’éviter tout écrasement et
        d’améliorer la traçabilité.

    Limitation:
        - Ne contrôle pas le format ou la cohérence des dates fournies.
        - Utilise 'xxxx-xx-xx' comme placeholder si une borne est manquante.

    See also:
        - Toute fonction de sauvegarde/export de fichiers nommés par période.
        - filtrer_liste_par_date pour générer une période à encoder ici.
    """
    if periode and (periode[0] or periode[1]):
        debut = periode[0] or 'xxxx-xx-xx'
        fin   = periode[1] or 'xxxx-xx-xx'
        return f"_{debut}_to_{fin}"
    return ""


def dates_apres(date_str, jours_suivants):
    """Retourne une liste de dates (YYYY-MM-DD) à partir de date_str + jours_suivants."""
    base_date = datetime.strptime(date_str, "%Y-%m-%d")
    return [
        (base_date + timedelta(days=delta)).strftime("%Y-%m-%d")
        for delta in range(jours_suivants + 1)
    ]


def dates_autour(date_str, nb_jours):
    """Retourne une liste de dates x jours avant et après la date donnée (inclus)."""
    base_date = datetime.strptime(date_str, "%Y-%m-%d")
    return [
        (base_date + timedelta(days=delta)).strftime("%Y-%m-%d")
        for delta in range(-nb_jours, nb_jours + 1)
    ]