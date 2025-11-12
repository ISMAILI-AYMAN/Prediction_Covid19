# -*- coding: utf-8 -*-

"""
utils_io.py
-----------
Outils réutilisables pour la gestion des logs et l’intégration des avertissements Python
dans tous vos scripts ou projets. Ce module centralise l’initialisation des loggers,
la personnalisation du format de sortie console, et la redirection des warnings.

Ce module fournit :
    - Des fonctions pour créer/configurer des loggers (console uniquement, format lisible).
    - Un mécanisme pour rediriger les warnings Python dans les logs (niveau WARNING).
    - Des décorateurs pour tracer automatiquement le début, la fin et la durée d’exécution
      de vos fonctions ou étapes de pipeline.

Objectif :
    - Simplifier la gestion centralisée des logs pour tous les scripts Python d’un projet.
    - Permettre une traçabilité propre, pédagogique et automatisable des étapes importantes.

Fonctionnalités principales :
    - obtenir_logger() : création rapide d’un logger prêt à l’emploi (niveau configurable).
    - creer_logger(), ajouter_sortie_console_au_logger() : création fine et ajout de handlers.
    - configurer_logging() : combine création logger + redirection warnings.
    - redirection_avertissement() : attrape tous les warnings Python pour les logguer.
    - Décorateurs log_debut_fin / log_debut_fin_logger_dynamique : traces automatiques
      de toutes les fonctions importantes (début, fin, chrono, étape, etc).

Prérequis :
    - Aucune dépendance externe (librairies Python standard uniquement).
    - Pour tirer profit du logging avancé, utiliser Python >= 3.10 recommandé.
    - Aucun format de données particulier requis : s’intègre à tous vos scripts.

Philosophie :
    - Centralise la configuration des logs et warnings pour éviter les pièges débutants.
    - Facilite le debug et l’audit (exécution pipeline, étapes, erreurs...).
    - Favorise la maintenance et la clarté sur tous les projets partagés.

Utilisation typique :
    >>> from utils_io import obtenir_logger, log_debut_fin
    >>> logger = obtenir_logger("mon_script", est_detaille=True)
    >>> @log_debut_fin(logger)
    ... def ma_fonction():
    ...     pass
    >>> ma_fonction()

Best Practice :
    - Toujours appeler obtenir_logger/configurer_logging au début de chaque script.
    - Nommer vos loggers selon le module/fichier pour filtrer facilement les logs.
    - Utiliser les décorateurs sur les fonctions-clés pour tracer vos traitements.

Conseils :
    - Pour du logging dans un fichier, ajoutez un FileHandler sur le logger.
    - Pour les projets à plusieurs modules : centralisez la config logger dans ce module.

Limitations :
    - Pas de log dans un fichier ou sur le réseau par défaut (console seule).
    - Un seul logger d’avertissements Python global (warning) par processus Python.
    - Ne filtre pas automatiquement les doublons de warnings (voir warnings.simplefilter).

Maintenance :
    - Toute évolution sur la logique de logs/warnings doit se faire ici pour garder la cohérence.
    - Les logs sont là pour vous aider à comprendre, déboguer ou auditer vos traitements.

Documentation :
    - Toutes les fonctions/décorateurs sont documentés en style Google/PEP 257, avec exemples.
    - Pour en savoir plus sur logging : https://docs.python.org/3/library/logging.html
    - Sur warnings : https://docs.python.org/3/library/warnings.html

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
# 'os' : outils pour manipuler le système de fichiers : créer/effacer des dossiers, lister ou déplacer des fichiers.
import os
# 'time' : fonctions pour mesurer précisément la durée d’une opération, faire des pauses (sleep) ou chronométrer.
import time
# 'warnings' : personnalise la gestion des avertissements Python (masquer, filtrer ou rediriger les warnings).
import warnings
# 'typing' : outils pour typer les fonctions/classes : Callable (fonction), TypeVar (générique), Any (tout type).
from typing import Callable, TypeVar, Any

# --- Variables globales ---
# Compteur global pour numéroter chaque étape dans les logs (utile pour les pipelines).
ETAPE: int = 0
# F : TypeVar borné à Callable, sert à typer proprement les décorateurs génériques.
# Garantit l’auto-complétion et la vérification de type sur toute fonction décorée.
F = TypeVar("F", bound = Callable[..., Any])
# Logger pour rediriger les warnings Python ; None par défaut tant qu’il n’est pas configuré.
logger_avertissement:logging.Logger | None = None

def obtenir_logger(nom_logger: str, est_detaille: bool = False) -> logging.Logger:
    """
    Crée et retourne un logger configuré pour l’application, avec niveau de détail ajustable.

    Args:
        nom_logger (str): Nom du logger, pour distinguer les modules/scripts.
        est_detaille (bool, optionnel): Si True, active le mode DEBUG (détails max). False=INFO.

    Returns:
        logging.Logger: Logger Python prêt à l’emploi, configuré pour afficher
            les logs au bon niveau et rediriger les avertissements.

    Note:
        - Utilise configurer_logging pour gérer niveau et handlers.
        - Redirige aussi les warnings Python vers le logger.
        - Compatible Python 3.10+.

    Example:
        >>> logger = obtenir_logger("mon_script", est_detaille=True)
        >>> logger.info("Message d'information")
        >>> logger.debug("Message très détaillé")

    Étapes:
        1. Définit le niveau (DEBUG ou INFO).
        2. Appelle configurer_logging.
        3. Retourne le logger prêt à l’emploi.

    Tips:
        - Utiliser un nom unique pour chaque module pour distinguer les logs.
        - Passer est_detaille=True pour le debug ou le développement.

    Utilisation:
        À appeler en tout début de script ou module avant d’émettre des logs.

    Limitation:
        - Un seul logger par nom : les logs de modules du même nom se mélangeront.
        - L’écriture dans un fichier log n’est pas configurée ici (console seule).

    See also:
        - configurer_logging, creer_logger, ajouter_sortie_console_au_logger.
    """

    niveau = logging.DEBUG if est_detaille else logging.INFO
    return configurer_logging(niveau = niveau, nom_logger = nom_logger)


def creer_logger(nom: str, niveau: int) -> logging.Logger:
    """
    Crée un logger Python avec le nom et le niveau de journalisation souhaités.

    Args:
        nom (str): Nom du logger.
        niveau (int): Niveau de logs (logging.INFO, logging.DEBUG...).

    Returns:
        logging.Logger: Logger configuré, sans handlers en double et sans propagation.

    Note:
        - Supprime tous les handlers existants pour éviter plusieurs sorties identiques.
        - Évite la propagation vers le root logger, pour ne pas avoir de doublons dans la console.
        - Compatible Python 3.10+.

    Example:
        >>> logger = creer_logger("test", logging.INFO)
        >>> logger.info("Hello logger !")

    Étapes:
        1. Récupère (ou crée) un logger nommé.
        2. Fixe son niveau (INFO, DEBUG...).
        3. Supprime les handlers existants (s'il y en a).
        4. Désactive la propagation vers le logger racine.

    Tips:
        - Toujours compléter avec ajouter_sortie_console_au_logger
          pour avoir une sortie visible.
        - Prendre soin de ne pas réutiliser le même nom de logger partout.

    Utilisation:
        Pour créer un logger “neuf” dans chaque module/sous-module.

    Limitation:
        - Ne crée pas de handler console par défaut : ajouter après la fonction.
    """
    logger = logging.getLogger(nom)
    logger.setLevel(niveau)
    # Supprime tous les handlers existants pour éviter les doublons
    if logger.hasHandlers():
        logger.handlers.clear()
    # Évite la remontée des logs au root logger
    logger.propagate = False
    return logger


def ajouter_sortie_console_au_logger(logger: logging.Logger) -> None:
    """
    Ajoute une sortie console (StreamHandler) au logger, formatée pour lecture humaine.

    Args:
        logger (logging.Logger): Logger cible à configurer pour afficher les logs.

    Returns:
        None

    Note:
        - Supprime d'abord tous les handlers existants pour éviter les doublons d’affichage.
        - Formate les logs : date, niveau, message (lisible en console).
        - Compatible Python 3.10+.

    Example:
        >>> logger = creer_logger("test", logging.DEBUG)
        >>> ajouter_sortie_console_au_logger(logger)
        >>> logger.info("Ceci s'affiche joliment formaté.")

    Étapes:
        1. Supprime tous les handlers déjà présents.
        2. Crée et configure un handler console (StreamHandler).
        3. Ajoute ce handler au logger.

    Tips:
        - Utile pour les scripts lancés en ligne de commande.
        - Pour loguer dans un fichier : ajouter un FileHandler (hors scope ici).

    Utilisation:
        À appeler une fois par logger (sinon, plusieurs sorties possibles).

    Limitation:
        - Un seul handler ajouté : pas de log fichier/log réseau par défaut.
        - N’influence pas les logs enfants si la propagation est désactivée.
    """
    # Supprime tous les handlers déjà présents
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def avertissement_personnalise(
        message: str, categorie: type[Warning], fichier: str, ligne: int,
        file: object = None, line: object = None) -> None:
    """
    Handler global pour rediriger les warnings Python dans le logger courant.

    Il faut que logger_avertissement soit défini avant appel (global dans ce module).

    Args:
        message (str): Texte du warning à afficher/logguer.
        categorie (type): Classe du warning (UserWarning, DeprecationWarning...).
        fichier (str): Nom du fichier source où le warning est généré.
        ligne (int): Numéro de la ligne source.
        file, line (object, optionnel): Non utilisés, imposés par l’API warnings.

    Returns:
        None

    Note:
        - Si logger_avertissement est None, affiche dans la console.
        - Sinon, loggue le warning dans le logger configuré.
        - Cette fonction doit être référencée par warnings.showwarning.
        - Compatible Python 3.10+.

    Example:
        >>> warnings.warn("Attention !")
        # Sera loggué via logger_avertissement si configuré.

    Étapes:
        1. Vérifie si logger_avertissement est disponible.
        2. Si oui, loggue le warning (niveau WARNING).
        3. Sinon, imprime sur la sortie standard.

    Tips:
        - À ne pas appeler directement, mais via warnings.showwarning.
        - Pour centraliser tous les avertissements Python dans les logs.

    Utilisation:
        S’utilise pour remplacer le comportement standard de warnings.warn dans tous les scripts.

    Limitation:
        - Fonctionne module-global seulement (un logger “avertissement” à la fois).
        - Pas de formatage “avancé” du message, seulement texte et infos de base.

    See also:
        - warnings.warn, warnings.showwarning, redirection_avertissement
    """
    # Utilise la variable globale (ou module) logger_avertissement
    global logger_avertissement
    if logger_avertissement is not None:
        logger_avertissement.warning(f"{categorie.__name__}: {message} ({fichier}:{ligne})")
    else:
        # En dernier recours, warning classique
        print(f"{categorie.__name__}: {message} ({fichier}:{ligne})")


def redirection_avertissement(logger: logging.Logger) -> None:
    """
    Redirige tous les warnings Python vers le logger fourni,
    pour centraliser les avertissements dans les logs.

    Args:
        logger (logging.Logger): Logger cible pour les avertissements.

    Returns:
        None

    Note:
        - Modifie warnings.showwarning pour pointer vers avertissement_personnalise.
        - Nécessite un logger global logger_avertissement (défini ici).
        - Compatible Python 3.10+.

    Example:
        >>> logger = obtenir_logger("test")
        >>> redirection_avertissement(logger)
        >>> warnings.warn("Ceci sera loggé comme WARNING.")

    Étapes:
        1. Met à jour la variable globale logger_avertissement.
        2. Redéfinit warnings.showwarning pour utiliser le logger.
        3. Tous les warnings Python sont ensuite loggués, plus affichés en clair.

    Tips:
        - À appeler avant d’utiliser des warnings.warn dans vos scripts.
        - Idéal pour capter les problèmes de dépréciation, erreurs silencieuses, etc.

    Utilisation:
        À placer une fois par programme ou module principal.

    Limitation:
        - Écrase la redirection des warnings pour tout le process Python courant.
        - Un seul logger “avertissement” global à la fois.
    """

    global logger_avertissement
    logger_avertissement = logger
    warnings.showwarning = avertissement_personnalise


def configurer_logging(niveau: int, nom_logger: str) -> logging.Logger:
    """
    Crée, configure un logger prêt à l'emploi, ajoute la sortie console et redirige les warnings.

    Args:
        niveau (int): Niveau du logging (logging.INFO, logging.DEBUG, ...).
        nom_logger (str): Nom du logger à créer/configurer.

    Returns:
        logging.Logger: Logger totalement prêt, sortie console, warnings redirigés.

    Note:
        - Combine creation du logger, ajout du handler console, redirection warnings.
        - Le logger est prêt pour utilisation dans tout script ou module Python.
        - Compatible Python 3.10+.

    Example:
        >>> logger = configurer_logging(logging.INFO, "mon_script")
        >>> logger.info("Tout est prêt !")
        >>> warnings.warn("Attention, warning dans les logs.")

    Étapes:
        1. Crée un logger (nom et niveau voulu).
        2. Ajoute la sortie console formatée.
        3. Redirige warnings Python dans le logger.
        4. Retourne le logger.

    Tips:
        - Pour la plupart des usages, utiliser cette fonction unique au début du script.
        - Changez le niveau (DEBUG, INFO, WARNING) selon besoin.

    Utilisation:
        À utiliser dans tout script principal pour centraliser la config logs.

    Limitation:
        - Le logger ne logge que sur la console (pas de fichier).
        - Pas d’options avancées (multiples handlers, rotation, etc.).
    """
    logger = creer_logger(nom_logger, niveau)
    ajouter_sortie_console_au_logger(logger)
    redirection_avertissement(logger)
    return logger


def log_debut_fin(logger: logging.Logger) -> Callable[[F], F]:
    """
    Décorateur : loggue automatiquement le début et la fin d’une fonction, avec chrono et étape.

    Args:
        logger (logging.Logger): Logger utilisé pour journaliser chaque appel.

    Returns:
        Callable[[F], F]: Décorateur générique : retourne une fonction décorée avec exactement
            la même signature et le même type de retour que la fonction d’origine.
            (Astuce typage : F est un “filet de sécurité” pour que l’auto-complétion et les
            vérifications de type (MyPy, Pyright…) restent correctes, même avec n’importe
            quelle fonction décorée.)

    Note:
        - Incrémente une variable globale ETAPE à chaque appel décoré.
        - Loggue début, nom de fichier, nom de la fonction, ligne, durée.
        - Ajoute une barre d’underscore pour la lisibilité dans les logs.
        - F = TypeVar(“F”, bound=Callable[…]) permet de typer le décorateur sans perdre
          les types d’entrée et de sortie : auto-complétion, aide IDE, et vérifications
          type checker restent actives pour toutes les fonctions décorées.
        - Compatible Python 3.10+.

    Example:
        >>> @log_debut_fin(logger)
        ... def traitement():
        ...     time.sleep(1)
        ...     return 42
        >>> traitement()
        # Logue [DEBUT], puis [FIN], et la durée.

    Étapes:
        1. Incrémente ETAPE.
        2. Loggue début, fichier, fonction, ligne.
        3. Mesure la durée d’exécution.
        4. Loggue fin, durée.
        5. Retourne le résultat de la fonction décorée.

    Tips:
        - Idéal pour les pipelines ou fonctions longues.
        - Permet de tracer précisément chaque étape et durée de calcul.

    Utilisation:
        À appliquer sur chaque fonction à monitorer pour l’audit ou le debug.

    Limitation:
        - La variable ETAPE est globale et partagée pour toutes les fonctions décorées.
        - Un seul logger fixé à la création du décorateur.
    """

    def decorateur(fonction):
        @functools.wraps(fonction)
        def fonction_decoree(*args, **kwargs):
            global ETAPE
            ETAPE += 1
            fichier = os.path.basename(fonction.__code__.co_filename)
            numero_ligne = fonction.__code__.co_firstlineno

            message_debut = (
                f"[DEBUT] ÉTAPE {ETAPE} : {fichier} - {fonction.__name__} → "
                f"ligne {numero_ligne}"
            )
            logger.info("\n")
            logger.info(message_debut)
            logger.info(len(message_debut) * "_" + "\n")
            t0 = time.time()
            resultat = fonction(*args, **kwargs)
            duree = time.time() - t0
            message_fin = (
                f"[FIN] {fichier} - {fonction.__name__} → ligne {numero_ligne} "
                f"- durée {duree:.3f}s\n"
            )
            logger.info(message_fin)
            return resultat
        return fonction_decoree
    return decorateur


def log_debut_fin_logger_dynamique(nom_logger: str = "logger") -> Callable[[F], F]:
    """
    Décorateur avancé : loggue début et fin de fonction, avec recherche dynamique du logger.

    Args:
        nom_logger (str, optionnel): Nom d’attribut contenant le logger si dans une classe,
            ou recherche dans les arguments (kwargs, puis args).

    Returns:
        Callable[[F], F]: Décorateur générique : retourne une fonction décorée avec exactement
            la même signature et le même type de retour que la fonction d’origine.
            (Astuce typage : F est un “filet de sécurité” pour que l’auto-complétion et les
            vérifications de type (MyPy, Pyright…) restent correctes, même avec n’importe
            quelle fonction décorée.)

    Note:
        - Cherche d’abord le logger dans les kwargs, puis args, puis dans self.<nom_logger>.
        - Permet d’utiliser ce décorateur même quand le logger est injecté ou dans un objet.
        - F = TypeVar(“F”, bound=Callable[…]) permet de typer le décorateur sans perdre
          les types d’entrée et de sortie : auto-complétion, aide IDE, et vérifications
          type checker restent actives pour toutes les fonctions décorées.
        - Compatible Python 3.10+.

    Example:
        >>> @log_debut_fin_logger_dynamique()
        ... def ma_fonction(x, logger=None):
        ...     pass
        >>> ma_fonction(1, logger=logger)
        # Logue début et fin via logger fourni.

    Étapes:
        1. Recherche le logger à utiliser (ordre : kwargs, args, self.<nom_logger>).
        2. Loggue début, fichier, fonction, ligne.
        3. Mesure la durée d’exécution.
        4. Loggue fin, durée.
        5. Retourne le résultat.

    Tips:
        - Pratique pour les méthodes de classes ou scripts à logger injecté.
        - Si aucun logger trouvé, affiche dans la console avec print().

    Utilisation:
        Pour fonctions utilitaires, méthodes d’objets, ou pipelines paramétrables.

    Limitation:
        - Si aucun logger n’est trouvé, print seulement (pas de log dans un fichier/console avancé).
        - Ne supporte qu’un logger par fonction, pas de gestion multi-niveaux.

    See also:
        - log_debut_fin, logging.Logger
    """
    def decorateur(fonction):
        @functools.wraps(fonction)
        def fonction_decoree(*args, **kwargs):
            # Recherche logger dans kwargs, args, puis self.logger
            logger = None
            for valeur in kwargs.values():
                if isinstance(valeur, logging.Logger):
                    logger = valeur
                    break
            else:
                for arg in args:
                    if isinstance(arg, logging.Logger):
                        logger = arg
                        break
                if logger is None and len(args) > 0:
                    self_obj = args[0]
                    if hasattr(self_obj, nom_logger):
                        possible_logger = getattr(self_obj, nom_logger)
                        if isinstance(possible_logger, logging.Logger):
                            logger = possible_logger

            # Récupère nom du fichier et numéro de ligne
            fichier = os.path.basename(fonction.__code__.co_filename)
            numero_ligne = fonction.__code__.co_firstlineno

            message_debut = (
                f"[DEBUT] {fichier} - {fonction.__name__} → ligne {numero_ligne}\n"
            )
            if logger is None:
                print(message_debut)
            else:
                logger.info(message_debut)
            t0 = time.time()
            resultat = fonction(*args, **kwargs)
            duree = time.time() - t0
            message_fin = (
                f"[FIN] {fichier} - {fonction.__name__} → ligne {numero_ligne} "
                f"- durée {duree:.3f}s\n"
            )
            if logger is None:
                print(message_fin)
            else:
                logger.info(message_fin)
            return resultat
        return fonction_decoree
    return decorateur



'''
# ============ Configuration du journal (logging) ============
def configurer_logging(niveau: logging.Logger, nom_logger:str):
    logger = logging.getLogger(nom_logger)
    logger.setLevel(niveau)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    # Supprime tous les handlers existants pour éviter les doublons
    if logger.hasHandlers():
        logger.handlers.clear()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False  # Évite la remontée des logs au root logger

    # Redirige les warnings Python vers le logger
    def avertissement_personnalise(
            message: str, categorie: type, fichier: str, ligne: int, file=None, 
            line=None) -> None:
        """
        Redirige les warnings Python vers le système de logging.

        Args:
            message (str): Message d'avertissement.
            categorie (type): Type de warning.
            fichier (str): Nom du fichier source.
            ligne (int): Ligne dans le fichier source.
            file, line: Argument inutilisés.

        Returns:
            None
        """
        logger.warning(f"{categorie.__name__}: {message} ({fichier}:{ligne})")
    warnings.showwarning = avertissement_personnalise
    return logger
'''