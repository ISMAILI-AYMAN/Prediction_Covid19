# -*- coding: utf-8 -*-
"""
03_savitzky_golay.py
--------------------
Pipeline de traitement, lissage et visualisation interactive des données COVID-19 par commune
pour Bruxelles-Capitale.

Ce script charge les séries temporelles de cas COVID pour chaque commune (données propres JSON),
applique un filtre Savitzky-Golay (lissage), sauvegarde le résultat structuré en JSON et offre
une interface interactive pour explorer la dynamique des cas par commune.

Objectifs pédagogiques :
    - Montrer à un·e débutant·e comment structurer un pipeline scientifique complet (chargement,
      traitement, sauvegarde, visualisation, interface interactive).
    - Présenter l'utilisation de bonnes pratiques : typage fort (PEP 484), logging, modularité,
      séparation des étapes, exceptions gérées.
    - Illustrer la centralisation des paramètres (style, emplacement, options) en haut de fichier
      pour faciliter la maintenance et la personnalisation rapide du projet.
    - Expliquer comment automatiser l’export des résultats lissés, pour un usage dans d’autres
      outils ou analyses.

Architecture et modules :
    - Fonction principale : `main()`, qui orchestre le workflow global.
    - Étapes :
        1. Chargement des données brutes (JSON propre, multi-communes, multi-dates).
        2. Application du filtre de lissage Savitzky-Golay sur chaque commune (paramétrable).
        3. Sauvegarde du résultat lissé dans un JSON daté, sans écrasement accidentel.
        4. Lancement d'une interface graphique interactive :
            * Choix de la commune (radio).
            * Ajustement du filtre (taille fenêtre, degré polynôme).
            * Visualisation dynamique des effets sur la courbe lissée.
    - Toutes les constantes graphiques, labels, emplacements de fichiers sont rassemblées en haut de
      fichier (blocs `STYLE`, `EMPLACEMENT_ENTREE`, etc.) : à modifier ici pour tout le projet.
    - Les fonctions métiers sont compactes, testables et logguent chaque étape clé.
    - Aucun effet de bord caché : tout export (JSON, PNG) est explicitement loggué.

Bonnes pratiques et conseils pro :
    - Le logger est configuré dès le début : chaque anomalie, avertissement, ou étape est
      consignée (facilite le debug pour les débutants).
    - Les exceptions sont capturées et expliquées dans les logs, pour ne jamais “planter
      silencieusement”.
    - La visualisation interactive se lance automatiquement si matplotlib est disponible,
      sinon le script propose une visualisation statique de secours.
    - L’architecture modulaire (fonctions indépendantes) permet la réutilisation dans
      d’autres projets ou notebooks.

Prérequis :
    - Python 3.9+ (pour le typage moderne).
    - Fichier JSON des données COVID propre, structuré par date et commune.
    - Les modules utilitaires du projet (`utils.py`) : chargement, extraction, etc.
    - matplotlib, scipy installés (pour le graphique et le lissage).
    - (Optionnel) : un environnement supportant les interfaces matplotlib (sinon fallback statique).

Limites et extensions :
    - Le script suppose des données complètes et cohérentes (pas de correction automatique
      des erreurs de structure).
    - Le lissage Savitzky-Golay n’est pas pertinent pour des séries ultra-courtes (le script
      le signale explicitement).
    - Le style graphique par défaut est simple : adapter le bloc `STYLE` pour des rendus
      personnalisés ou pro.
    - Possibilité d’intégrer d’autres métriques (incidence, mortalité) en adaptant les
      fonctions utilitaires.
    - Pour traitement batch/automatisé, réutilisez `traitement_principal()` dans un pipeline externe.

Workflow typique :
    1. Lancement du script (`python 03_savitzky_golay.py`)
    2. Vérification des paramètres de lissage (fenêtre, polynôme).
    3. Extraction et lissage des séries temporelles pour toutes les communes.
    4. Sauvegarde du résultat au format JSON (nom daté, sans écrasement).
    5. Affichage de l’interface graphique interactive (choix commune, lissage dynamique).

Pour aller plus loin :
    - Utilisez le JSON de sortie pour créer des dashboards ou rapports automatisés.
    - Intégrez la visualisation interactive dans un notebook Jupyter pour l’analyse exploratoire.
    - Expérimentez d’autres filtres de lissage ou visualisations avancées (voir scipy, plotly, etc.).
    - Proposez une analyse multi-communes (carte, clustering) en enrichissant le projet.

Auteur   : Harry FRANCIS (2025)
Contact  : github.com/documentHarry

License  : Code source ouvert à la consultation.
           Toute redistribution ou usage commercial est soumis à accord écrit préalable.

Version  : 1.0.0 (2025-07-23)
"""


__version__ = "1.0.0"

# Compatible Python 3.9+  | Typage PEP 484 | Type PEP257

# --- Librairies standards
import os
import warnings
import logging

from typing import Any, Callable, Optional

# --- Librairies tiers
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")

# --- Modules locaux
from utils_io import charger_json, sauvegarder_json, creer_dossier_si_absent
from utils_log import log_debut_fin, configurer_logging
from utils_dates import (
    log_stats_filtrage, str_vers_datetime, filtrer_liste_par_date, 
    afficher_periode_liste, suffixe_periode
)
from constantes import (
    EMPLACEMENT_DONNEES_COVID, EMPLACEMENT_DONNEES_LISSEES, EMPLACEMENT_LISSAGE,
    COMMUNES, DonneesCovid, extraire_serie
)
from lissage_savitzky_golay import (
    appliquer_lissage, creer_lissage, extraire_brut_lisse, DonneesLissage
)
from interface_matplotlib import (
    interface_interactive, STYLE, tracer_brut_lisse, tracer_serie
)

# === PARAMÈTRES ET CONSTANTES GLOBALES ===
COMMUNE_DEFAUT: str = "Bruxelles"  # Commune sélectionnée par défaut à l'ouverture

TAILLE_FENETRE_INIT: int = 11      # Taille initiale (impair) de la fenêtre de lissage Savitzky-Golay
DEGRE_POLYNOME_INIT: int = 3       # Degré initial du polynôme de lissage
# Période réellement analysée
PERIODE_MIN:str = "2020-11-06"
PERIODE_MAX:str = "2022-10-02"
# Taille de la fenêtre impaires
fenetres = range(5, 22, 2)   # 5 à 21 inclus par intervalle de 2 (impaires)
# Degré du polynôme
degres = range(1, 8)         # 1 à 7 inclus

#: Niveau du logging (INFO, DEBUG, WARNING...).
NIVEAU_LOG: int = logging.INFO
logger = configurer_logging(NIVEAU_LOG, "03_savitzky_golay")


@log_debut_fin(logger)
def generer_lissage_pour_communes(
        donnees_json: Any, fenetre: int, degre: int, 
        extraire_serie: Callable[[Any, str], tuple[list[str], list[int]]],
        appliquer_lissage: Callable[[list[int], int, int, Optional[Any]], list[float]],
        logger: logging.Logger
        ) -> dict[str, dict[str, Any]]:
    """
    Génère, pour chaque commune, les séries brutes et lissées sur toute la période.

    Args:
        donnees_json (Any): Données brutes COVID structurées par date et commune.
        fenetre (int): Taille de la fenêtre du filtre Savitzky-Golay (impair).
        degre (int): Degré du polynôme pour le lissage.
        extraire_serie (callable): Fonction pour extraire (dates, valeurs) pour une commune.
        appliquer_lissage (callable): Fonction pour lisser les valeurs brutes.

    Returns:
        dict[str, dict[str, Any]]: Un dictionnaire où chaque clé est le nom d’une commune, 
        et la valeur un dict avec :
            - "dates": liste de dates (str)
            - "brut": valeurs brutes (int)
            - "lisse": valeurs lissées (float)

    Note:
        - Chaque commune définie dans LISTE_COMMUNES est traitée même si absente des données.
        - Les fonctions `extraire_serie` et `appliquer_lissage` doivent respecter les signatures.

    Example:
        >>> result = generer_lissage_pour_communes(data, 11, 3, extraire_serie, 
        ...                                        appliquer_lissage)
        >>> print(result["Ixelles"].keys())
        dict_keys(['dates', 'brut', 'lisse'])

    Étapes:
        1. Pour chaque commune de LISTE_COMMUNES :
        2. Extrait les dates et valeurs brutes.
        3. Applique le lissage Savitzky-Golay.
        4. Stocke les résultats dans un dict structuré.

    Tips:
        - Le logger permet de tracer le déroulement, surtout en cas de bug ou de série vide.
        - Adapté pour automatiser le traitement batch de toutes les communes.

    Utilisation:
        À utiliser pour préparer les résultats lissés pour visualisation ou export JSON.

    Limitation:
        - Si les données pour une commune sont absentes, la série peut être vide.
        - Les paramètres fenetre et degre ne sont pas validés ici (voir appelant).
    """
    resultat: dict[str, dict[str, Any]] = {}
    for commune in COMMUNES:
        dates, brut = extraire_serie(donnees_json, commune, logger)
        lisse = appliquer_lissage(brut, fenetre, degre, logger)
        resultat[commune] = {"dates": dates, "brut": brut, "lisse": lisse}
    return resultat


@log_debut_fin(logger)
def exporter_lissages(
        donnees_json: Any, communes: list[str], fenetres: list[int], degres: list[int],
        extraire_serie: Callable[[Any, str], tuple[list[str], list[int],
                                                   logger: logging.Logger]],
        appliquer_lissage: Callable[[list[int], int, int, Optional[Any]], list[float]],
        organisation: str = "par_degre_fenetre", dossier_base: str = "",
        periode: Optional[tuple[Optional[str], Optional[str]]] = None,
        logger: Optional[Any] = None) -> None:
    """
    Exporte en JSON toutes les combinaisons de lissage sur les communes sélectionnées.

    Args:
        donnees_json (Any): Données brutes COVID (multi-dates, multi-communes).
        communes (list[str]): Liste des communes à traiter.
        fenetres (list[int]): Tailles de fenêtres (impaires) pour le lissage.
        degres (list[int]): Degrés de polynôme à tester.
        extraire_serie (callable): Fonction pour extraire les séries brutes par commune.
        appliquer_lissage (callable): Fonction pour lisser chaque série.
        organisation (str): "par_degre_fenetre" ou "par_commune" (arborescence de fichiers).
        dossier_base (str): Dossier racine où exporter les JSON lissés.

    Returns:
        None

    Note:
        - Ne crée pas de doublon : chaque export porte le nom de la config utilisée.
        - Les dossiers sont créés automatiquement si absents.

    Example:
        >>> exporter_lissages(data, ["Ixelles"], [11], [3], extraire_serie, appliquer_lissage)

    Étapes:
        1. Crée le dossier de base si besoin.
        2. Boucle sur chaque combinaison fenêtre/degré.
        3. Pour chaque commune, extrait, lisse, sauvegarde un JSON.

    Tips:
        - Pour exporter tous les résultats d’un coup pour du machine learning, reporting, etc.
        - Les noms de fichiers suivent le format standard pour l’automatisation.

    Utilisation:
        Utilisé après le traitement principal pour rendre persistants tous les résultats.

    Limitation:
        - Les paramètres fenetres et degres ne sont pas validés ici.
        - Peut écraser les anciens exports si `ecrasement=True`.
    """
    suffixe_de_la_periode = suffixe_periode(periode)
    creer_dossier_si_absent(dossier_base, logger)
    if organisation == "par_degre_fenetre":
        for fenetre in fenetres:
            for degre in degres:
                if degre >= fenetre: continue
                data = generer_lissage_pour_communes(
                    donnees_json, fenetre, degre, extraire_serie, 
                    appliquer_lissage, logger)
                nom_fichier = (
                    f"lissages_fenetre{fenetre}_degre{degre}"
                    f"{suffixe_de_la_periode}.json")
                chemin = os.path.join(dossier_base, nom_fichier)
                sauvegarder_json(data, chemin, ecrasement = True, ensure_ascii = False,
                                 logger = logger)
                if logger: logger.info(f"Sauvegarde : {chemin}")
    elif organisation == "par_commune":
        for commune in COMMUNES:
            dossier_commune = os.path.join(dossier_base, commune)
            creer_dossier_si_absent(dossier_commune, logger)
            for fenetre in fenetres:
                for degre in degres:
                    if degre >= fenetre: continue
                    lissage = generer_lissage_pour_communes(
                        donnees_json, fenetre, degre, extraire_serie,
                        appliquer_lissage, logger)
                    data_export = {
                        "commune": commune, "fenetre": fenetre, "degre": degre,
                        **lissage[commune]
                    }
                    nom_fichier = (
                        f"{commune}_fenetre{fenetre}_degre{degre}"
                        f"{suffixe_de_la_periode}.json")
                    chemin = os.path.join(dossier_commune, nom_fichier)
                    sauvegarder_json(data_export, chemin, ecrasement=True,
                                     ensure_ascii=False, logger = logger)
                    if logger: logger.info(f"Sauvegarde : {chemin}")


@log_debut_fin(logger)
def traitement_principal(
        date_debut: Optional[str] = None, date_fin: Optional[str] = None
        ) -> tuple[DonneesCovid, DonneesLissage, str, 
                tuple[Optional[str], Optional[str]]]:
    """
    Pipeline complet : charge, filtre, lisse et exporte les données pour toutes les communes.

    Args:
        date_debut (str, optionnel): Date de début (incluse), format "YYYY-MM-DD".
        date_fin (str, optionnel): Date de fin (incluse), format "YYYY-MM-DD".

    Returns:
        tuple:
            DonneesCovid: Données COVID brutes, éventuellement filtrées.
            DonneesLissage: Données lissées structurées.
            str: Chemin du fichier JSON exporté (résultat lissé).
            tuple[str, str]: Période analysée (min, max), ou (None, None) si vide.

    Raises:
        ValueError: Si TAILLE_FENETRE_INIT est paire (erreur d’utilisation).

    Note:
        - Fonction centrale du pipeline : à appeler pour tout traitement.
        - Loggue tous les points clés et anomalies.
        - S’appuie sur config globale pour les chemins, paramètres, logger.

    Example:
        >>> d, liss, path, per = traitement_principal("2022-01-01", "2022-01-31")

    Étapes:
        1. Vérifie que la fenêtre de lissage est impaire.
        2. Charge le JSON des données COVID brutes.
        3. Filtre éventuellement sur la période demandée.
        4. Applique le lissage Savitzky-Golay.
        5. Sauvegarde le résultat lissé dans un JSON daté.
        6. Retourne les quatre objets principaux.

    Tips:
        - Peut servir pour l’automatisation, les notebooks, ou la CLI.

    Utilisation:
        À utiliser en point d’entrée pour tout workflow complet, ou dans un batch.

    Limitation:
        - Suppose un JSON de données propre (structure, complétude non contrôlées ici).
    """
    if TAILLE_FENETRE_INIT % 2 == 0:
        logger.error(
            "La taille de la fenêtre doit être impaire pour le filtre Savitzky-Golay.",
            exc_info = True)
        raise ValueError(
            "La taille de la fenêtre doit être impaire pour le filtre Savitzky-Golay.")

    donnees_json: DonneesCovid = charger_json(EMPLACEMENT_DONNEES_COVID, logger)
    try:
        # Filtrage de la période si demandé
        if date_debut or date_fin:
            # On transforme le dict de dict en liste de dicts pour filtrage
            liste  = [ { "DATE": d, **donnees_json[d] }
                                       for d in donnees_json ]
            liste_filtree, stats_filtrage = filtrer_liste_par_date(
                liste, logger = logger, date_debut = date_debut, date_fin = date_fin)
            # Log les statistiques sur les données filtrées
            log_stats_filtrage(logger, stats_filtrage)
            if (date_debut and date_fin) and date_debut > date_fin:
                logger.warning(f"Période invalide : date_debut ({date_debut}) > "
                               f"date_fin ({date_fin})")
            elif liste_filtree is not None and len(liste_filtree) == 0:
                logger.warning("Aucune donnée ne correspond à la période demandée : "
                               f"JSON filtré vide.")
            afficher_periode_liste(liste_filtree, logger = logger, cle_date = "DATE")
            if liste_filtree:
                dates = [d["DATE"] for d in liste_filtree]
                periode_min, periode_max = min(dates), max(dates)
                # Reconstruction du dict par date comme à l’origine :
                donnees_json = {
                    d["DATE"]: {k: v for k, v in d.items() if k != "DATE"}
                    for d in liste_filtree
                }
                logger.info(f"Période analysée : {periode_min} → {periode_max}")
            else:
                periode_min, periode_max = None, None
                donnees_json = {}  # ou garde donnees_json vide, up to you
            logger.info(f"{len(donnees_json)} jours après filtrage "
                        f"{date_debut}→{date_fin}")
        else:
            # Période sur tout le jeu de données
            if donnees_json:
                dates = sorted(donnees_json.keys())
                periode_min, periode_max = dates[0], dates[-1]
            else:
                periode_min, periode_max = None, None
                # Pas de reconstruction de donnees_json ici ! On laisse tel quel.
    except Exception as exception:
        logger.error(f"Erreur inattendue lors du filtrage : {exception}", 
                     exc_info = True)
        return {}, {}, "", (None, None)  # ou comportement par défaut

    donnees_lissees: DonneesLissage = creer_lissage(
        donnees_json, COMMUNES, TAILLE_FENETRE_INIT, DEGRE_POLYNOME_INIT,
        logger
    )
    # --- Création du dossier cible si besoin (robuste) ---
    dir_cible = os.path.dirname(EMPLACEMENT_DONNEES_LISSEES)
    if dir_cible:
        creer_dossier_si_absent(dir_cible, logger)

    # Ajout d'un suffixe période si applicable
    suffixe_periode = ""
    if date_debut or date_fin:
        # Noms propres et robustes même si une borne est None
        s_debut = date_debut or "xxxx-xx-xx"
        s_fin = date_fin or "xxxx-xx-xx"
        suffixe_periode = f"_{s_debut}_to_{s_fin}"

    # On insère le suffixe avant le .json
    base, ext = os.path.splitext(EMPLACEMENT_DONNEES_LISSEES)
    chemin_sortie = f"{base}{suffixe_periode}{ext}"

    emplacement_final:str = sauvegarder_json(
        donnees_lissees, chemin_sortie, ecrasement = True,
        logger = logger
    )
    logger.info(f"Données lissées sauvegardées dans {emplacement_final}")
    return donnees_json, donnees_lissees, emplacement_final, (periode_min, periode_max)


@log_debut_fin(logger)
def main_interactif(
        donnees_json: DonneesCovid, donnees_lissees: DonneesLissage,
        periode: tuple[Optional[str], Optional[str]] = None) -> None:
    """
    Lance la visualisation interactive. Bascule en mode statique si l’interface échoue.

    Args:
        donnees_json (DonneesCovid): Données brutes.
        donnees_lissees (DonneesLissage): Données déjà lissées.
        periode (tuple[str, str], optionnel): Période affichée (début, fin).
            Si None, la période complète est affichée, sans annotation particulière.

    Returns:
        None

    Note:
        - Si matplotlib ou l’interface échoue, affiche la commune par défaut en statique.
        - Loggue toutes les erreurs d’affichage pour debug.

    Example:
        >>> main_interactif(data, data_liss, ("2022-01-01", "2022-01-31"))

    Étapes:
        1. Tente de lancer l’interface interactive complète.
        2. Si erreur, passe en fallback statique sur la commune par défaut.

    Tips:
        - Utile pour garantir une UX même sur serveur distant ou backend incompatible.

    Utilisation:
        À enchaîner après traitement_principal() pour la visualisation.

    Limitation:
        - Affichage statique limité : pas de sélection interactive possible en fallback.
    """
    try:
        interface_interactive(
            donnees_json, COMMUNES, periode, logger, COMMUNE_DEFAUT,
            TAILLE_FENETRE_INIT, DEGRE_POLYNOME_INIT, str_vers_datetime, 
            extraire_serie, appliquer_lissage, tracer_brut_lisse, STYLE
        )
    except Exception as err:
        logger.exception("Affichage interactif non disponible")
        if COMMUNE_DEFAUT in COMMUNES:
            dates, brut, lisse = extraire_brut_lisse(
                donnees_json, donnees_lissees, COMMUNE_DEFAUT, logger)
            tracer_serie(dates, brut, lisse, COMMUNE_DEFAUT, logger, periode)


@log_debut_fin(logger)
def main() -> None:
    """
    Point d’entrée du script. Lance le traitement principal puis la visualisation.

    Étapes:
        1. Appelle traitement_principal() pour charger, filtrer, lisser et exporter.
        2. Appelle main_interactif() pour la visualisation (interactive ou statique).
        3. Exporte tous les lissages configurés pour toutes les communes.

    Returns:
        None

    Note:
        - Toute erreur fatale est logguée, jamais affichée brutalement à l’utilisateur.
        - Peut être appelée en CLI ou importée dans un notebook/test.

    Example:
        >>> main()

    Tips:
        - À placer tout en bas du script pour exécuter tout le pipeline.
        - Compatible usage CLI, batch ou importation modulaire.

    Utilisation:
        Fonction de lancement, à appeler en toute fin de fichier.

    Limitation:
        - Ne retourne rien, contrôle le pipeline global.
    """
    try:
        '''
        (donnees_json, donnees_lissees, emplacement_final,
            periode) = traitement_principal()
        main_interactif(donnees_json, donnees_lissees, periode)        
        exporter_lissages(
            donnees_json, COMMUNES, fenetres, degres, extraire_serie, 
            appliquer_lissage, organisation = "par_degre_fenetre",
            dossier_base = EMPLACEMENT_LISSAGE,
            logger = logger)
        '''
        
        (donnees_json, donnees_lissees, emplacement_final,
            periode) = traitement_principal(PERIODE_MIN, PERIODE_MAX)
        main_interactif(donnees_json, donnees_lissees, periode)
        exporter_lissages(
            donnees_json, COMMUNES, fenetres, degres, extraire_serie,
            appliquer_lissage, organisation = "par_degre_fenetre",
            dossier_base = EMPLACEMENT_LISSAGE, periode = (PERIODE_MIN, PERIODE_MAX),
            logger = logger)
        
        
    except Exception as exception:
        logger.critical(f"Erreur fatale : {exception}", exc_info = True)


if __name__ == "__main__":
    # Pour tout lisser :
    main()


