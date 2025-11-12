# -*- coding: utf-8 -*-

"""
04_covid_visualizer.py
----------------------
Pipeline complet de traitement, lissage et visualisation des données COVID-19 par commune
pour Bruxelles-Capitale.

Ce script centralise toute la logique d’analyse et d’export graphique du projet, sans duplication
de code utilitaire : il orchestre les modules annexes (`utils`, `savitzky_golay_03`, etc.)
et applique les bonnes pratiques de code modulaire, typé, et réutilisable.

Fonctionnalités principales :
    - Chargement des données brutes COVID depuis un fichier JSON propre.
    - Application du filtre de lissage Savitzky-Golay (réduction du bruit, mise en valeur des tendances).
    - Génération de graphiques clairs et homogènes par commune (avec visualisation brute+lissée).
    - Export des graphiques en PNG (par lot, un par page, nommés automatiquement).
    - Sauvegarde des résultats lissés au format JSON avec timestamp (aucun écrasement accidentel).
    - Logging détaillé de toutes les étapes et des éventuelles erreurs critiques.

Architecture :
    - Une seule classe principale, `CovidVisualizer`, qui :
        * encapsule toutes les données et le workflow ;
        * rend le pipeline réutilisable (un seul appel pour tout traiter) ;
        * expose chaque étape (lissage, visualisation, export) via des méthodes dédiées.
    - Les modules utilitaires (`utils.py`, `savitzky_golay_03.py`) gèrent le détail technique
      (IO, calcul, extraction des séries…) pour éviter la duplication ici.

Objectif pédagogique :
    - Montrer à un·e débutant·e comment structurer un vrai programme de visualisation scientifique
      (architecture classe, typage, pipeline, gestion d’erreur, logging, IO propres, etc.).
    - Encourager la réutilisabilité et l’automatisation : tout le workflow se lance en une ligne
      (`visualizer.pipeline()`).
    - Apprendre la séparation des responsabilités (classe = orchestration, modules = outils).

Prérequis :
    - Python 3.9+ (pour typage fort et syntaxe moderne).
    - Données COVID nettoyées au format JSON, avec la bonne structure.
    - Modules utilitaires du projet (chargement, sauvegarde, lissage, etc.).
    - matplotlib installé pour la génération graphique.
    - Un logger configuré pour suivre l’exécution et diagnostiquer les erreurs.

Conseils pro :
    - Ne modifiez jamais le code utilitaire depuis ce script : importez, appelez, mais
      gardez la logique de bas niveau dans les modules spécialisés.
    - Les styles graphiques, libellés et dossiers sont centralisés pour modifier l’apparence ou
      la destination des fichiers en un seul endroit.
    - Adaptez la taille des figures, la disposition ou les styles en changeant les constantes
      du bloc dédié (au début du fichier).

Workflow (pipeline) complet :
    1. Charger les données brutes.
    2. Appliquer le lissage Savitzky-Golay.
    3. Sauvegarder le résultat lissé (JSON, daté).
    4. Générer et sauvegarder les graphiques bruts+lissés (par page, par commune).

Limitation :
    - Les figures générées sont “prêtes à l’emploi” mais restent dépendantes du style
      matplotlib par défaut (adapter la classe Style si besoin).
    - Le script suppose que les données sont propres et complètes (pas de correction automatique).
    - Le pipeline ne gère pas d’analyse “aggrégée” (total, moyenne), uniquement par commune.

Pour aller plus loin :
    - Intégrez ce script dans un pipeline de reporting automatisé ou un notebook pour
      des analyses exploratoires avancées.
    - Ajoutez des métriques ou visualisations supplémentaires (carte, histogramme, etc.)
      en enrichissant les modules utilitaires.

Auteur   : Harry FRANCIS (2025)
Contact  : github.com/documentHarry

License  : Code source ouvert à la consultation.
           Toute redistribution ou usage commercial est soumis à accord écrit préalable.

Version  : 1.0.0 (2025-07-23)
"""

__version__ = "1.0.0"

# Compatible Python 3.9+ | Typage PEP 484 | PEP 257 | PEP 8 (88 caractères max)

import logging
import os
from typing import Optional
from datetime import datetime

# --- Librairies tiers
import matplotlib.pyplot as plt

# --- Modules locaux
from utils_io import charger_json, sauvegarder_json
from utils_log import configurer_logging, log_debut_fin
from constantes import COMMUNES, DonneesCovid
from lissage_savitzky_golay import DonneesLissage, creer_lissage
from matplotlib_renderer import *
# ========================= CONSTANTES GRAPHIQUES & LIBELLÉS =========================
"""
Constantes de configuration graphique et de libellés pour la visualisation COVID.
À centraliser dans ce bloc pour une modification rapide du style général.
"""

# --- Dossier et suffixes ---
DOSSIER_SAUVEGARDE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "visualizations")
)  # Dossier par défaut de sauvegarde des graphes

#: Niveau du logging (INFO, DEBUG, WARNING...).
NIVEAU_LOG: int = logging.INFO
logger: logging.Logger = configurer_logging(NIVEAU_LOG, "04_covid_visualizer")

class CovidVisualizer:
    """
    Classe permettant de traiter, lisser et visualiser les données COVID-19 par commune.

    Cette classe centralise l'orchestration du pipeline de traitement : application du lissage
    Savitzky-Golay aux séries temporelles, sauvegarde des résultats en JSON,
    et génération automatique de graphiques par lot pour chaque commune.

    Args:
        donnees_brutes (DonneesCovid): Dictionnaire des données COVID brutes {date: {commune: valeur}}.
        communes (List[str], optional): Liste des communes à traiter (défaut : toutes les communes de Bxl).

    Attributes :
        donnees_brutes (DonneesCovid): Données originales.
        communes (List[str]): Communes traitées.
        donnees_lissees (DonneesLissage): Résultat du lissage, rempli après appel à .lisser().

    Raises:
        ValueError: Si aucune donnée brute n’est fournie lors de l’instanciation.

    Example:
        >>> donnees = charger_json("Data/C0VID19BE_CASES_MUNI_CLEAN.json", logger)
        >>> visualizer = CovidVisualizer(donnees)
        >>> visualizer.pipeline(taille_fenetre=9, degre_polynome=2, communes_par_page=3)

    Note:
        Cette classe dépend fortement des fonctions utilitaires du projet (voir utils.py et savitzky_golay_03.py).
        Les figures générées sont sauvegardées dans le dossier spécifié, au format PNG.
    """


    @log_debut_fin(logger)
    def __init__(
            self, donnees_brutes: Optional[DonneesCovid] = None, 
            communes: Optional[list[str]] = None):
        if donnees_brutes is None:
            raise ValueError("Il faut fournir les données brutes COVID-19.")
        self.donnees_brutes = donnees_brutes
        self.communes = communes or COMMUNES
        self.donnees_lissees: Optional[DonneesLissage] = None


    @log_debut_fin(logger)
    def lisser(self, taille_fenetre: int = 11, degre_polynome: int = 3) -> None:
        """
        Calcule les données lissées pour chaque commune de la liste courante,
        en utilisant le filtre Savitzky-Golay via les fonctions utilitaires du projet.

        Args:
            taille_fenetre (int, optional): Taille de la fenêtre du filtre Savitzky-Golay
                (doit être impaire). Défaut : 11.
            degre_polynome (int, optional): Degré du polynôme du filtre. Défaut : 3.

        Returns:
            None

        Example:
            >>> visualizer = CovidVisualizer(donnees)
            >>> visualizer.lisser(taille_fenetre=9, degre_polynome=2)

        Note:
            Les résultats sont stockés dans l’attribut `donnees_lissees` et écrasent toute valeur précédente.
            Les données brutes doivent avoir été fournies à l’instanciation de l’objet.
        """
        self.donnees_lissees = creer_lissage(
            self.donnees_brutes, self.communes, taille_fenetre, degre_polynome, logger
        )
        logger.info(f"Lissage calculé pour {len(self.communes)} communes.")


    @log_debut_fin(logger)
    def visualiser_par_page(
            self, communes_par_page: int = 6,
            dossier_de_sauvegarde: str = DOSSIER_SAUVEGARDE) -> None:
        """
        Génère et sauvegarde les graphiques bruts/lissés (un subplot par commune, par page).

        Args:
            communes_par_page (int): Nombre de communes par page/figure (défaut : 6).
            dossier_de_sauvegarde (str): Dossier où sauvegarder les images PNG.

        Returns:
            None

        Raises:
            RuntimeError: Si les données lissées ne sont pas encore calculées.

        Example:
            >>> visualizer.lisser()
            >>> visualizer.visualiser_par_page(communes_par_page=4)

        Note:
            Les images générées sont nommées automatiquement selon les communes affichées.
        """
        if self.donnees_lissees is None:
            raise RuntimeError("Il faut d'abord lancer .lisser() avant de visualiser.")
        os.makedirs(dossier_de_sauvegarde, exist_ok=True)

        nbr_communes = len(self.communes)
        nbr_pages = (nbr_communes + communes_par_page - 1) // communes_par_page

        for numero_page in range(nbr_pages):
            index_initial = numero_page * communes_par_page
            index_final = min(index_initial + communes_par_page, nbr_communes)
            noms_des_communes = self.communes[index_initial:index_final]

            fig, axes = creer_figure(numero_page, nbr_pages, noms_des_communes)
            axes = axes.flatten()

            for index_commune, commune in enumerate(noms_des_communes):
                tracer_commune_sur_axe(
                    axes[index_commune], commune, self.donnees_brutes,
                    self.donnees_lissees, logger
                )

            supprimer_axes_vides(axes, len(noms_des_communes))
            ajouter_legende(fig)
            sauvegarder_figure(
                fig, dossier_de_sauvegarde, numero_page + 1, noms_des_communes, logger
            )
            plt.show()
            plt.close(fig)


    @log_debut_fin(logger)
    def sauvegarder_lissage(
            self, dossier="exports_json", prefixe="lissage_covid") -> str:
        """
        Sauvegarde les données lissées en JSON dans un dossier.

        Args:
            dossier (str): Dossier de sauvegarde.
            prefixe (str): Préfixe du nom du fichier.

        Returns:
            str: Emplacement réel du fichier JSON créé.

        Raises:
            RuntimeError: Si les données lissées ne sont pas encore calculées.

        Example:
            >>> path = self.sauvegarder_lissage()

        Note:
            Un timestamp est ajouté au nom pour éviter l’écrasement.
        """
        if self.donnees_lissees is None:
            raise RuntimeError(
                "Aucune donnée lissée à sauvegarder. Lancez .lisser() d'abord."
            )
        os.makedirs(dossier, exist_ok=True)
        # Compose le nom avec un timestamp (optionnel) pour éviter l’écrasement
        date_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        emplacement = os.path.join(dossier, f"{prefixe}_{date_str}.json")
        return sauvegarder_json(self.donnees_lissees, emplacement, logger = logger)


    @log_debut_fin(logger)
    def pipeline(
            self, taille_fenetre: int = 11, degre_polynome: int = 3, 
            communes_par_page: int = 6 ) -> None:
        """
        Orchestration complète du workflow :
        - Calcule les données lissées,
        - Sauvegarde le JSON des données lissées,
        - Génére et sauvegarde les graphiques par page.

        Args:
            taille_fenetre (int, optional): Fenêtre du filtre Savitzky-Golay. Défaut 11.
            degre_polynome (int, optional): Degré du polynôme de lissage. Défaut 3.
            communes_par_page (int, optional): Nombre de communes par page de graphique. Défaut 6.

        Returns:
            None

        Example:
            visualizer.pipeline(taille_fenetre=9, degre_polynome=2, communes_par_page=3)
        """
        self.lisser(taille_fenetre, degre_polynome)
        self.sauvegarder_lissage()
        self.visualiser_par_page(communes_par_page)


@log_debut_fin(logger)
def main() -> None:
    """
    Point d’entrée principal du script.

    Orchestration du workflow complet :
        - Chargement des données brutes COVID.
        - Instanciation de la classe CovidVisualizer.
        - Lancement du pipeline (lissage, sauvegarde, visualisation).

    Gère également la capture et la journalisation des exceptions critiques.

    Returns:
        None

    Raises:
        Exception: Toute erreur inattendue lors de l’exécution du pipeline.

    Example:
        (Appel implicite lors de l’exécution du script : python covid_visualizer_04.py)
    """
    # ========================= MAIN =========================
    try:
        # Construction du chemin absolu vers le fichier de données nettoyées
        EMPLACEMENT_DATA = os.path.abspath(
            os.path.join("Data", "C0VID19BE_CASES_MUNI_CLEAN.json")
        )

        # Chargement des données COVID nettoyées au format dictionnaire
        donnees = charger_json(EMPLACEMENT_DATA, logger)

        # Instanciation de la classe de visualisation/lissage avec les données chargées
        visualizer = CovidVisualizer(donnees)

        # Lancement du pipeline complet : lissage, sauvegarde JSON, génération des graphiques
        # (Par défaut : taille fenêtre 11, degré 3, 6 communes par page, dossier 'visualizations/')
        visualizer.pipeline()
    except Exception as exception:
        logger.exception(
            "Erreur critique lors de l'exécution du script : %s", exception
        )


if __name__ == "__main__":
    main()
