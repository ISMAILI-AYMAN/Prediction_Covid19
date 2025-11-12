# -*- coding: utf-8 -*-

"""
matplotlib_renderer.py
----------------------
Outils graphiques pour la visualisation multi-communes des données COVID-19 par matplotlib.

Ce module centralise toutes les fonctions utilitaires de création, habillage et sauvegarde
de figures matplotlib pour la visualisation comparative des cas COVID-19 par commune à
Bruxelles-Capitale.

Objectifs pédagogiques et structure :
    - Montrer comment organiser proprement le code de visualisation dans un projet scientifique.
    - Séparer clairement le “back-end graphique” des autres traitements pour :
        * Réutiliser les fonctions dans d’autres scripts, notebooks ou pipelines automatiques.
        * Modifier le style global d’un seul endroit (voir module `style.py`).
    - Expliquer, pour un·e débutant·e, l’intérêt de la modularité :
        * Chaque étape du rendu est découpée : création de figure, tracé, légende, export.
        * Toutes les constantes graphiques sont paramétrées ailleurs (voir `Style`).
        * Pas d’effet de bord caché, pas d’état global parasite.

Principales fonctionnalités :
    - Création d’une figure multi-axes (grille) pour plusieurs communes (fonction `creer_figure`).
    - Tracé sur chaque sous-axe de la série brute et de la série lissée d’une commune
      (fonction `tracer_commune_sur_axe`).
    - Gestion intelligente des axes vides sur les dernières pages (fonction `supprimer_axes_vides`).
    - Ajout d’une légende “globale” à la figure (fonction `ajouter_legende`).
    - Export propre de la figure au format PNG, nommée par page et liste des communes
      (fonction `sauvegarder_figure`).

Workflow d’utilisation conseillé :
    1. Créer la figure multi-axes en passant la page, le nombre de pages et la liste des communes.
    2. Boucler sur chaque commune et utiliser `tracer_commune_sur_axe` pour remplir chaque sous-axe.
    3. Appeler `supprimer_axes_vides` si certaines cases de la grille ne sont pas utilisées
       (pour éviter les sous-graphiques “blancs”).
    4. Ajouter la légende globale avec `ajouter_legende`.
    5. Sauvegarder la figure avec `sauvegarder_figure`.
    6. Toujours fermer la figure avec `plt.close(fig)` pour libérer la mémoire (bon réflexe pro !).

Bonnes pratiques et pédagogie :
    - Tous les paramètres de style (couleur, police, taille, légende…) sont **centralisés** dans le module
      `style.py` (classe `Style`). Pour changer l’apparence, modifiez une seule fois dans ce fichier.
    - Le module ne dépend d’aucune variable globale : tout est explicitement passé en argument.
    - La documentation de chaque fonction explique le rôle, les “étapes internes”, les limitations et les astuces.
    - Chaque fonction suit le même modèle pédagogique :
        * “Args”, “Returns”, “Raises” (si pertinent), “Note”, “Étapes”, “Tips”, “Utilisation”, “Limitation”,
          “See also”.
    - Toutes les erreurs, avertissements et informations de suivi sont gérés par un logger
      passé explicitement à chaque fonction concernée.
      Adaptez la destination/log level du logger via le module `utils`.
    - Toutes les figures sont sauvegardées en PNG de haute qualité (dpi=200, bbox_inches="tight").

Note importante :
    - Toutes les informations, résumés, alertes ou erreurs sont transmises via le logger
      configuré (niveau INFO, DEBUG, etc.).
    - Tout est loggé pour permettre une meilleure traçabilité et un filtrage ultérieur.      

Pour aller plus loin :
    - Ce module peut servir de base à n’importe quel projet scientifique “multi-plots”, pas seulement COVID.
    - Pour intégrer de nouveaux styles, étendez le module `style.py` ou surchargez la classe `Style`.
    - Intégrez-le dans des pipelines automatisés pour générer des rapports PDF, des dashboards, etc.

Auteur  : Harry FRANCIS (2025)
Contact : github.com/documentHarry

Version : 1.0.0 (2025-07-23)
License : Code source ouvert à la consultation.
          Toute redistribution ou usage commercial est soumis à accord écrit préalable.
"""


__version__ = "1.0.0"

# --- Librairies standards
import logging
import os

# --- Librairies tiers
import numpy as np
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# --- Modules locaux
from lissage_savitzky_golay import extraire_brut_lisse, DonneesLissage
from style import Style
from utils_log import log_debut_fin_logger_dynamique
from utils_dates import dates_str_vers_obj
from constantes import DonneesCovid

@log_debut_fin_logger_dynamique("logger")
def creer_figure(
        index_page: int, nbr_pages: int, noms_des_communes: list[str]
        ) -> tuple[Figure, np.ndarray]:
    """
    Crée une figure matplotlib avec une grille de sous-graphes (axes), prête à recevoir les données
    de plusieurs communes.

    Args:
        index_page (int): Numéro de la page courante (commence à 0).
        nbr_pages (int): Nombre total de pages (pour le titre).
        noms_des_communes (list[str]): Liste des communes à afficher sur cette page.

    Returns:
        tuple[Figure, np.ndarray]:
            - Figure matplotlib créée (objet principal, sert à gérer l’ensemble du dessin).
            - Tableau numpy d’axes (`matplotlib.axes.Axes`), chaque case est un sous-graphe.

    Attributes:
        - Utilise les constantes de style (`Style`) pour la taille de la figure, le nombre de lignes/
          colonnes, etc.
        - Le titre général et la liste des communes sont ajoutés en haut de la figure.

    Note:
        Le nombre de sous-graphes (axes) dépend des paramètres globaux (`Style.GRAPHE_LIGNES`,
        `Style.GRAPHE_COLONNES`), et **non** du nombre de communes passées à la fonction.
        Les axes sont vides à la sortie : il faut ensuite tracer les courbes avec `tracer_commune_sur_axe`.

    Example:
        >>> fig, axes = creer_figure(0, 4, ["Ixelles", "Bruxelles"])
        >>> isinstance(axes, np.ndarray)
        True
        >>> axes.shape
        (2, 3)

    Étapes:
        1. Crée une figure vide avec la taille définie dans `Style.FIGSIZE`.
        2. Ajoute autant de sous-graphes que défini par `Style.GRAPHE_LIGNES` et `COLONNES`.
        3. Prépare le titre général et la liste des communes en haut.
        4. Retourne la figure et la grille d’axes.

    Tips:
        - Fermez toujours la figure après utilisation avec `plt.close(fig)` pour libérer la mémoire.
        - La disposition des axes est contrôlée uniquement par le style global.
        - Vous pouvez modifier la disposition en changeant `Style.GRAPHE_LIGNES`/`COLONNES`.

    Utilisation:
        À utiliser en tout début de page de rapport ou de séquence d’export matplotlib, avant
        d’ajouter des courbes sur les axes.

    Limitation:
        - La grille est de taille fixe : il peut y avoir des axes vides si moins de communes que de
          cases.
        - Pas d’ajustement automatique du nombre d’axes selon le nombre de communes.

    See also:
        - tracer_commune_sur_axe (pour remplir chaque axe)
        - sauvegarder_figure, ajouter_legende
        - Style (paramétrage global)
    """
    fig, axes = plt.subplots(   nrows = Style.GRAPHE_LIGNES, 
                                ncols = Style.GRAPHE_COLONNES,
                                figsize = Style.FIGSIZE)
    # Ajout de la liste des communes en haut
    noms_affiches = [
        c.replace(Style.SUFFIXE_COMMUNE_BRUXELLES, "")
            .strip() for c in noms_des_communes
    ]
    texte_communes = " | ".join(noms_affiches)
    titre = Style.TITRE_GRAPHE.format(texte_communes, index_page + 1, nbr_pages)
    fig.suptitle(titre, fontsize = Style.FONTSIZE)
    '''
    fig.text(
        x = Style.XPOSITION, y = Style.YPOSITION, s = texte_communes, 
        ha = Style.ALIGNEMENT_HORIZONTAL, va = Style.ALIGNEMENT_VERTICAL, 
        fontsize = Style.LIBELLE_FONTSIZE,
        color = Style.COULEUR_TEXTE, wrap = True
    )
    '''
    return fig, axes


@log_debut_fin_logger_dynamique("logger")
def tracer_commune_sur_axe(
        ax: plt.Axes, commune: str, donnees_brutes: DonneesCovid, 
        donnees_lissees: DonneesLissage, logger: logging.Logger) -> None:
    """
    Trace sur un axe matplotlib la série brute et la série lissée d’une commune donnée.

    Args:
        ax (plt.Axes): Axe matplotlib à utiliser pour le tracé.
        commune (str): Nom de la commune à afficher (doit exister dans les données).
        donnees_brutes (DonneesCovid): Données brutes par date et commune.
        donnees_lissees (DonneesLissage): Données lissées (par ex. Savitzky-Golay).
        logger (logging.Logger): Logger utilisé pour signaler toute erreur ou information.

    Returns:
        None

    Raises:
        ValueError: Si la commune n’existe pas dans les données fournies.

    Attributes:
        - N’utilise que les styles centralisés (Style).
        - Ne modifie rien d’autre que l’axe passé.

    Note:
        Fonction “pure” : elle ne touche qu’à l’axe en question.
        Toute erreur (ex : commune absente) est transmise uniquement via le logger.
        Peut être utilisée dans une boucle pour générer de multiples sous-graphiques.

    Théorie:
        - Le lissage Savitzky-Golay permet de voir la tendance générale tout en gardant les pics.
        - L’affichage simultané brut/lissé permet de comparer la variabilité quotidienne à la tendance.
        - Matplotlib permet la superposition de courbes, facilitant la comparaison visuelle.

    Example:
        >>> tracer_commune_sur_axe(ax, "Ixelles", donnees_brutes, donnees_lissees)

    Étapes:
        1. Extrait la série temporelle brute et lissée pour la commune.
        2. Transforme les dates au format attendu par matplotlib.
        3. Trace la courbe brute (style : orange, alpha, épaisseur…).
        4. Trace la courbe lissée (style : bleu, épaisseur…).
        5. Met à jour le titre, les labels, la grille et le format de l’axe X.

    Tips:
        - Peut être appelée plusieurs fois pour remplir une grille d’axes.
        - Gère la rotation automatique des labels de date.
        - Pour changer le style, modifiez le module `style.py`.

    Utilisation:
        À utiliser juste après la création des axes, pour remplir chaque case de la grille.

    Limitations:
        - Trace toujours par-dessus l’axe : ne “vide” pas l’axe si déjà utilisé.
        - Ne traite pas les axes partagés ou synchronisés.

    See also:
        - extraire_brut_lisse
        - creer_figure, sauvegarder_figure
        - Style
    """
    try:
        dates, brut, lisse = extraire_brut_lisse(
            donnees_brutes, donnees_lissees, commune, logger)
    except KeyError as err:
        logger.error(f"Commune '{commune}' non trouvée dans les données.", 
                     exc_info=True)
        raise ValueError(f"Commune '{commune}' non trouvée dans les données.") from err
    dates_obj = dates_str_vers_obj(dates, logger)
    ax.plot(
        dates_obj, brut, label = Style.LEGENDE_BRUT, color = Style.COULEUR_BRUT,
        alpha = Style.ALPHA_BRUT, linewidth = Style.EPAISSEUR_BRUT
    )
    ax.plot(
        dates_obj, lisse, label = Style.LEGENDE_LISSE, color = Style.COULEUR_LISSE,
        linewidth = Style.EPAISSEUR_LISSE
    )
    ax.set_title(
        commune.replace(Style.SUFFIXE_COMMUNE_BRUXELLES, "").strip(), 
        fontweight="bold"
    )
    ax.set_ylabel(Style.YLABEL)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt = Style.DATE_FORMAT))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval = Style.MOIS_INTERVAL))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation = Style.ROTATION_LABELS)


@log_debut_fin_logger_dynamique("logger")
def supprimer_axes_vides(axes: "np.ndarray[plt.Axes]", nbr_communes: int) -> None:
    """
    Cache les axes inutilisés dans une figure matplotlib, s’il reste des cases vides.

    Args:
        axes (np.ndarray[plt.Axes]): Tableau d’axes matplotlib (grille).
        nbr_communes (int): Nombre de cases effectivement utilisées pour l’affichage.

    Returns:
        None

    Note:
        Permet de ne pas afficher de sous-graphiques vides en fin de page.
        L’axe reste présent mais invisible (n’occupe plus de place dans la figure sauvegardée).

    Example:
        >>> supprimer_axes_vides(axes, 4)

    Étapes:
        1. Parcourt tous les axes à partir de l’indice `nbr_communes`.
        2. Cache chaque axe vide en appelant `set_visible(False)`.

    Tips:
        - À utiliser après avoir tracé toutes les courbes et avant la sauvegarde.
        - Utile pour que la disposition finale soit propre, surtout pour les dernières pages d’un rapport.

    Utilisation:
        Pour les figures multi-pages où le nombre de sous-graphiques varie d’une page à l’autre.

    Limitation:
        - Ne supprime pas les axes du tableau numpy, les rend juste invisibles.
        - Pas de message d’avertissement si tous les axes sont déjà utilisés.

    See also:
        - creer_figure
        - sauvegarder_figure
    """
    for index_commune in range(nbr_communes, len(axes)):
        axes[index_commune].set_visible(False)


@log_debut_fin_logger_dynamique("logger")
def ajouter_legende(fig: plt.Figure) -> None:
    """
    Ajoute une légende commune à l’ensemble de la figure matplotlib (hors des axes).

    Args:
        fig (plt.Figure): Figure matplotlib à annoter.

    Returns:
        None

    Attributes:
        - Les styles de la légende sont définis dans le module `Style`.
        - Ajoute des “handles” représentant les styles de courbe brute et lissée.

    Note:
        La légende est placée sur la figure entière (à droite par défaut), pas sur chaque sous-graphique individuel.

    Example:
        >>> ajouter_legende(fig)

    Étapes:
        1. Prépare les objets de légende (brut/lissé) selon `Style`.
        2. Ajoute la légende à la figure globale via `fig.legend`.

    Tips:
        - Pour déplacer la légende, modifiez `Style.LEGENDE_POSITION`.
        - À faire avant la sauvegarde, pour un rendu complet.

    Utilisation:
        À utiliser juste après avoir rempli les axes, avant sauvegarde ou export.

    See also:
        - creer_figure, tracer_commune_sur_axe, sauvegarder_figure
        - Style
    """
    elements_de_la_legende = [
        plt.Line2D(
            [0], [0], color = Style.COULEUR_BRUT, alpha = Style.ALPHA_BRUT, 
            linewidth = Style.EPAISSEUR_BRUT
        ),
        plt.Line2D([0], [0], color = Style.COULEUR_LISSE, 
                    linewidth = Style.EPAISSEUR_LISSE 
        )
    ]
    libelles = [Style.LEGENDE_BRUT, Style.LEGENDE_LISSE]
    fig.legend(elements_de_la_legende, libelles,  **Style.LEGENDE_POSITION)


@log_debut_fin_logger_dynamique("logger")
def sauvegarder_figure(
        fig: plt.Figure, dossier_de_sauvegarde: str, 
        index_page: int, noms_des_communes: list[str],
        logger: logging.Logger ) -> None:
    """
    Sauvegarde la figure matplotlib au format PNG, avec un nom de fichier codant la page et les communes.

    Args:
        fig (plt.Figure): Figure matplotlib à sauvegarder.
        dossier_de_sauvegarde (str): Dossier où sauvegarder l’image.
        index_page (int): Numéro de la page (utilisé pour le nom de fichier).
        noms_des_communes (list[str]): Communes affichées (sert au nommage du fichier).
        logger (logging.Logger): Logger utilisé pour suivre et signaler la sauvegarde.

    Returns:
        None

    Raises:
        FileNotFoundError: Si le dossier de sauvegarde n’existe pas et ne peut être créé.
        OSError: Pour toute erreur de sauvegarde sur le disque.

    Attributes:
        - Fonction stateless : ne modifie rien d’autre que le fichier cible.
        - Utilise les styles et conventions de nommage définis dans Style.

    Note:
        Le fichier est systématiquement écrasé s’il existe déjà.
        Toutes les opérations critiques (création dossier, sauvegarde, erreurs) sont journalisées
        uniquement via le logger passé en argument.
        Utilisez `plt.close(fig)` après la sauvegarde pour libérer la mémoire graphique.

    Théorie:
        - `plt.savefig` crée un fichier image “figé” à partir de l’état courant de la figure matplotlib.
        - PNG est un format sans perte, idéal pour archivage et impression.

    Example:
        >>> sauvegarder_figure(fig, "visualizations", 2, ["Ixelles", "Bruxelles"], logger)

    Étapes:
        1. Génère le nom du fichier à partir des communes et du numéro de page.
        2. Crée le dossier cible s’il n’existe pas (avec os.makedirs).
        3. Ajuste la disposition (tight_layout, subplots_adjust).
        4. Sauvegarde le fichier PNG (dpi=200).
        5. Loggue l’action et affiche l'emplacement.

    Tips:
        - Adaptez le nom du fichier à vos conventions ou à vos besoins métier.
        - Le PNG généré est indépendant, pas besoin de garder matplotlib ouvert ensuite.
        - Toute information de sauvegarde ou d’erreur est logguée.
        - À utiliser dans un pipeline de génération de rapports ou d’export automatique.

    Utilisation:
        À appeler après avoir créé et rempli la figure, juste avant de fermer la figure avec `plt.close(fig)`.

    Limitation:
        - Ne vérifie pas la place disponible sur le disque.
        - Les emplacements avec caractères spéciaux peuvent poser problème sur certains systèmes.
        - L’agencement peut ne pas être optimal si la figure est très chargée.

    See also:
        - plt.savefig (matplotlib)
        - creer_figure, ajouter_legende, supprimer_axes_vides
        - Style (personnalisation globale)
    """
    try:
        # Retire le suffixe pour chaque nom de commune, garde accents/espaces
        communes_sans_suffixe = [
            c.replace(Style.SUFFIXE_COMMUNE_BRUXELLES, "")
                .strip() for c in noms_des_communes
        ]
        # Utilise les noms tels quels, séparés par "__"
        partie_communes = "__".join(communes_sans_suffixe)
        nom_png = f"savitzky_brut_lisse_{index_page}_{partie_communes}.png"
        fichier_png = os.path.join(dossier_de_sauvegarde, nom_png)
        if not os.path.exists(dossier_de_sauvegarde):
            os.makedirs(dossier_de_sauvegarde, exist_ok=True)
            logger.debug(f"Dossier créé : {dossier_de_sauvegarde}")
        #plt.tight_layout()
        #plt.subplots_adjust(right=0.85)
        plt.savefig(fichier_png, dpi=200, bbox_inches="tight")
        logger.info(f"Figure sauvegardée : {fichier_png}")
    except FileNotFoundError as err:
        logger.error(f"Impossible de créer le dossier {dossier_de_sauvegarde}: {err}",
                     exc_info = True)
        raise
    except OSError as err:
        logger.error(f"Erreur lors de la sauvegarde du fichier : {err}",
                     exc_info = True)
        raise
    except Exception as err:
        logger.error(f"Erreur inattendue dans sauvegarder_figure: {err}",
                     exc_info = True)
        raise