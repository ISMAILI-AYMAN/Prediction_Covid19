# -*- coding: utf-8 -*-

"""
interface_matplotlib.py
--------

"""

import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.widgets import RadioButtons, Slider, RangeSlider
from typing import Any, Optional, Sequence, TypedDict, Callable

from utils_log import log_debut_fin_logger_dynamique
from utils_dates import dates_str_vers_obj
# ============ Typage centralisé ============

class StyleConfig(TypedDict):
    """
    Décrit la configuration de style graphique pour les graphiques matplotlib.
    """
    couleur_brut: str
    couleur_lisse: str
    fond: str
    etiquette_brut: str
    etiquette_lisse: str
    taille_police: int

STYLE: StyleConfig = {
    "couleur_brut": "red",                        # Couleur de la courbe brute
    "couleur_lisse": "blue",                      # Couleur de la courbe lissée
    "fond": "#f5f5fa",                          # Couleur de fond du graphique
    "etiquette_brut": "Données brutes",           # Légende des données brutes
    "etiquette_lisse": "Lissage Savitzky-Golay",  # Légende des données lissées
    "taille_police": 13                           # Taille de police générale pour les labels
}

TITRE_GRAPHE: str = "Évolution des cas COVID-19 - {}"   # Titre du graphique (formaté avec le nom de la commune)
ABSCISSE: str = "Date"                                  # Label de l’axe des X
ORDONNEE: str = "Nombre de cas"                         # Label de l’axe des Y
FORMAT_DATE: str = "%Y-%m"                              # Format des dates affichées en abscisse
INTERVALLE_MOIS: int = 6                                # Espacement des labels de mois sur l’axe X


@log_debut_fin_logger_dynamique("logger")
def tracer_brut_lisse(
        axe: plt.Axes, dates: Sequence[str], brut: Sequence[int], 
        lisse: Sequence[float], commune: str, logger: logging.Logger, 
        periode: Optional[tuple[str, str]] = None,
        style: Optional[StyleConfig] = None ) -> None:
    """
    Met à jour un axe matplotlib avec la courbe brute et la courbe lissée d’une commune.

    Args:
        axe (plt.Axes): L’axe matplotlib à dessiner.
        dates (Sequence[str]): Séquence de dates (format chaîne "YYYY-MM-DD").
        brut (Sequence[int]): Séquence de valeurs brutes.
        lisse (Sequence[float]): Séquence de valeurs lissées (même taille).
        commune (str): Nom de la commune à afficher dans le titre.
        periode (tuple[str, str], optionnel): Période affichée, min et max, ou None.

    Returns:
        None

    Note:
        - Met à jour titres, légendes, couleurs, format X/Y, fond, et labels graphiques.
        - S’appuie sur la constante globale STYLE pour les paramètres graphiques.
        - Suppose que les listes sont synchronisées et non vides.

    Example:
        >>> tracer_brut_lisse(ax, ["2023-01-01"], [5], [4.7], "Ixelles")

    Étapes:
        1. Convertit les dates en objets datetime.
        2. Vide l’axe, puis trace les deux courbes.
        3. Met à jour tous les éléments de style, titres, légendes.

    Tips:
        - Toujours appeler plt.tight_layout() ensuite pour éviter le chevauchement des labels.
        - Utile pour actualiser un graphique interactif sans en recréer un nouveau.

    Utilisation:
        À utiliser dans les callbacks de l’interface interactive, ou pour du tracé statique.

    Limitation:
        - Pas de vérification de la cohérence des tailles de listes.
        - Suppose matplotlib déjà importé.
    """
    """Affiche les données brutes/lissées pour une commune."""
    if style is None:
        style = STYLE
    dates_obj = dates_str_vers_obj(dates, logger)
    axe.clear()
    axe.set_facecolor(str(style["fond"]))
    axe.plot(dates_obj, brut, label=str(style["etiquette_brut"]),
             color=str(style["couleur_brut"]), alpha=0.5)
    axe.plot(dates_obj, lisse, label=str(style["etiquette_lisse"]),
             color=str(style["couleur_lisse"]), linewidth=2)
    axe.legend(fontsize=int(style["taille_police"]))
    axe.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    axe.set_xlabel(ABSCISSE, fontsize=int(style["taille_police"]))
    axe.set_ylabel(ORDONNEE, fontsize=int(style["taille_police"]))
    if periode and periode[0] and periode[1]:
        titre = (f"{TITRE_GRAPHE.format(commune)}\n"
                 f"Période : {periode[0]} → {periode[1]}")
    else:
        titre = TITRE_GRAPHE.format(commune)
    axe.set_title(titre, pad=25, fontsize=int(style["taille_police"])+1)
    axe.xaxis.set_major_formatter(mdates.DateFormatter(FORMAT_DATE))
    axe.xaxis.set_major_locator(mdates.MonthLocator(interval=INTERVALLE_MOIS))
    plt.setp(axe.get_xticklabels(), rotation=0)
    axe.figure.text(0.99, 0.01, "LissageCovid - © Harry FRANCIS 2025",
        fontsize=8, color="#777", ha="right", va="bottom", alpha=0.5
    )
    if dates_obj:
        axe.set_xlim(min(dates_obj), max(dates_obj))
        axe.set_ylim(0, max(max(brut), max(lisse)) * 1.05)
    logger.debug(f"Graphique mis à jour pour {commune} ({len(dates_obj)} points)")


@log_debut_fin_logger_dynamique("logger")
def tracer_serie(
        dates: Sequence[str], brut: Sequence[int], lisse: Sequence[float], commune: str, 
        logger: logging.Logger, periode: Optional[tuple[str, str]] = None, 
        style: Optional[StyleConfig] = None) -> None:
    """
    Affiche la série brute et la série lissée d’une commune dans une figure matplotlib.

    Args:
        dates (Sequence[str]): Dates de la série.
        brut (Sequence[int]): Valeurs brutes de la commune.
        lisse (Sequence[float]): Valeurs lissées correspondantes.
        commune (str): Nom de la commune.
        periode (tuple[str, str], optionnel): Période affichée, min et max, ou None.

    Returns:
        None

    Note:
        - Ouvre une nouvelle fenêtre matplotlib adaptée à la visualisation statique.
        - S’appuie sur tracer_brut_lisse pour la mise en forme détaillée.

    Example:
        >>> tracer_serie(dates, brut, lisse, "Ixelles")

    Étapes:
        1. Crée une nouvelle figure et un axe.
        2. Délègue l’affichage à tracer_brut_lisse().
        3. Affiche la fenêtre graphique.

    Tips:
        - Parfait comme fallback si l’interface interactive ne marche pas (backend, remote).
        - Peut être adapté pour sauvegarder en PNG en ajoutant plt.savefig("...").

    Utilisation:
        Idéal pour des rapports, des captures d’écran ou des vérifications rapides.

    Limitation:
        - Ne retourne aucune donnée, purement visuel.
        - Suppose matplotlib déjà importé.
    """
    fig, axe = plt.subplots(figsize=(12, 6))
    tracer_brut_lisse(
        axe, dates, brut, lisse, commune, logger = logger, periode = periode, 
        style = style
    )
    plt.tight_layout()
    plt.show()
    logger.info(f"Graphique affiché pour {commune}")


def mise_a_jour_libelle_periode(
        val: tuple[float, float], range_slider: Optional[RangeSlider],
        label_periode: Optional[Text], fig: Figure) -> None:
    """
    Met à jour le texte affichant la période sélectionnée.
    """
    if range_slider is not None and label_periode is not None:
        # Cacher les labels numériques après chaque changement
        for label in range_slider.ax.get_xticklabels():
            label.set_visible(False)
        dmin, dmax = val
        # Conversion float matplotlib vers datetime, puis str
        dmin_str = mdates.num2date(dmin).strftime("%d-%m-%Y")
        dmax_str = mdates.num2date(dmax).strftime("%d-%m-%Y")
        label_periode.set_text(f"Période sélectionnée : {dmin_str} → {dmax_str}")
        fig.canvas.draw_idle()


def mettre_a_jour(
        radio: RadioButtons, slider_fenetre: Slider, slider_poly: Slider,
        range_slider: Optional[RangeSlider], fig: Figure, axe: plt.Axes,
        donnees_json: dict[str, Any], str_vers_datetime: Callable[[str], Any], 
        extraire_serie: Callable[[dict[str, Any], str, logging.Logger], 
                                 tuple[list[str], list[int]]],
        appliquer_lissage: Callable[[list[int], int, int, 
                                     logging.Logger], list[float]],
        tracer_brut_lisse: Callable[[plt.Axes, list[str], list[int], list[float], str, 
                                     logging.Logger, Optional[tuple[str, str]], 
                                     Optional[Any]], None],
        logger: logging.Logger, style: dict[str, Any]) -> None:
    """
    Met à jour le graphique interactif selon les paramètres actuels des widgets.
    """
    commune = radio.value_selected
    taille_fenetre = int(slider_fenetre.val)
    degre_polynome = int(slider_poly.val)
    # --- Correction automatique du degré polynôme si >= fenêtre ---
    if degre_polynome >= taille_fenetre:
        degre_polynome = max(1, taille_fenetre - 1)
        slider_poly.set_val(degre_polynome)  # met à jour visuellement le slider

    try:
        dates, brut = extraire_serie(donnees_json, commune, logger)
        periode_affichee = None
        if range_slider is not None and dates:
            borne_min, borne_max = range_slider.val
            dates_dt = [str_vers_datetime(d, logger) for d in dates]
            mask = [(mdates.date2num(dt) >= borne_min) and 
                    (mdates.date2num(dt) <= borne_max) for dt in dates_dt]
            dates = [d for d, keep in zip(dates, mask) if keep]
            brut = [v for v, keep in zip(brut, mask) if keep]
        if dates:
            periode_affichee = (dates[0], dates[-1])
        if degre_polynome >= taille_fenetre:
            axe.clear()
            axe.set_title(
                f"Degré ({degre_polynome}) >= Fenêtre ({taille_fenetre})",
                            fontsize=style["taille_police"])
            fig.canvas.draw_idle()
            return
        lisse = appliquer_lissage(brut, taille_fenetre, degre_polynome, logger)
        tracer_brut_lisse(
            axe, dates, brut, lisse, commune, logger = logger, 
            periode = periode_affichee, style = style)
        fig.canvas.draw_idle()
    except Exception as err:
        axe.clear()
        axe.set_title(f"Erreur : {err}", fontsize=style["taille_police"])
        fig.canvas.draw_idle()


def interface_interactive(
        donnees_json: dict[str, Any], communes: list[str], 
        periode: Optional[tuple[Optional[str], Optional[str]]], logger: logging.Logger,
        commune_defaut: str, taille_fenetre_init: int, degre_polynome_init: int,
        str_vers_datetime: Callable[[str], Any],
        extraire_serie: Callable[[dict[str, Any], str, logging.Logger],
                                 tuple[list[str], list[int]]],
        appliquer_lissage: Callable[[list[int], int, int, logging.Logger], list[float]],
        tracer_brut_lisse: Callable[[plt.Axes, list[str], list[int], list[float], str,
                                     Optional[logging.Logger], 
                                     Optional[tuple[str, str]]], None],
        style: Optional[StyleConfig] = None ) -> None:
    """Interface interactive matplotlib pour le lissage."""
    plt.close('all')
    fig, axe = plt.subplots(figsize=(15, 8))
    fig.canvas.manager.set_window_title("Lissage Covid méthode Savitzky-Golay")
    plt.subplots_adjust(left=0.28, right=0.98, bottom=0.31)

    # --- Widgets ---
    zone_radio = plt.axes([0.03, 0.29, 0.22, 0.64])
    radio = RadioButtons(zone_radio, communes, active=communes.index(commune_defaut))
    zone_slider_fenetre = plt.axes([0.30, 0.13, 0.60, 0.04])
    slider_fenetre = Slider(
        zone_slider_fenetre, "Fenêtre lissage", 5, 21,
        valinit = taille_fenetre_init, valstep = 2
    )
    zone_slider_poly = plt.axes([0.30, 0.07, 0.60, 0.04])
    slider_poly = Slider(
        zone_slider_poly, "Degré polynôme", 1, 7, valinit = degre_polynome_init,
        valstep = 1)
    zone_slider_dates = plt.axes([0.30, 0.18, 0.60, 0.04])
    toutes_les_dates = sorted(donnees_json.keys())
    if toutes_les_dates:
        dates_float = mdates.date2num(
            [str_vers_datetime(d, logger) for d in toutes_les_dates]
        )
        range_slider = RangeSlider(
            zone_slider_dates, "Période", valmin = dates_float[0], 
            valmax = dates_float[-1], valinit = (
                dates_float[0], dates_float[-1]
                ), valstep = 1)
        label_periode = fig.text(0.30, 0.23, "", fontsize=11, color="#333")
    else:
        range_slider = None
        label_periode = None

    # --- Initialisation + Callbacks ---
    if range_slider is not None:
        # MAJ du texte label dès le début
        mise_a_jour_libelle_periode(range_slider.val, range_slider, label_periode, fig)
        # MAJ du texte label sur mouvement
        range_slider.on_changed(lambda val: mise_a_jour_libelle_periode(val, range_slider, label_periode, fig))
        # MAJ du graphe sur mouvement
        range_slider.on_changed(lambda val: _maj())

    # Lier callbacks externes, tout plat
    #def _maj_lib_periode(val):
    #    mise_a_jour_libelle_periode(val, range_slider, label_periode, fig)
    def _maj(*args):
        mettre_a_jour(
            radio, slider_fenetre, slider_poly, range_slider, fig, axe,
            donnees_json, str_vers_datetime, extraire_serie, appliquer_lissage,
            tracer_brut_lisse, logger, style
        )


    radio.on_clicked(lambda label: _maj())
    slider_fenetre.on_changed(lambda val: _maj())
    slider_poly.on_changed(lambda val: _maj())
    _maj()
    plt.show()
