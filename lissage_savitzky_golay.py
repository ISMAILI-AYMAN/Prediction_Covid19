

import logging
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from scipy.signal import savgol_filter

from utils_log import log_debut_fin_logger_dynamique
from constantes import DonneesCovid, extraire_serie

DonneesLissage = dict[str, dict[str, float]]


@log_debut_fin_logger_dynamique("logger")
def extraire_brut_lisse(
        donnees_json: DonneesCovid, donnees_lissees: DonneesLissage, commune: str,
        logger: logging.Logger) -> tuple[list[str], list[int], list[float]]:
    """
    Extrait pour une commune donnée : les dates, les valeurs brutes et les valeurs lissées,
    synchronisées sur les dates communes aux deux jeux de données.

    Args:
        donnees_json (DonneesCovid): Cas bruts, par date et commune.
        donnees_lissees (DonneesLissage): Séries lissées, par date et commune.
        commune (str): Nom de la commune concernée.

    Returns:
        tuple[list[str], list[int], list[float]]:
            - Liste de dates (str) synchronisées.
            - Valeurs brutes (int) pour ces dates.
            - Valeurs lissées (float) pour ces dates.

    Note:
        - Ne retourne que les points présents dans les deux séries.
        - Utile pour affichage ou analyse comparative.

    Example:
        >>> dates, brut, lisse = extraire_brut_lisse(data, data_liss, "Ixelles")

    Étapes:
        1. Extrait la série brute (dates, valeurs) pour la commune.
        2. Parcourt chaque date, retient celles où le lissage existe.
        3. Retourne trois listes synchronisées.

    Tips:
        - À utiliser avant tout affichage ou sauvegarde commune/brute+lissé.

    Utilisation:
        - Pour synchroniser deux sources de données avant traitement graphique.

    Limitation:
        - Si aucune valeur lissée n’existe : listes retournées vides.

    See also:
        - extraire_serie (dans utils)
    """
    dates, brut = extraire_serie(donnees_json, commune, logger)
    lisse: list[float] = []
    dates_ok: list[str] = []
    brut_ok: list[int] = []
    for date, val in zip(dates, brut):
        val_lisse: Optional[float] = donnees_lissees.get(date, {}).get(commune, None)
        if val_lisse is not None:
            lisse.append(val_lisse)
            dates_ok.append(date)
            brut_ok.append(val)
    logger.debug(f"Extraction brute/lissée pour {commune} : {len(dates_ok)} points.")
    return dates_ok, brut_ok, lisse


@log_debut_fin_logger_dynamique("logger")
def appliquer_lissage(
        valeurs: list[int], taille_fenetre: int, degre_polynome: int, 
        logger: logging.Logger) -> list[float]:
    """
    Applique le filtre Savitzky-Golay à une série d’entiers : lissage local.

    Args:
        valeurs (List[int]): Série d’entiers à lisser.
        taille_fenetre (int): Taille de la fenêtre (impair, > degré).
        degre_polynome (int): Degré du polynôme du filtre.

    Returns:
        List[float]: Valeurs lissées (même taille).

    Raises:
        Warning: Série trop courte pour la fenêtre : retourne la série brute.

    Note:
        - Retourne toujours une liste de floats, même si entrée entière.
        - Si la fenêtre dépasse la taille, aucune modification n’est appliquée.

    Example:
        >>> appliquer_lissage([1,2,3,4,5], 5, 2)
        [1.0, 2.0, 3.0, 4.0, 5.0]

    Étapes:
        1. Vérifie la taille de la série.
        2. Si trop courte, copie brute (log warning).
        3. Sinon, applique le filtre Savitzky-Golay (scipy.signal).

    Tips:
        - Adapter la taille de fenêtre selon la longueur de la série : 
          fenêtre trop grande = lissage impossible.

    Utilisation:
        - Pour tout prétraitement de données temporelles bruitées.

    Limitation:
        - Ne gère pas le cas où le degré ≥ taille de fenêtre (à vérifier en amont).

    See also:
        - scipy.signal.savgol_filter (doc officielle)
    """
    if len(valeurs) < taille_fenetre:
        msg = (f"Série trop courte pour le lissage : "
               f"{len(valeurs)} < {taille_fenetre}. Brute copiée.")
        warnings.warn(msg)
        logger.warning(msg)
        return [float(v) for v in valeurs]
    resultat = savgol_filter(valeurs, window_length=taille_fenetre, 
                             polyorder=degre_polynome).tolist()
    logger.debug(f"Lissage : fen={taille_fenetre}, "
                 f"poly={degre_polynome}, points={len(resultat)}")
    return resultat


@log_debut_fin_logger_dynamique("logger")
def creer_lissage(
        donnees_json: DonneesCovid, communes: list[str], taille_fenetre: int, 
        degre_polynome: int, logger: logging.Logger) -> DonneesLissage:
    """
    Applique le lissage Savitzky-Golay à chaque commune, retourne une structure {date: {commune: valeur}}.

    Args:
        donnees_json (DonneesCovid): Données brutes, par date et commune.
        communes (List[str]): Communes à traiter.
        taille_fenetre (int): Taille de la fenêtre du filtre.
        degre_polynome (int): Degré du polynôme.

    Returns:
        DonneesLissage: Valeurs lissées indexées par date puis commune.

    Note:
        - Les valeurs lissées sont arrondies à 2 décimales.
        - Structure compatible export JSON/visualisation.

    Example:
        >>> lissages = creer_lissage(data, ["Ixelles"], 11, 3)

    Étapes:
        1. Pour chaque commune : extrait la série brute.
        2. Applique le lissage.
        3. Range le résultat dans {date: {commune: valeur}}.

    Tips:
        - Choisissez la taille de fenêtre adaptée à la densité de données.
        - Permet une sauvegarde ou analyse multicommunes.

    Utilisation:
        - En amont de tout affichage, export, ou analyse.

    Limitation:
        - Retourne des 0.0 si la fenêtre est inadaptée (trop grande).
    """
    resultat: DonneesLissage = {}
    for commune in communes:
        dates, valeurs = extraire_serie(donnees_json, commune, logger)
        valeurs_lissees = appliquer_lissage(
            valeurs, taille_fenetre, degre_polynome, logger
        )
        for date, val in zip(dates, valeurs_lissees):
            resultat.setdefault(date, {})[commune] = round(float(val), 2)
        logger.info(f"Lissage {commune}: {len(valeurs_lissees)} points")
    return resultat
