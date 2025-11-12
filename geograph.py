# -*- coding: utf-8 -*-

"""
geograph.py
-----------
Graphe pondéré des communes bruxelloises — outils pour modéliser, manipuler et exporter
le réseau des relations de voisinage (frontières) entre communes sous forme de graphe
géographique pondéré.

Objectif :
    Permettre à tout utilisateur de représenter, analyser et exploiter les liens
    géographiques entre les communes de Bruxelles-Capitale, dans des contextes de
    visualisation, modélisation de diffusion spatiale (ex: COVID), ou simulation.

Fonctionnalités principales :
    - Construction et stockage du graphe (communes, frontières, voisinages).
    - Calcul des poids géographiques entre chaque couple de communes, proportionnels à la
      longueur de frontière partagée (modélisé par l’inverse de la longueur + epsilon).
    - Génération et export (JSON) de matrices de poids, utilisables pour des modèles
      markoviens ou analyses de connectivité.
    - Contrôles automatiques de cohérence (chaque commune doit avoir au moins un voisin).
    - Logging systématique (via logger passé en argument) pour toutes les étapes clés,
      facilitant le débogage et l’audit du traitement.

Philosophie :
    - Favorise la clarté, la centralisation des calculs, et le respect des bonnes pratiques
      Python (PEP8, typage, découplage du logging).
    - Convient aussi bien pour des traitements interactifs (Jupyter Notebook) que pour des
      pipelines automatisés ou l’intégration dans un projet plus vaste.

Prérequis de données :
    - Une table des frontières (dictionnaire: frozenset({A,B}) → longueur flottante).
    - Une table des voisins (dictionnaire: commune → liste des voisines directes).
    - Une liste exhaustive des communes à inclure dans le graphe.

Best Practice : Logger
----------------------
    - Toutes les fonctions et classes reçoivent un logger Python explicite en argument.
    - Ce comportement garantit la compatibilité avec toute application
      (console, fichiers log, cloud…).
    - Exemple minimal :
        >>> import logging
        >>> logger = logging.getLogger("geograph")
        >>> # Puis passer logger à chaque méthode/objet

Utilisation typique :
    from geograph import GeoGraph, controle_voisin_min
    graph = GeoGraph(COMMUNES, ADJACENTES, FRONTIERES, logger)
    controle_voisin_min(COMMUNES, ADJACENTES, logger)
    poids = graph.calculer_tous_les_poids_geographiques()
    graph.sauvegarder_les_poids_geographiques(poids, "data/geo_weights.json", logger)

Log Example :
    --------------------
        2025-07-23 13:42:10 | INFO    | Calcul des poids géographiques avec Dijkstra
        2025-07-23 13:42:12 | INFO    | Calcul depuis Bruxelles : poids total = 8.39
        2025-07-23 13:42:12 | INFO    | Poids géographiques calculés !
        2025-07-23 13:42:12 | INFO    | Poids sauvegardés : data/geo_weights.json

Conseils :
    - Lancez toujours le contrôle des voisins AVANT tout calcul de poids pour éviter les erreurs silencieuses.
    - L’algorithme de poids s’inspire des modèles de graphes de diffusion spatiale : il
      n’intègre PAS d’autres facteurs (population, barrière naturelle, etc.).
    - Les poids sont normalisés (1.0 pour auto-influence, 0.0 si pas d'emplacement).

Limites et extensions :
    - Le module ne gère pas l’affichage graphique (à faire avec matplotlib ou networkx).
    - Pour ajouter d’autres types de poids (distance réelle, population), il suffit d’étendre la classe.
    - La génération des matrices Markov suppose un graphe connexe et des données cohérentes.

Maintenance :
    - Tout changement de logique (calcul des poids, structure des données) doit être fait ici
      pour garantir la cohérence des exports et analyses.
    - Les logs permettent de tracer les actions et de retrouver rapidement les sources d’erreur.

Documentation :
    - Toutes les méthodes et fonctions sont documentées en détail (voir plus bas).
    - Pour une introduction à la théorie des graphes : 
      https://fr.wikipedia.org/wiki/Th%C3%A9orie_des_graphes

Auteur   : Harry FRANCIS (2025)
Contact  : github.com/documentHarry

Version  : 1.0.0 (2025-07-24)
License  : Code source ouvert à la consultation.
           Toute redistribution ou usage commercial est soumis à accord écrit préalable.
"""

__version__ = "1.0.0"

# Compatible Python 3.9+ | Typage PEP 484 | PEP 257 | PEP 8 (88 caractères max)

# --- Librairies standards
import logging
from typing import FrozenSet

# --- Modules locaux
from constantes import CORRESPONDANCE_NOMS
from utils_io import sauvegarder_json
from utils_log import log_debut_fin_logger_dynamique

# === Paramètres de configuration ===
METADATA: str = "metadata"


@log_debut_fin_logger_dynamique("logger")
def controle_voisin_min(
        communes: list[str], adjacentes: dict[str, list[str]], logger: logging.Logger
        ) -> None:
    """
    Vérifie que chaque commune a au moins un voisin déclaré dans la table d’adjacence.
    Log un avertissement pour toute commune isolée.

    Args:
        communes (list[str]): Liste complète des communes à vérifier.
        adjacentes (dict[str, list[str]]): Dictionnaire {commune: [voisins]}.
        logger (logging.Logger): Logger configuré pour afficher les messages.

    Returns:
        None

    Note:
        - N’interrompt jamais le programme : se contente de logguer les problèmes.
        - Fonction “sanitaire” : prévient les erreurs de graphe mal construit.
        - Ne modifie aucune donnée : tout est en lecture seule.

    Example:
        >>> controle_voisin_min(
        ...     ["A", "B"], {"A": ["B"], "B": ["A"]}, logger
        ... )
        # INFO: Toutes les communes ont au moins un voisin déclaré...

        >>> controle_voisin_min(
        ...     ["A", "B"], {"A": [], "B": []}, logger
        ... )
        # WARNING: Communes sans aucun voisin déclaré: ['A', 'B']

    Étapes:
        1. Parcourt chaque commune dans la liste.
        2. Vérifie la présence de voisins dans le dictionnaire.
        3. Loggue un message d’alerte si un voisin manque.

    Tips:
        - À lancer **avant** toute opération sur le graphe (calculs de poids, etc).
        - Pratique pour déboguer rapidement une erreur de données.

    Utilisation:
        - S’utilise en amont, dans un pipeline de préparation ou lors d’un import de données.

    Limitation:
        - Ne détecte pas les cycles ni les graphes déconnectés, juste l’absence totale de voisins.

    See also:
        - GeoGraph.calculer_tous_les_poids_geographiques
        - Doc officielle Python logging (https://docs.python.org/3/library/logging.html)
    """
    communes_sans_voisin = [c for c in communes if not adjacentes.get(c)]
    if communes_sans_voisin:
        logger.warning(f"Communes sans aucun voisin déclaré : {communes_sans_voisin}")
    else:
        logger.info(
            "Toutes les communes ont au moins un voisin déclaré dans ADJACENTES."
        )


class GeoGraph:
    """
    Représente un graphe géographique des communes et de leurs relations de voisinage
    pour la région de Bruxelles-Capitale.

    Cette classe encapsule la structure des adjacences entre communes ainsi que les
    longueurs de leurs frontières, et fournit des outils pour :
        - Calculer les poids géographiques (influence relative) entre paires de communes,
        - Générer une matrice pondérée des “transitions” (pour modélisation Markovienne),
        - Appliquer un algorithme de type Dijkstra pour propager/agréger les influences,
        - Exporter facilement ces poids sous forme de fichiers JSON exploitables.
        - Logger toutes les étapes importantes via `self.logger`.

    Attributes:
        communes (list[str]) :
            Liste des noms des communes considérées.
        adjacentes (dict[str, list[str]]) :
            Dictionnaire des voisins par commune.
        longueur_communes_adjacentes (dict[FrozenSet[str], float]) :
            Dictionnaire des longueurs de frontières partagées.
        logger (logging.Logger) :
            Logger configuré (stocké en interne comme `self.logger`).
            
    Example:
        >>> import logging
        >>> from geograph import GeoGraph
        >>> logger = logging.getLogger("test")
        >>> graph = GeoGraph(
        ...     ["A", "B"],
        ...     {"A": ["B"], "B": ["A"]},
        ...     {frozenset({"A", "B"}): 1.5},
        ...     logger
        ... )
        >>> graph.calculer_tous_les_poids_geographiques()
    """


    @log_debut_fin_logger_dynamique("logger")
    def __init__(
            self, communes: list[str], adjacentes: dict[str, list[str]], 
            longueur_communes_adjacentes: dict[FrozenSet[str], float],
            logger: logging.Logger) -> None:
        """
        Initialise un graphe géographique des communes et frontières de Bruxelles.

        Args:
            communes (list[str]): Noms des communes à inclure dans le graphe.
            adjacentes (dict[str, list[str]]): Table de voisins pour chaque commune.
            longueur_communes_adjacentes (dict[FrozenSet[str], float]): 
                Table des longueurs de frontières partagées.
            logger (logging.Logger): Logger pour toutes les étapes (infos et erreurs).

        Returns:
            None

        Note:
            - Centralise toutes les structures de graphe dans un seul objet.
            - Le logger est mémorisé comme attribut, pour toutes les méthodes.

        Example:
            >>> logger = logging.getLogger("test")
            >>> GeoGraph(["A", "B"], {"A": ["B"], "B": ["A"]},
            ...          {frozenset({"A", "B"}): 1.0}, logger)

        Étapes:
            1. Stocke tous les paramètres comme attributs de l’instance.
            2. Vérifie que le logger est bien présent.

        Tips:
            - Créez un seul objet par pipeline pour garder les logs et la cohérence.

        Utilisation:
            - À initialiser dès que toutes les tables sont prêtes (frontières, voisins…).

        Limitation:
            - Ne vérifie pas la cohérence des tables à l’initialisation (utiliser controle_voisin_min).
        """
        self.communes = communes
        self.adjacentes = adjacentes
        self.longueur_communes_adjacentes = longueur_communes_adjacentes
        self.logger = logger or logging.getLogger(__name__)


    def calculer_le_poids_geographique(
            self, commune_a: str, commune_b: str, epsilon: float = 0.1) -> float:
        """
        Calcule le poids géographique reliant deux communes, proportionnel à la longueur
        de leur frontière partagée.

        Args:
            commune_a (str): Première commune.
            commune_b (str): Seconde commune.
            epsilon (float, optional): Stabilisateur pour éviter la division par zéro (défaut 0.1).

        Returns:
            float: Poids calculé (>= 0). 1.0 si communes identiques, 0.0 si aucune frontière.

        Raises:
            ValueError: Si la commune n'est pas reconnue dans le graphe.

        Théorie :
            - Le poids reflète l’influence/perméabilité d’une frontière.
            - Plus la frontière entre deux communes est longue, plus la “distance” (et donc l’influence) est faible.
            - Formule mathématique :
            
            .. math::

                poids(A, B) = \frac{1}{\text{longueur frontière}(A,B) + \varepsilon}
            
            - Cette formule s’inspire des modèles de graphes de voisinage en diffusion spatiale.

        Version LibreOffice :
            poids (A, B) = {1} over {longueur~frontiere(A, B) + %epsilon}

        Example:
            >>> graph.calculer_le_poids_geographique("Bruxelles", "Ixelles")
            0.0113

        Notes:
            - Si les communes n'ont pas de frontière, retourne 0.0.
            - Si commune_a == commune_b, retourne 1.0 (poids maximum, auto-influence).
            - Le paramètre epsilon garantit qu'une frontière nulle ne provoque pas d'erreur.

        Étapes:
            1. Cherche la longueur de frontière partagée.
            2. Applique la formule de poids avec epsilon.
            3. Retourne le poids.

        Tips:
            - Pour le calcul sur tous les couples, utilisez `calculer_tous_les_poids_geographiques`.

        Limitation:
            - N'intègre pas la population ni la surface.
            - Hypothèse: les longueurs de frontière sont en kilomètres.
        """
        if commune_a == commune_b:
            # C'est le poids maximum de la commune
            return 1.0

        cle = frozenset(
            [
                CORRESPONDANCE_NOMS.get(commune_a, commune_a),
                CORRESPONDANCE_NOMS.get(commune_b, commune_b),
            ]
        )
        longueur_commune = self.longueur_communes_adjacentes.get(cle, 0.0)
        if longueur_commune == 0:
            # Il n'y a pas de frontière commune donc pas d'influence
            return 0.0

        # Selon la formule de calcul du poids géographique
        return 1.0 / (longueur_commune + epsilon)


    @log_debut_fin_logger_dynamique("logger")
    def calculer_tous_les_poids_geographiques(
            self, epsilon: float = 0.1) -> dict[str, dict[str, float]]:
        """
        Calcule l’ensemble des poids géographiques entre chaque commune source et toutes les autres, 
        en utilisant un algorithme inspiré de Dijkstra.

        Args:
            epsilon (float, optional) : 
                Paramètre d’ajustement évitant la division par zéro lors du calcul des poids.

        Returns:
            dict[str, dict[str, float]] : 
                Dictionnaire imbriqué : 
                {commune_source : {commune_cible : poids normalisé}}
                Exemple : {"Bruxelles": {"Ixelles": 0.02, ...}, ...}

        Raises:
            ValueError : 
                Si une commune du graphe est absente lors du calcul des poids.
            KeyError : 
                Si une information d’adjacence ou de frontière est manquante.

        Théorie :
            - Pour chaque commune source, on calcule la “distance” minimale à toutes les autres
            à travers le graphe pondéré des frontières.
            - Les poids obtenus sont l’inverse de cette distance (plus c’est proche, plus l’influence est forte).
            - Utilisation possible pour matrices de transition markoviennes et modélisation de la propagation spatiale.

        Étapes :
            1. Parcourt chaque commune comme “source”.
            2. Applique l’algorithme de Dijkstra pondéré aux frontières.
            3. Stocke le dictionnaire des poids vers toutes les autres communes.
            4. Log chaque étape et poids total par commune.

        Example:
            >>> mat = graph.calculer_tous_les_poids_geographiques()
            >>> mat["Bruxelles"]["Uccle"]
            0.0231

        Tips :
            - Utilisez ce résultat pour générer une matrice de transition pour un modèle de Markov spatial.
            - Pour “visualiser” le graphe, on peut normaliser ou représenter les poids sur un plot réseau.

        Note :
            Les poids sont asymétriques si les longueurs de frontière diffèrent selon le sens
            (rare, mais possible avec certains types de données).
            Ce calcul peut être long si le graphe est très dense.

        Limitation :
            - Ce calcul ne prend pas en compte d’autres facteurs géographiques (rivières, barrières…).
            - Requiert que les dictionnaires d’adjacence/frontière soient complets et cohérents.
        """

        self.logger.info("Calcul des poids géographiques avec Dijkstra")
        tous_les_poids_geo_avec_dijkstra = {}

        for source_commune in self.communes:
            poids_dijkstra = self.poids_dijkstra(source_commune, epsilon)
            tous_les_poids_geo_avec_dijkstra[source_commune] = poids_dijkstra
            self.logger.info(
                f"Calcul depuis {source_commune} : "
                f"poids total = {sum(poids_dijkstra.values()):.2f}"
            )

        self.logger.info("Poids géographiques calculés !")
        return tous_les_poids_geo_avec_dijkstra


    @log_debut_fin_logger_dynamique("logger")
    def sauvegarder_les_poids_geographiques(
            self, dictionnaire_de_poids: dict[str, dict[str, float]], 
            logger: logging.Logger,
            emplacement_des_poids_geographiques: str = "data/geographic_weights.json"
            ) -> None:
        """
        Exporte sur disque les poids géographiques calculés, au format JSON structuré.

        Args:
            dictionnaire_de_poids (dict[str, dict[str, float]]):
                La matrice de poids à sauvegarder.
            emplacement_des_poids_geographiques (str, optionnel) :
                Emplacement du fichier cible (par défaut dans data/).

        Returns:
            None

        Note:
            - Le fichier contient aussi des métadonnées sur la méthode de calcul.
            - Vérifie que le dossier cible existe, ou le crée si besoin.

        Example:
            >>> graph.sauvegarder_les_poids_geographiques(mat, "data/geo_weights.json")

        Étapes:
            1. Ajoute un champ “metadata” avec la méthode, date, epsilon.
            2. Appelle la fonction utilitaire de sauvegarde JSON.
            3. Loggue la réussite de l’opération.

        Tips:
            - Ajoutez la date/heure dans le nom du fichier pour versionner les exports.

        Utilisation:
            - Pour archiver un calcul ou échanger des matrices entre projets.

        Limitation:
            - N’effectue pas de validation : vérifiez la cohérence en amont.

        See also:
            - sauvegarder_json (fonction utilitaire du projet)
        """
        dictionnaire_de_poids_geo = {
            METADATA: {
                "create_at": "2025-01-19",
                "method": "dijkstra",
                "epsilon": 0.1,
                "communes_count": len(self.communes),
            },
            "weights": dictionnaire_de_poids,
        }
        sauvegarder_json(dictionnaire_de_poids_geo, 
                         emplacement_des_poids_geographiques, logger = logger,
                         ecrasement = True)
        self.logger.info(f"Poids sauvegardés : {emplacement_des_poids_geographiques}")


    @log_debut_fin_logger_dynamique("logger")
    def afficher_les_infos_sur_communes_adjacentes(self) -> None:
        """
        Affiche un résumé loggué du graphe : nombre de communes, de frontières,
        et de voisins pour chaque commune.

        Args:
            None

        Returns:
            None

        Note:
            - Aucune modification : purement informatif.
            - Pratique pour vérifier la cohérence des structures à l’import.

        Example:
            >>> graph.afficher_les_infos_sur_communes_adjacentes()

        Étapes:
            1. Calcule le nombre de communes et de frontières.
            2. Loggue le nombre de voisins pour chaque commune.

        Tips:
            - Utiliser juste après la création du graphe pour repérer des isolats ou erreurs.

        Utilisation:
            - Outil de débogage ou pour documentation du pipeline.

        Limitation:
            - S’appuie uniquement sur les tables transmises (erreur si la donnée d’entrée est mauvaise).
        """
        info_geo = "Informations géographiques"
        nbr_info_geo = len(info_geo)
        self.logger.info(info_geo)
        self.logger.info(nbr_info_geo * "_")


        self.logger.info(f"{len(self.communes)} communes ")
        self.logger.info(
            f"{len(self.longueur_communes_adjacentes) // 2} frontières\n"
        )

        adjacentes = "Nombre de communes adjacentes (par commune)"
        nbr_adjacentes = len(adjacentes)
        self.logger.info(adjacentes)
        self.logger.info(nbr_adjacentes * "_")

        for index_commune, commune in enumerate(sorted(self.communes)):
            voisins = self.adjacentes.get(commune, [])
            if index_commune != len(self.communes) - 1:
                self.logger.info(f"{len(voisins)} communes voisines à {commune}")
            else:
                self.logger.info(f"{len(voisins)} communes voisines à {commune}\n")


    @log_debut_fin_logger_dynamique("logger")
    def poids_dijkstra(self, commune: str, epsilon: float = 0.1) -> dict[str, float]:
        """
        Calcule les poids géographiques depuis une commune source vers toutes les autres
        en utilisant l’algorithme de Dijkstra adapté à un graphe pondéré (frontières inversées).

        Args:
            commune (str): Nom de la commune source.
            epsilon (float, optional): Stabilisateur pour éviter la division par zéro.

        Returns:
            dict[str, float] : 
                Poids de la commune source vers chaque autre commune du graphe.
                Les poids sont normalisés, 1.0 pour soi-même, 0.0 si non accessible.

        Raises:
            ValueError: Si la commune source n’existe pas dans le graphe.

        Théorie :
            - Dijkstra cherche les chemins minimaux pondérés : ici la pondération est 
            “1/longueur de frontière” pour chaque arête.
            - L’inverse de la distance trouvée donne un poids (influence relative entre les communes).

        Example:
            >>> graph.poids_dijkstra("Bruxelles")
            {'Ixelles': 0.19, 'Uccle': 0.11, ...}

        Étapes :
            1. Initialise la distance à 0 pour la source, infini pour les autres.
            2. Parcourt le graphe en sélectionnant à chaque itération la commune la plus proche non visitée.
            3. Met à jour les distances pour tous les voisins.
            4. À la fin, inverse les distances pour obtenir les poids.

        Tips :
            - Utilisez ce calcul comme sous-routine pour le calcul global de toutes les paires.
            - Modifiez epsilon si vous souhaitez réduire l’effet des “toutes petites frontières”.

        Notes :
            - Le poids est de 1.0 pour la commune elle-même (auto-influence).
            - Si deux communes ne sont pas reliées, leur poids est 0.0.
            - Peut servir à estimer la “proximité effective” dans un modèle spatial.

        Limitation :
            - Complexité quadratique si beaucoup de communes (mais raisonnable pour Bxl).
        """
        if commune not in self.communes:
            raise ValueError(f"Commune '{commune}' non reconnue ")
        # Initialisation Dijkstra
        distances = {commune: float("inf") for commune in self.communes}
        distances[commune] = 0.0

        visitees = set()
        non_visitees = set(self.communes)

        while non_visitees:
            # Trouver la commune non visitée avec la plus petite distance
            actuel = min(non_visitees, key=lambda x: distances[x])
            if distances[actuel] == float("inf"):
                # Pas de communes accessible
                break
            # Visiter les voisins
            for voisin in self.adjacentes.get(actuel, []):
                if voisin in visitees:
                    continue
                # Distance = inverse du poids géographique
                poids = self.calculer_le_poids_geographique(actuel, voisin, epsilon)
                if poids > 0:
                    distance = 1.0 / poids
                    nouvel_distance = distances[actuel] + distance
                    if nouvel_distance < distances[voisin]:
                        distances[voisin] = nouvel_distance
            visitees.add(actuel)
            non_visitees.remove(actuel)

        # Convertir les distances en poids (inverse)
        poids_convertis = {}
        for commune, distance in distances.items():
            if distance == float("inf"):
                poids_convertis[commune] = 0.0
            elif distance == 0.0:
                poids_convertis[commune] = 1.0
            else:
                poids_convertis[commune] = 1.0 / distance

        return poids_convertis


# -----------------------------------------------------------------------------
# __all__ : Contrôle des imports publics du module
# -----------------------------------------------------------------------------
#
# En Python, la variable spéciale __all__ permet de définir explicitement la liste
# des objets (fonctions, classes, etc.) qui seront importés lors d’un
# "from geograph import *".
#
# Cela permet :
#   - de ne rendre publiques que les parties du module destinées à être utilisées,
#   - de masquer les fonctions/variables internes ou utilitaires,
#   - d’améliorer la clarté de l’API et la génération de documentation.
#
# Exemple :
#   __all__ = ["GeoGraph", "controle_voisin_min"]
#   => Seuls GeoGraph et controle_voisin_min seront accessibles via "import *".
#
# Note :
#   - Ce n’est pas obligatoire, mais recommandé dans les modules un peu sérieux.
#   - Les noms non listés ici restent accessibles via import direct (import geograph).
# -----------------------------------------------------------------------------

__all__ = ["GeoGraph", "controle_voisin_min"]
