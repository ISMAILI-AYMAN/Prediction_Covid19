# -*- coding: utf-8 -*-

"""
style.py
--------
Centralise toutes les constantes graphiques utilisées pour la visualisation des données COVID-19
par commune.

Ce module fournit :
    - Les couleurs, tailles de police, styles et paramètres matplotlib pour assurer une
      cohérence visuelle des graphes.
    - Un point d’entrée unique pour modifier l’apparence de tous les graphiques du projet.
    - Des outils pour harmoniser les légendes, titres, formats de dates, et la disposition
      générale des figures.

Objectif :
    - Offrir un référentiel unique pour la charte graphique des analyses COVID par commune.
    - Permettre une modification rapide et globale du style sans toucher aux scripts principaux.

Fonctionnalités principales :
    - Classe Style : stocke toutes les constantes graphiques du projet (couleurs, tailles...).
    - (Peut être étendue pour d’autres styles ou variantes graphiques du projet.)

Prérequis :
    - Aucune dépendance externe, sauf matplotlib pour l’usage final.
    - À importer partout où la cohérence graphique est souhaitée.

Philosophie :
    - Centralise le style pour éviter les redondances et garantir l’uniformité.
    - Encourage la clarté et la maintenance du code.

Utilisation typique :
    >>> from style import Style
    >>> plt.plot(x, y, color=Style.COULEUR_LISSE, linewidth=Style.EPAISSEUR_LISSE)

Best Practice :
    - Ne jamais dupliquer les constantes : toujours passer par Style.<CONSTANTE>.
    - Modifier uniquement ici pour changer la charte graphique du projet.

Conseils :
    - Pour ajouter un nouveau style, dérivez la classe Style (ex : class Style2(Style) : ...).
    - Pour des modifications temporaires, surchargez localement les constantes.

Limitations :
    - Ce module ne gère pas la logique métier ni la génération de figures.
    - Pas de contrôle dynamique : tout est statique, pas de setters/getters.

Maintenance :
    - Toute évolution de la charte graphique doit passer par ce fichier.
    - Bien commenter toute nouvelle constante ajoutée.

Documentation :
    - Tous les attributs de Style sont commentés et regroupés par catégorie.
    - Pour la personnalisation avancée, voir la doc officielle matplotlib :
      https://matplotlib.org/stable/users/explain/customizing.html

Auteur   : Harry FRANCIS (2025)
Contact  : github.com/documentHarry
Version  : 1.0.0 (2025-07-23)
License  : Code source ouvert à la consultation.
           Toute redistribution ou usage commercial est soumis à accord écrit préalable.
"""

class Style:
    """
    Constantes de style pour les graphes COVID-19 par commune.

    Cette classe centralise tous les paramètres d’habillage graphique : couleurs, épaisseurs,
    labels, titres, options de grille, positionnements de la légende, et paramètres d’axes X.
    Les modifications ici s’appliquent à toutes les fonctions de rendu du projet.

    Catégories :
        - Couleurs et styles de courbes
        - Titres, labels, et légendes
        - Dimensions des figures et disposition des sous-graphiques
        - Format des axes de dates
        - Positionnement de la légende
        - Suffixes pour affichage “propre” des noms de commune

    Utilisation :
        - Importer la classe puis référencer les attributs via `Style.<CONSTANTE>`.
        - Permet de modifier facilement la charte graphique du projet en un seul endroit.

    Tips :
        - Étendez cette classe pour d’autres styles de figures si besoin :
          `class AutreStyle(Style): ...`
        - Pour des variantes temporaires : créez une nouvelle classe dérivée ou surchargez en local.

    Exemple :
        ax.plot(x, y, color=Style.COULEUR_BRUT, linewidth=Style.EPAISSEUR_BRUT)

    Note :
        Le nommage en MAJUSCULE n'est pas obligatoire (puisque attributs de classe),
        mais facilite la lecture comme pour de “vraies” constantes.
    """

    # --- Couleurs et styles de courbes ---
    COULEUR_BRUT = "orange"           #: Couleur de la courbe brute
    COULEUR_LISSE = "blue"            #: Couleur de la courbe lissée
    ALPHA_BRUT = 0.5                  #: Transparence de la courbe brute
    EPAISSEUR_BRUT = 1                #: Épaisseur de la courbe brute
    EPAISSEUR_LISSE = 2               #: Épaisseur de la courbe lissée
    FONTSIZE = 16                     #: Taille de police du titre principal

    # --- Libellés et titres ---
    TITRE_GRAPHE = "COVID-19 Bruxelles – {} : Page {}/{}"  #: Titre du graphique
    YLABEL = "Nombre de cas"                               #: Label axe Y
    LEGENDE_BRUT = "Données brutes"                        #: Légende brute
    LEGENDE_LISSE = "Lissage Savitzky-Golay"               #: Légende lissée
    LIBELLE_FONTSIZE = 12                                  #: Police des sous-titres
    ALIGNEMENT_HORIZONTAL = "center"
    ALIGNEMENT_VERTICAL = "top"
    XPOSITION = 0.5                                        #: Position X pour texte de fig.
    YPOSITION = 0.93                                       #: Position Y pour texte de fig.
    COULEUR_TEXTE = "#555"                                 #: Couleur texte info commune

    # --- Paramètres de grille/affichage ---
    FIGSIZE = (16, 10)         #: Taille de la figure (largeur, hauteur)
    GRAPHE_LIGNES = 2          #: Nombre de lignes de subplots par page
    GRAPHE_COLONNES = 3        #: Nombre de colonnes de subplots par page

    # --- Dates et axes X ---
    DATE_FORMAT = "%Y-%m"      #: Format affichage des dates sur X
    MOIS_INTERVAL = 6          #: Intervalle de mois entre labels de l'axe X
    ROTATION_LABELS = 45       #: Rotation des labels de dates

    # --- Position de la légende ---
    LEGENDE_POSITION = {"bbox_to_anchor": (0.18, 0.95)}

    # --- Suffixe à retirer pour l’affichage ---
    SUFFIXE_COMMUNE_BRUXELLES = "(Bruxelles-Capitale)"     #: Suffixe à masquer dans l’affichage

