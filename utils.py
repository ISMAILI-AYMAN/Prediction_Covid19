# -*- coding: utf-8 -*-

"""
utils.py
--------
Outils utilitaires pour la manipulation, l’import/export et le traitement des données COVID-19
par commune, pour toute la Région Bruxelles-Capitale.

Ce module fournit :
    - Fonctions pour télécharger, charger et sauvegarder des fichiers JSON en toute sécurité.
    - Outils pour manipuler, filtrer et extraire des séries temporelles dans les structures de données.
    - Fonctions utilitaires pour gérer les logs, le formatage des dates, et la création de dossiers.
    - Listes et types utiles pour travailler avec les communes bruxelloises.

Objectif :
    - Centraliser tous les traitements et helpers génériques pour la gestion des données COVID-19.
    - Faciliter la réutilisation, la maintenance et la robustesse du projet grâce à un code factorisé.

Fonctionnalités principales :
    - Téléchargement robuste de données JSON/GeoJSON via URL (gestion des erreurs et logs).
    - Chargement et sauvegarde de fichiers JSON (avec gestion des dossiers et noms auto-uniques).
    - Extraction de séries temporelles pour une commune, conversion de dates, filtrage par période.
    - Création automatique des dossiers nécessaires à l’export de fichiers.
    - Logging personnalisé et redirection des warnings Python vers le système de logs.
    - Outils pour afficher la période d’une série ou filtrer des listes par date.

Prérequis :
    - Python 3.9+ ; dépendances : requests, json, logging, os, datetime (standards).
    - Les fonctions attendent parfois un logger (configuré par le projet appelant).

Philosophie :
    - Centralise tous les “petits outils” utiles pour éviter la redondance.
    - Favorise la robustesse (try/except/logging explicite) et la pédagogie (exemples, typage).

Utilisation typique :
    >>> from utils import charger_json, sauvegarder_json, extraire_serie, LISTE_COMMUNES
    >>> data = charger_json("data/covid.json")
    >>> dates, valeurs = extraire_serie(data, "Ixelles")

Best Practice :
    - Utilisez un logger explicite (configurer_logging) pour garder la trace des erreurs.
    - Centralisez tous les accès fichier ici, pas dans les scripts principaux.

Conseils :
    - Ajoutez vos propres helpers dans ce fichier au fur et à mesure des besoins du projet.
    - En cas d’automatisation, vérifiez l’existence des dossiers avant export (déjà géré ici !).

Limitations :
    - Ce module ne gère pas la visualisation graphique ni les algorithmes métier avancés.
    - Ne fait pas de validation poussée des structures JSON (à faire en amont si besoin).

Maintenance :
    - Toute nouvelle fonction utilitaire doit être ajoutée ici pour garder le projet propre.
    - Les logs sont présents pour vous aider à diagnostiquer tout problème I/O ou logique.

Documentation :
    - Toutes les fonctions sont documentées avec format Google, exemples, et exceptions.
    - Pour plus de détails sur les modules utilisés : voir la doc officielle Python (json, os, logging).

Auteur   : Harry FRANCIS (2025)
Contact  : github.com/documentHarry
Version  : 1.0.0 (2025-07-24)
License  : Code source ouvert à la consultation.
           Toute redistribution ou usage commercial est soumis à accord écrit préalable.
"""

__version__ = "1.0.0"

# Compatible Python 3.9+ | Typage PEP 484 | PEP 257 | PEP 8 (88 caractères max)

# --- Librairies standards






