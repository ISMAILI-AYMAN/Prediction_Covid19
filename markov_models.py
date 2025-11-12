# -*- coding: utf-8 -*-

"""
markov_models.py
----------------

Note:
    - Toutes les étapes critiques, warnings, erreurs ou résumés sont uniquement 
      journalisés via logger passé en argument. Aucune sortie console directe.
    - Toutes les matrices manipulées sont supposées alignées en taille et en ordre ;
      leur cohérence doit être vérifiée en amont (voir matrix_data_loader.py).

Auteur   : Harry FRANCIS (2025)
Contact  : github.com/documentHarry

License  : Code source ouvert à la consultation.
           Toute redistribution ou usage commercial est soumis à accord écrit préalable.

Version  : 1.0.0 (2025-07-23)
"""

__version__ = "1.0.0"

# Compatible Python 3.9+ | Typage PEP 484 | PEP 257 | PEP 8 (88 caractères max)

# --- Librairies standards
import logging

# --- Librairies tiers
import numpy as np


from utils_log import log_debut_fin_logger_dynamique


@log_debut_fin_logger_dynamique("logger")
def estimer_matrice_de_transition_par_moindre_carre(
        X_t: np.ndarray, X_t1: np.ndarray, nbr_communes:int,
        logger: logging.Logger) -> np.ndarray:
    """
    Estime la matrice de transition d’un modèle Markov via la méthode des moindres
    carrés, à partir de données d’évolution de cas, avec régularisation et journalisation.

    Args:
        X_t (np.ndarray): Matrice des cas au temps t (n_communes x n_dates-1).
        X_t1 (np.ndarray): Matrice des cas au temps t+1 (n_communes x n_dates-1), 
            alignée sur X_t mais décalée d’un jour.
        nbr_communes (int): Nombre total de communes, utilisé pour la régularisation.
        logger (logging.Logger): Logger pour journaliser chaque étape et les erreurs.

    Returns:
        np.ndarray: Matrice de transition estimée (n_communes x n_communes), à utiliser
            pour modéliser ou prédire la propagation entre communes.

    Note:
        - Ajoute une faible régularisation (1e-6) sur la diagonale pour éviter 
          les inversions de matrices singulières (instables).
        - Le calcul exploite le produit matriciel NumPy (@).
        - Les dimensions et plages de valeurs sont tracées via logger.

    Théorie:
        Méthode mathématique avec la version LibreOffice:
            A = X_t1 * transpose(X_t) * inverse(X_t * transpose(X_t) + lambda * I)

    Example:
        >>> X_t = np.random.rand(19, 100)
        >>> X_t1 = np.random.rand(19, 100)
        >>> mat = estimer_matrice_de_transition_par_moindre_carre(X_t, X_t1, 19, logger)
        >>> logger.info(mat.shape)
        (19, 19)

    Étapes:
        1. Calcule X_t @ X_t.T puis ajoute la régularisation.
        2. Inverse la matrice obtenue (si possible).
        3. Applique la formule des moindres carrés pour estimer la matrice de transition.
        4. Trace les infos de dimension et valeurs via logger.

    Tips:
        - S’assurer que X_t et X_t1 couvrent les mêmes communes/dates pour de bons résultats.
        - En cas d’erreur LinAlgError : vérifier les données ou augmenter la régularisation.

    Utilisation:
        Utiliser pour obtenir une première matrice de transition à raffiner, par
        exemple avant d’appliquer des contraintes géographiques ou de la normalisation.

    Limitation:
        - Ne force ni la positivité, ni la normalisation des lignes (à faire après).
        - Suppose une dynamique linéaire entre t et t+1.

    See also:
        - appliquer_des_contraintes_geographiques
        - forcer_valeurs_positives_et_normaliser
        - https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html
    """
    logger.info("Estimation de la matrice de transition de base...")
    try:
        # Méthode des moindres carrés: A = X(t+1) * X(t)^T * (X(t) * X(t)^T)^(-1)
        # '@' permet d'effectuer le produit matriciel
        XtXt_T = X_t @ X_t.T
        # Pour éviter la singularité - Régularisation
        regularisation = 1e-6 * np.eye(nbr_communes)
        XtXt_T_regularisee = XtXt_T + regularisation

        # Inversion de la matrice
        XtXt_T_inversee = np.linalg.inv(XtXt_T_regularisee)

        # Estimation finale
        matrice_de_transition_de_base = X_t1 @ X_t.T @ XtXt_T_inversee
        logger.debug(
            f"Dimensions de la matrice estimée : "
            f"{matrice_de_transition_de_base.shape} | "
            f"Min={np.min(matrice_de_transition_de_base):.2f} | "
            f"Max={np.max(matrice_de_transition_de_base):.2f}"
        )
        return matrice_de_transition_de_base
    except np.linalg.LinAlgError as erreur_algorithmique:
        logger.error("Erreur d'inversion de matrice : %s", erreur_algorithmique,
                     exc_info = True)
        raise
    except Exception as exception:
        logger.error("Erreur inattendue lors de l'estimation LS : %s", exception,
                     exc_info = True)
        raise


@log_debut_fin_logger_dynamique("logger")
def appliquer_des_contraintes_geographiques(
        matrice_de_transition_de_base:np.ndarray, 
        matrice_de_poids_ponderes : np.ndarray,
        nbr_communes: int, logger: logging.Logger, alpha_geographique: float = 0.5,
        ) -> np.ndarray:
    """
    Applique un lissage géographique à une matrice de transition Markov en combinant
    la matrice de base et une version pondérée par les poids géographiques.

    Args:
        matrice_de_transition_de_base (np.ndarray): Matrice de transition initiale (n x n).
        matrice_de_poids_ponderes (np.ndarray): Matrice des poids géographiques (n x n).
        nbr_communes (int): Nombre de communes (dimensions des matrices).
        logger (logging.Logger): Logger pour tracer chaque étape.
        alpha_geographique (float): Poids relatif de la composante géographique 
            (de 0 à 1, défaut 0.5).

    Returns:
        np.ndarray: Nouvelle matrice de transition, lissée selon la géographie.

    Note:
        - alpha=0 : pas de lissage ; alpha=1 : 100% géographique.
        - Chaque ligne est rééchelonnée si sa somme absolue dépasse 2.0 (stabilité).
        - L’impact moyen géographique est tracé par logger.

    Example:
        >>> mat_geo = appliquer_des_contraintes_geographiques(mat, poids, 19, logger, 0.6)
        >>> logger.info(mat_geo.shape)
        (19, 19)

    Étapes:
        1. Calcule la matrice pondérée géographiquement : matrice_de_base * poids.
        2. Mélange linéaire des matrices selon alpha_geographique.
        3. Rééchelonne chaque ligne si besoin pour éviter des valeurs trop élevées.
        4. Trace l’impact moyen via logger.

    Tips:
        - Pour forcer une normalisation stricte, combiner avec forcer_valeurs_positives_et_normaliser.
        - Adapter alpha entre 0.3 et 0.7 pour un bon compromis entre stabilité et géographie.

    Utilisation:
        À appliquer après estimation moindres carrés, avant phase de test ou
        évaluation du modèle.

    Limitation:
        - Les matrices doivent être alignées (même ordre de communes).
        - L’ajustement de la somme à 2.0 est empirique.

    See also:
        - forcer_valeurs_positives_et_normaliser
        - https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    """
    logger.info(f"Application des contraintes géographiques "
                f"(alpha={alpha_geographique})")
    try:
        # Pondération géographique
        matrice_geo = matrice_de_transition_de_base * matrice_de_poids_ponderes
        # Combinaison linéaire
        matrice_final = (( 1 - alpha_geographique ) * matrice_de_transition_de_base 
                        + matrice_geo * alpha_geographique 
        )
        # Normalisation pour stabilité (optionnel)
        # Chaque ligne représente comment une commune va influencer les autres
        for index_commune in range(nbr_communes):
            nombre_de_lignes = np.sum(np.abs(matrice_final[index_commune, :]))
            # Pour éviter une divergence
            if nombre_de_lignes > 2.0:
                matrice_final[index_commune, :] = (
                    matrice_final[index_commune, :] / nombre_de_lignes * 1.5
                )

            logger.debug(f"Contraintes appliquées.\nImpact géographique moyen : "
                        f"{np.mean(matrice_final * matrice_de_poids_ponderes):.2f}")
            return matrice_final
    except Exception as exception:
        logger.error("Erreur lors de l'application des contraintes géographiques : %s", 
                     exception, exc_info = True)
        raise


@log_debut_fin_logger_dynamique("logger")
def esperance_maximisation_markov_avec_geo(
        X_t: np.ndarray, X_t1: np.ndarray, matrice_de_poids_geo: np.ndarray, 
        logger: logging.Logger, alpha_geo: float = 0.5, nbr_iteration:int = 50,
        total: float = 1e-6) -> np.ndarray:
    """
    Estime une matrice de transition Markov par un algorithme d’espérance-maximisation
    (EM), en appliquant un lissage géographique à chaque itération.

    Args:
        X_t (np.ndarray): Matrice des cas à t (n_communes x n_dates-1).
        X_t1 (np.ndarray): Matrice des cas à t+1 (n_communes x n_dates-1).
        matrice_de_poids_geo (np.ndarray): Matrice de poids géographiques (n x n).
        logger (logging.Logger): Logger pour journaliser le suivi et la convergence.
        alpha_geo (float): Intensité du lissage géographique à chaque itération (défaut 0.5).
        nbr_iteration (int): Nombre maximal d’itérations EM (défaut 50).
        total (float): Seuil de convergence (variation min pour stopper, défaut 1e-6).

    Returns:
        np.ndarray: Matrice de transition finale, positive et normalisée ligne par ligne.

    Note:
        - EM : alterne estimation (E-step) et maximisation (M-step).
        - À chaque itération, lissage géographique puis normalisation.
        - Convergence détectée si la variation entre deux itérations devient < total.

    Théorie:
        Algorithme EM ajusté :
        .. math::
            A^{new} = (1-\alpha) A^{EM} + \alpha A^{Geo}
        où A^{Geo} est la version pondérée de la matrice EM.

        Version LibreOffice:
            A_nouveau = (1 - alpha) * A_EM + alpha * (A_EM * poids_geo)

    Example:
        >>> mat_em = esperance_maximisation_markov_avec_geo(X_t, X_t1, poids, logger)
        >>> logger.info(mat_em.shape)
        (19, 19)

    Étapes:
        1. Initialise une matrice aléatoire positive et normalisée.
        2. Boucle : E-step, M-step, lissage géographique et normalisation.
        3. Teste la convergence (norme des différences < total).
        4. Arrête si convergé ou atteint nbr_iteration.

    Tips:
        - Utiliser alpha_geo < 0.7 pour limiter l’impact géographique.
        - Vérifier que X_t et X_t1 ne contiennent pas de valeurs négatives.

    Utilisation:
        Appeler pour estimer un modèle Markov plus robuste, ou comparer à la version moindres carrés.

    Limitation:
        - Les matrices doivent être non négatives et alignées.
        - Peut ne pas converger sur des données très bruitées ou incohérentes.

    See also:
        - estimer_matrice_de_transition_par_moindre_carre
        - https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm
    """
    nbr_communes, nbr_etapes = X_t.shape
    # Initialisation positive et normalisée
    matrice_finale = np.abs(np.random.rand(nbr_communes, nbr_communes))
    matrice_finale /= matrice_finale.sum(axis=1, keepdims=True)
    try: 
        for s in range(nbr_iteration):
            matrice_precedente = matrice_finale.copy()        
            ### E-step ###
            R = np.zeros((nbr_communes, nbr_communes, nbr_etapes))
            for index_etape in range(nbr_etapes):
                cas_predits_totaux  = (matrice_finale @ X_t[:, index_etape])
                # Éviter division par zéro
                cas_predits_totaux [cas_predits_totaux  == 0] = 1e-10  
                for index_commune in range(nbr_communes):
                    R[index_commune, :, index_etape] = (
                        matrice_finale[index_commune, :] * X_t[:, index_etape]
                        ) / cas_predits_totaux [index_commune]        
            ### M-step ###
            estimation_matrice_de_transition = np.zeros_like(matrice_finale)
            # Somme sur le temps pour chaque commune j
            cas_emis_totaux_par_j = np.sum(X_t, axis=1)        
            for index_commune in range(nbr_communes):
                for index_autre_commune in range(nbr_communes):
                    estimation_matrice_de_transition[index_commune, index_autre_commune] = ( 
                        np.sum(X_t1[index_commune, :] * R[index_commune, index_autre_commune, :])        
                    )
            # Mis à jour sans contraintes
            matrice_nouvelle = estimation_matrice_de_transition / (
                cas_emis_totaux_par_j[np.newaxis, :] + 1e-10)
            # Appliquer pondération géographique
            matrice_geo = matrice_nouvelle * matrice_de_poids_geo
            # Combinaison avec alpha_geo
            matrice_finale = (((1 - alpha_geo) * matrice_nouvelle) 
                              + alpha_geo * matrice_geo)
            # Contraintes : positivité + normalisation lignes
            matrice_finale = np.maximum(matrice_finale, 0)
            nombre_de_lignes = matrice_finale.sum(axis=1, keepdims=True)
            matrice_finale = matrice_finale / np.maximum(nombre_de_lignes, 1e-10)
            # Convergence
            difference = np.linalg.norm(matrice_finale - matrice_precedente)
            if difference < total:
                logger.info(f"Convergence atteinte en {s+1} itérations "
                            f"(diff={difference:.2e})")
                break    
        return matrice_finale
    except Exception as e:
        logger.error("Erreur dans l'EM Markov avec contrainte géo : %s", e, 
                     exc_info = True)
        raise


@log_debut_fin_logger_dynamique("logger")
def forcer_valeurs_positives_et_normaliser(
        matrice: np.ndarray, logger: logging.Logger, 
        total_souhaite_par_ligne: float = 1.0) -> np.ndarray:
    """
    Rend toutes les valeurs d’une matrice positives ou nulles puis normalise
    chaque ligne pour que sa somme soit égale à la valeur fixée (par défaut 1.0).

    Args:
        matrice (np.ndarray): Matrice à corriger (n_lignes x n_colonnes).
        logger (logging.Logger): Logger pour tracer la correction et les problèmes éventuels.
        total_souhaite_par_ligne (float): Valeur cible de la somme par ligne (défaut 1.0).

    Returns:
        np.ndarray: Nouvelle matrice, strictement positive et normalisée ligne par ligne.

    Note:
        - Toute ligne totalement nulle est remplacée pour éviter la division par zéro.
        - La matrice d’entrée n’est jamais modifiée : une nouvelle matrice est créée.

    Example:
        >>> mat_corr = forcer_valeurs_positives_et_normaliser(mat, logger, 1.0)
        >>> np.all(mat_corr >= 0)
        True
        >>> np.allclose(mat_corr.sum(axis=1), 1.0)
        True

    Étapes:
        1. Remplace toute valeur négative par zéro.
        2. Calcule la somme de chaque ligne.
        3. Remplace les sommes nulles par 1e-10 pour éviter la division par zéro.
        4. Divise chaque ligne par sa somme et multiplie par total_souhaite_par_ligne.
        5. Trace le succès via logger.

    Tips:
        - À appliquer après toute opération pouvant engendrer des négatifs ou un déséquilibre.
        - Indispensable avant l’évaluation ou la sauvegarde d’un modèle Markov.

    Utilisation:
        À appeler après toute étape de calcul de matrice de transition.

    Limitation:
        - Si une ligne est totalement nulle à l’entrée, elle le reste après correction.
        - La normalisation ne corrige pas la structure de dépendance des communes.

    See also:
        - appliquer_des_contraintes_geographiques
        - https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
    """

    
        # Remplacer les valeurs négatives par zéro
    try:
        matrice_avec_valeurs_positives = np.maximum(matrice, 0)
        # Faire la somme des lignes
        nombre_de_lignes = matrice_avec_valeurs_positives.sum(axis=1, keepdims=True)
        # Éviter la division par zéro en cas de ligne toute nulle
        nombre_de_lignes[nombre_de_lignes == 0] = 1e-10
        # Renormaliser avec un total de 1 par défaut
        matrice_normalisee = ( 
            matrice_avec_valeurs_positives / nombre_de_lignes * total_souhaite_par_ligne
        )
        logger.debug("Normalisation et positivité appliquées à la matrice.")
        return matrice_normalisee
    except Exception as e:
        logger.error("Erreur lors de la normalisation : %s", e, exc_info = True)
        raise


@log_debut_fin_logger_dynamique("logger")
def fusionner_les_modeles(
        matrice_par_moindre_carre: np.ndarray, 
        matrice_par_esperance_maximisation: np.ndarray, logger: logging.Logger,
        poids_moindre_carre: float = 0.5) -> np.ndarray:
    """
    Combine deux matrices de transition Markov (moindres carrés et EM) via une moyenne
    pondérée puis normalise chaque ligne pour garantir un modèle valide.

    Args:
        matrice_par_moindre_carre (np.ndarray): Matrice estimée par moindres carrés (n x n).
        matrice_par_esperance_maximisation (np.ndarray): Matrice estimée par EM (n x n).
        logger (logging.Logger): Logger pour journaliser la fusion.
        poids_moindre_carre (float): Poids du modèle moindres carrés (entre 0 et 1, défaut 0.5).

    Returns:
        np.ndarray: Matrice fusionnée et normalisée ligne par ligne.

    Note:
        - Le poids du modèle EM est calculé automatiquement (1 - poids_moindre_carre).
        - La normalisation des lignes est systématique après la fusion.

    Example:
        >>> mat = fusionner_les_modeles(mat_ls, mat_em, logger, poids_moindre_carre=0.7)
        >>> logger.info(mat.shape)
        (19, 19)

    Étapes:
        1. Calcule la moyenne pondérée des deux matrices.
        2. Calcule la somme des lignes (avec protection contre la division par zéro).
        3. Normalise chaque ligne pour que la somme soit égale à 1.
        4. Journalise le succès via logger.

    Tips:
        - Tester différents poids pour optimiser la performance finale du modèle.
        - Toujours vérifier que les matrices sont de même taille et alignées.

    Utilisation:
        À utiliser pour obtenir un modèle hybride (LS/EM) à tester ou valider.

    Limitation:
        - Les matrices doivent avoir exactement le même ordre de communes et les mêmes dimensions.

    See also:
        - generer_modeles_ponderes
        - https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    """
    try:
        poids_esperance_maximisation = 1 - poids_moindre_carre
        matrice_fusionnee = ( 
            poids_moindre_carre * matrice_par_moindre_carre + 
            poids_esperance_maximisation * matrice_par_esperance_maximisation
        )
        # Calcul de la somme des lignes (avec dimension conservée)
        somme_lignes = matrice_fusionnee.sum(axis=1, keepdims=True)
        # On évite la division par zéro
        somme_lignes = np.maximum(somme_lignes, 1e-10)
        # Normalisation ligne par ligne
        matrice_normalisee = matrice_fusionnee / somme_lignes
        logger.debug(f"Matrice fusionnée et normalisée "
                     f"(poids LS={poids_moindre_carre:.2f})")
        return matrice_normalisee
    except Exception as e:
        logger.error("Erreur lors de la fusion des modèles : %s", e, exc_info = True)
        raise
