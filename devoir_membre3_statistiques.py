#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INF232 - Theme D - Question 1
Equipe 2 / Membre 3 : calculs statistiques et detection des valeurs atypiques

Ce script ne contient AUCUN tirage aleatoire : il lit toujours le meme
fichier eleves.csv (deja fige par la graine du groupe). Le resultat est
donc garanti identique a chaque execution, tant que le fichier ne change
pas. C'est ce qui rend les calculs "fiables" au sens du sujet.

Chaque fonction a une seule responsabilite, pour rester simple a lire,
a tester et a expliquer a l'oral.
"""

import pandas as pd

FICHIER_DONNEES = "eleves.csv"


# ----------------------------------------------------------------------
# 1. Chargement des donnees
# ----------------------------------------------------------------------
def charger_donnees(chemin=FICHIER_DONNEES):
    """Lit le fichier CSV et renvoie un DataFrame pandas."""
    df = pd.read_csv("data/eleves.csv")
    return df


# ----------------------------------------------------------------------
# 2. Calcul des indicateurs statistiques de base
#    Une fonction = un seul indicateur, pour rester simple a lire,
#    a tester et a expliquer individuellement.
# ----------------------------------------------------------------------
def calculer_moyenne(notes):
    """Moyenne arithmetique : somme des notes / nombre de notes."""
    return notes.mean()


def calculer_mediane(notes):
    """Valeur qui separe la serie triee en deux moities egales."""
    return notes.median()


def calculer_ecart_type(notes):
    """
    Ecart-type "echantillon" (ddof=1) : mesure la dispersion des notes
    autour de la moyenne. On utilise ddof=1 car nos 200 eleves sont
    consideres comme un echantillon, pas la population entiere.
    """
    return notes.std(ddof=1)


def calculer_quartiles(notes):
    """
    Renvoie (Q1, Q3) : les valeurs qui delimitent respectivement les
    25% de notes les plus basses et les 25% de notes les plus hautes.
    """
    q1 = notes.quantile(0.25)
    q3 = notes.quantile(0.75)
    return q1, q3


def calculer_ecart_interquartile(q1, q3):
    """Largeur de la moitie centrale de la distribution (Q3 - Q1)."""
    return q3 - q1


def calculer_statistiques(notes):
    """
    Fonction "chef d'orchestre" : appelle chaque fonction individuelle
    et rassemble tous les resultats dans un seul dictionnaire, pratique
    pour l'affichage et pour le rapport.
    """
    q1, q3 = calculer_quartiles(notes)

    stats = {
        "effectif": len(notes),
        "moyenne": calculer_moyenne(notes),
        "mediane": calculer_mediane(notes),
        "ecart_type": calculer_ecart_type(notes),
        "minimum": notes.min(),
        "maximum": notes.max(),
        "q1": q1,
        "q3": q3,
        "ecart_interquartile": calculer_ecart_interquartile(q1, q3),
    }
    return stats


# ----------------------------------------------------------------------
# 3. Detection des valeurs atypiques (methode IQR)
# ----------------------------------------------------------------------
def detecter_atypiques(df, colonne="note_evaluation"):
    """
    Detecte les valeurs atypiques avec la regle de l'ecart interquartile
    (regle de Tukey, 1.5 x IQR) : methode standard, simple a justifier.

    Un eleve est considere atypique si sa note est :
        - inferieure a  Q1 - 1.5 x IQR   (atypique "vers le bas")
        - ou superieure a  Q3 + 1.5 x IQR   (atypique "vers le haut")

    Renvoie :
        - un DataFrame des eleves atypiques
        - les deux bornes utilisees (basse, haute)
    """
    notes = df[colonne]
    q1 = notes.quantile(0.25)
    q3 = notes.quantile(0.75)
    iqr = q3 - q1

    borne_basse = q1 - 1.5 * iqr
    borne_haute = q3 + 1.5 * iqr

    est_atypique = (notes < borne_basse) | (notes > borne_haute)
    eleves_atypiques = df[est_atypique][["eleve_id", colonne]]

    return eleves_atypiques, borne_basse, borne_haute


# ----------------------------------------------------------------------
# 4. Affichage lisible des resultats
# ----------------------------------------------------------------------
def afficher_resultats(stats, atypiques, borne_basse, borne_haute):
    print("=" * 60)
    print("QUESTION 1 - Statistiques sur la note d'evaluation")
    print("=" * 60)
    print(f"Effectif            : {stats['effectif']} eleves")
    print(f"Moyenne             : {stats['moyenne']:.2f} / 20")
    print(f"Mediane             : {stats['mediane']:.2f} / 20")
    print(f"Ecart-type          : {stats['ecart_type']:.2f}")
    print(f"Minimum / Maximum   : {stats['minimum']:.2f} / {stats['maximum']:.2f}")
    print(f"Q1 (25%) / Q3 (75%) : {stats['q1']:.2f} / {stats['q3']:.2f}")
    print(f"Ecart interquartile : {stats['ecart_interquartile']:.2f}")
    print("-" * 60)
    print(f"Bornes de detection (regle 1.5 x IQR) : "
          f"< {borne_basse:.2f}  ou  > {borne_haute:.2f}")
    print(f"Nombre d'eleves atypiques : {len(atypiques)}")
    if len(atypiques) > 0:
        print(atypiques.to_string(index=False))
    print("=" * 60)


# ----------------------------------------------------------------------
# Point d'entree
# ----------------------------------------------------------------------
if __name__ == "__main__":
    df = charger_donnees()
    stats = calculer_statistiques(df["note_evaluation"])
    atypiques, borne_basse, borne_haute = detecter_atypiques(df)
    afficher_resultats(stats, atypiques, borne_basse, borne_haute)
