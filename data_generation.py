#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import unicodedata
import os

# ============================================================================
# PARAMETRES CONFIGURABLES
# ============================================================================

# NOM DU CHEF DE GROUPE (a remplacer par le vrai nom)
# Format attendu : "Prenom Nom" (ex: "Jean Dupont")
NOM_CHEF_GROUPE = "Mfopit Mvu Maryam"

# Taille de l'echantillon (nombre d'eleves)
# 200 eleves est un bon compromis : suffisant pour le clustering et la classification
NB_ELEVES = 200

# Fichier de sortie
FICHIER_SORTIE = "data/eleves.csv"

# ============================================================================
# ETAPE 1 : GENERATION DE LA GRAINE A PARTIR DU NOM
# ============================================================================

def normaliser_nom(nom_complet):
  
    # Passer en majuscules
    nom = nom_complet.upper()
    
    # Supprimer les accents (decomposer les caracteres accentues)
    nom = unicodedata.normalize('NFD', nom)
    nom = ''.join(c for c in nom if unicodedata.category(c) != 'Mn')
    
    # Supprimer les espaces et tout caractere non alphabetique
    nom = ''.join(c for c in nom if c.isalpha())
    
    return nom


def nom_vers_graine(nom_normalise):

    graine = 0
    for position, caractere in enumerate(nom_normalise, start=1):
        code_ascii = ord(caractere)
        graine += code_ascii * position
    
    # On prend le resultat modulo 2**31 pour rester dans des bornes raisonnables
    # pour numpy (eviter les depassements)
    graine = graine % (2**31)
    
    return graine


# ============================================================================
# ETAPE 2 : GENERATION DES VARIABLES SCOLAIRES
# ============================================================================

def generer_donnees(graine, n=NB_ELEVES):
    
    # ------------------------------------------------------------------------
    # Initialisation du generateur pseudo-aleatoire avec la graine du groupe
    # ------------------------------------------------------------------------
    np.random.seed(graine)
    
    # ------------------------------------------------------------------------
    # Variable 1 : Note d'evaluation (sur 20)
    # ------------------------------------------------------------------------
    # On genere une distribution normale :
    # - moyenne = 12 (la moyenne de la classe est autour de 12/20)
    # - ecart-type = 3 (la majorite des notes entre 9 et 15)
    notes_brutes = np.random.normal(loc=12.0, scale=3.0, size=n)
    
    # Tronquer les valeurs hors bornes [2, 20]
    notes_brutes = np.clip(notes_brutes, 2.0, 20.0)
    
    # Arrondir a 1 decimale pour plus de realisme
    note_evaluation = np.round(notes_brutes, 1)
    
    # ------------------------------------------------------------------------
    # Variable 2 : Heures de travail personnel par semaine
    # ------------------------------------------------------------------------
    # On veut une correlation positive avec la note.
    # Formule : heures = base + coefficient * note + bruit_aleatoire
    # 
    # L'idee : un eleve qui travaille plus a tendance a avoir de meilleures notes,
    # mais ce n'est pas une relation parfaite (certains eleves travaillent peu
    # mais ont de bonnes notes, et inversement).
    
    base = 2.0                      # Minimum theorique d'heures
    coefficient = 0.8               # 0.8 heure de plus par point de note
    bruit = np.random.normal(0, 2.5, size=n)  # Bruit aleatoire (ecart-type 2.5)
    
    heures_brutes = base + coefficient * note_evaluation + bruit
    
    # Tronquer entre 0 et 25 heures (realiste pour des eleves)
    heures_brutes = np.clip(heures_brutes, 0.0, 25.0)
    
    # Arrondir a 1 decimale
    heures_travail = np.round(heures_brutes, 1)
    
    # ------------------------------------------------------------------------
    # Variable 3 : Orientation recommandee (scientifique / litteraire)
    # ------------------------------------------------------------------------
    # Le conseil de classe recommande l'orientation en fonction de :
    #   - La note (poids fort)
    #   - Les heures de travail (poids modere)
    # 
    # Hypothese pedagogique :
    #   - Filiere scientifique : eleves motives (bonnes notes + travail regulier)
    #   - Filiere litteraire   : eleves moins a l'aise en sciences ou moins travailleurs
    #
    # On calcule un "score scientifique" pour chaque eleve :
    #   score_scientifique = 0.6 * note + 0.04 * heures_travail + bruit
    
    bruit_orientation = np.random.normal(0, 1.5, size=n)
    score_scientifique = 0.6 * note_evaluation + 0.04 * heures_travail + bruit_orientation
    
    # ------------------------------------------------------------------------
    # Calcul AUTOMATIQUE du seuil pour garantir la repartition 55%/45%
    # ------------------------------------------------------------------------
    # On veut : 55% scientifique (score > seuil) et 45% litteraire (score <= seuil)
    # Le seuil optimal est le 45eme percentile du score_scientifique :
    #   - 45% des eleves ont un score <= seuil  -> litteraire
    #   - 55% des eleves ont un score > seuil   -> scientifique
    seuil = np.percentile(score_scientifique, 45)
    
    orientation = np.where(score_scientifique > seuil, "scientifique", "litteraire")
    
    # ------------------------------------------------------------------------
    # Assemblage dans un DataFrame
    # ------------------------------------------------------------------------
    df = pd.DataFrame({
        'eleve_id': [f"ELEVE_{i+1:03d}" for i in range(n)],
        'note_evaluation': note_evaluation,
        'heures_travail': heures_travail,
        'orientation': orientation
    })
    
    return df


# ============================================================================
# ETAPE 3 : SAUVEGARDE ET AFFICHAGE
# ============================================================================

def sauvegarder_donnees(df, chemin=FICHIER_SORTIE):
    """Sauvegarde le DataFrame au format CSV."""
    # Creer le dossier data/ s'il n'existe pas
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    df.to_csv(chemin, index=False, encoding='utf-8')
    print(f"[OK] Donnees sauvegardees dans : {chemin}")
    print(f"     Nombre d'eleves : {len(df)}")
    return chemin


def afficher_recap(df, graine, nom_chef):
    """Affiche un resume des donnees generees."""
    print("=" * 70)
    print("GENERATION DES DONNEES - RECAPITULATIF")
    print("=" * 70)
    print(f"Nom du chef de groupe : {nom_chef}")
    print(f"Nom normalise         : {normaliser_nom(nom_chef)}")
    print(f"Graine generee        : {graine}")
    print(f"Nombre d'eleves       : {len(df)}")
    print("-" * 70)
    print("\n--- EXTRAIT DES 10 PREMIERS ELEVES ---")
    print(df.head(10).to_string(index=False))
    print("\n--- STATISTIQUES DESCRIPTIVES ---")
    print(df[['note_evaluation', 'heures_travail']].describe().round(2).to_string())
    print("\n--- REPARTITION DES ORIENTATIONS ---")
    print(df['orientation'].value_counts().to_string())
    # Pourcentages
    repartition = df['orientation'].value_counts(normalize=True) * 100
    print("\n--- POURCENTAGES ---")
    for orient, pct in repartition.items():
        print(f"  {orient:<15s} : {pct:.1f}%")
    print("\n--- CORRELATION NOTE vs HEURES DE TRAVAIL ---")
    corr = df['note_evaluation'].corr(df['heures_travail'])
    print(f"Coefficient de correlation de Pearson : {corr:.3f}")
    print("=" * 70)


# ============================================================================
# POINT D'ENTREE PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 70)
    print("INF232 - THEME D : GENERATION DES DONNEES PERSONNALISEES")
    print("=" * 70 + "\n")
    
    # --- Etape 1 : Generation de la graine ---
    print("[1/3] Generation de la graine...")
    nom_normalise = normaliser_nom(NOM_CHEF_GROUPE)
    graine = nom_vers_graine(nom_normalise)
    print(f"      Nom original   : {NOM_CHEF_GROUPE}")
    print(f"      Nom normalise  : {nom_normalise}")
    print(f"      Graine obtenue : {graine}")
    print()
    
    # --- Etape 2 : Generation des donnees ---
    print(f"[2/3] Generation des donnees ({NB_ELEVES} eleves)...")
    df_eleves = generer_donnees(graine, n=NB_ELEVES)
    print("      Generation terminee.")
    print()
    
    # --- Etape 3 : Sauvegarde et affichage ---
    print("[3/3] Sauvegarde et affichage...")
    sauvegarder_donnees(df_eleves, FICHIER_SORTIE)
    print()
    afficher_recap(df_eleves, graine, NOM_CHEF_GROUPE)
    
    print("\n[OK] Generation terminee avec succes !")
    print(f"     Fichier CSV : {FICHIER_SORTIE}")