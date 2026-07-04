#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

FICHIER_DONNEES = "data/eleves.csv"
DOSSIER_SORTIE = "resultats_question1"


# ============================================================================
# PARTIE MEMBRE 3 : CALCULS STATISTIQUES ET DETECTION DES VALEURS ATYPIQUES
# ============================================================================

def charger_donnees(chemin=FICHIER_DONNEES):

    df = pd.read_csv(chemin)
    return df


def calculer_statistiques(notes):
   
    q1 = notes.quantile(0.25)
    q3 = notes.quantile(0.75)

    stats = {
        "effectif": len(notes),
        "moyenne": notes.mean(),
        "mediane": notes.median(),
        "ecart_type": notes.std(ddof=1),
        "minimum": notes.min(),
        "maximum": notes.max(),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }
    return stats


def detecter_atypiques(df, colonne="note_evaluation"):

    notes = df[colonne]
    q1 = notes.quantile(0.25)
    q3 = notes.quantile(0.75)
    iqr = q3 - q1

    borne_basse = q1 - 1.5 * iqr
    borne_haute = q3 + 1.5 * iqr

    est_atypique = (notes < borne_basse) | (notes > borne_haute)
    eleves_atypiques = df[est_atypique][["eleve_id", colonne]].copy()
    eleves_atypiques = eleves_atypiques.sort_values(by=colonne)

    return eleves_atypiques, borne_basse, borne_haute


# ============================================================================
# PARTIE MEMBRE 4 : GRAPHIQUES (HISTOGRAMME + BOXPLOT)
# ============================================================================

def tracer_histogramme(notes, stats, dossier=DOSSIER_SORTIE):
   
    os.makedirs(dossier, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogramme
    ax.hist(notes, bins=15, color='#4C72B0', edgecolor='white', alpha=0.8)

    # Lignes de reference : moyenne et mediane
    ax.axvline(stats["moyenne"], color='red', linestyle='--', linewidth=2,
               label=f"Moyenne = {stats['moyenne']:.2f}")
    ax.axvline(stats["mediane"], color='green', linestyle='--', linewidth=2,
               label=f"Mediane = {stats['mediane']:.2f}")

    # Annotation Q1 et Q3
    ax.axvline(stats["q1"], color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(stats["q3"], color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(stats["q1"], ax.get_ylim()[1] * 0.9, f"Q1 = {stats['q1']:.1f}",
            ha='center', fontsize=9, color='orange')
    ax.text(stats["q3"], ax.get_ylim()[1] * 0.9, f"Q3 = {stats['q3']:.1f}",
            ha='center', fontsize=9, color='orange')

    ax.set_xlabel("Note d'evaluation (/20)", fontsize=12)
    ax.set_ylabel("Nombre d'eleves", fontsize=12)
    ax.set_title("Distribution des notes d'evaluation - Histogramme", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    chemin = os.path.join(dossier, "histogramme_notes.png")
    plt.savefig(chemin, dpi=150)
    plt.close()
    print(f"[OK] Histogramme sauvegarde : {chemin}")
    return chemin


def tracer_boxplot(notes, stats, eleves_atypiques, dossier=DOSSIER_SORTIE):
   
    os.makedirs(dossier, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Boxplot
    bp = ax.boxplot(notes, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='#4C72B0', color='black', alpha=0.8),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='black', linewidth=1.5),
                    capprops=dict(color='black', linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='#C44E52',
                                    markersize=8, linestyle='none', label='Valeurs atypiques'))

    ax.set_ylabel("Note d'evaluation (/20)", fontsize=12)
    ax.set_title("Distribution des notes - Boxplot (diagramme en boite)",
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels(["Tous les eleves"], fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Annotation : nombre de valeurs atypiques
    if len(eleves_atypiques) > 0:
        ax.annotate(f"{len(eleves_atypiques)} eleve(s) atypique(s)",
                    xy=(1.15, stats['maximum']), fontsize=10, color='#C44E52')

    plt.tight_layout()
    chemin = os.path.join(dossier, "boxplot_notes.png")
    plt.savefig(chemin, dpi=150)
    plt.close()
    print(f"[OK] Boxplot sauvegarde : {chemin}")
    return chemin


# ============================================================================
# PARTIE MEMBRE 4 : REDACTION POUR LE PROVISEUR (LANGAGE SIMPLE)
# ============================================================================

def rediger_reponse_proviseur(stats, eleves_atypiques, borne_basse, borne_haute):
   
    n_atypiques = len(eleves_atypiques)

    texte = f"""
{'='*70}
REPONSE AU PROVISEUR - QUESTION 1
{'='*70}

Monsieur le Proviseur,

Voici ce que revelent les resultats de l'evaluation de vos {stats['effectif']} eleves :

1. LES NOTES EN GENERAL
-----------------------
La moyenne de la classe est de {stats['moyenne']:.2f} sur 20. Cela signifie que,
globalement, vos eleves ont un niveau {'satisfaisant' if stats['moyenne'] >= 10 else 'a renforcer'}.

La note "typique" (mediane) est de {stats['mediane']:.2f} sur 20. La moitie des
eleves ont obtenu plus que ca, l'autre moitie moins.

La majorite des eleves (les 50% du milieu) ont des notes comprises entre
{stats['q1']:.2f} et {stats['q3']:.2f} sur 20. C'est la fourchette "normale"
de votre promotion.

2. LES ELEVES HORS DU LOT
-------------------------
{n_atypiques} eleve(s) sort(ent) vraiment de l'ordinaire :
"""

    if n_atypiques > 0:
        for _, row in eleves_atypiques.iterrows():
            eleve = row['eleve_id']
            note = row['note_evaluation']
            if note > stats['moyenne']:
                texte += f"   - {eleve} : {note:.1f}/20  -->  excellent resultat, au-dessus de la norme\n"
            else:
                texte += f"   - {eleve} : {note:.1f}/20  -->  resultat tres faible, a suivre de pres\n"

    texte += f"""
Ces eleves meritent une attention particuliere du conseil pedagogique,
dans un sens comme dans l'autre.

3. COMMENT PRESENTER CELA AU CONSEIL PEDAGOGIQUE
------------------------------------------------
Vous pouvez resumer la situation en deux phrases :

   "La moyenne de la classe est de {stats['moyenne']:.1f}/20. La majorite des
    eleves se situent entre {stats['q1']:.1f} et {stats['q3']:.1f}, avec {n_atypiques}
    eleve(s) dont le resultat sort notablement du lot."

4. LIMITES DE CETTE ANALYSE
---------------------------
Il est important de garder a l'esprit que :

- Cette analyse porte sur {stats['effectif']} eleves seulement. Elle donne une
  bonne idee de la promotion actuelle, mais ne peut pas etre generalisee
  a d'autres classes sans precaution.

- La moyenne est sensible aux eleves avec des notes tres hautes ou tres
  basses. Un seul eleve avec 2/20 ou 19/20 peut faire baisser ou monter
  la moyenne de toute la classe. C'est pourquoi on regarde aussi la
  mediane ({stats['mediane']:.2f}), qui est plus robuste.

- Cette analyse ne dit pas POURQUOI certains eleves ont de mauvaises
  notes. D'autres facteurs (assiduite, methode de travail, situation
  personnelle) seraient necessaires pour comprendre les causes.

- Les notes atypiques ne sont pas forcement des "erreurs" : elles
  peuvent refleter des profils reels d'eleves (tres forts ou en grande
  difficulte) qu'il faut accompagner differemment.
{'='*70}
"""
    return texte


# ============================================================================
# AFFICHAGE CONSOLE DES RESULTATS BRUTS
# ============================================================================

def afficher_resultats_console(stats, eleves_atypiques, borne_basse, borne_haute):
    """Affiche un resume lisible dans la console."""
    print("=" * 70)
    print("QUESTION 1 - Statistiques sur la note d'evaluation")
    print("=" * 70)
    print(f"Effectif            : {stats['effectif']} eleves")
    print(f"Moyenne             : {stats['moyenne']:.2f} / 20")
    print(f"Mediane             : {stats['mediane']:.2f} / 20")
    print(f"Ecart-type          : {stats['ecart_type']:.2f}")
    print(f"Minimum / Maximum   : {stats['minimum']:.2f} / {stats['maximum']:.2f}")
    print(f"Q1 (25%) / Q3 (75%) : {stats['q1']:.2f} / {stats['q3']:.2f}")
    print(f"Ecart interquartile : {stats['iqr']:.2f}")
    print("-" * 70)
    print(f"Bornes de detection (regle 1.5 x IQR) : ")
    print(f"  Atypique 'bas'  : note < {borne_basse:.2f}")
    print(f"  Atypique 'haut' : note > {borne_haute:.2f}")
    print(f"Nombre d'eleves atypiques : {len(eleves_atypiques)}")
    if len(eleves_atypiques) > 0:
        print("\nListe des eleves atypiques :")
        print(eleves_atypiques.to_string(index=False))
    print("=" * 70)


# ============================================================================
# POINT D'ENTREE PRINCIPAL
# ============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("INF232 - THEME D - QUESTION 1 : STATISTIQUE UNIVARIEE")
    print("=" * 70 + "\n")

    # --- Etape 1 : Chargement des donnees ---
    print("[1/5] Chargement des donnees...")
    df = charger_donnees()
    print(f"      {len(df)} eleves charges.")

    # --- Etape 2 : Calculs statistiques ---
    print("[2/5] Calcul des statistiques descriptives...")
    notes = df["note_evaluation"]
    stats = calculer_statistiques(notes)

    # --- Etape 3 : Detection des valeurs atypiques ---
    print("[3/5] Detection des valeurs atypiques (methode IQR)...")
    eleves_atypiques, borne_basse, borne_haute = detecter_atypiques(df)

    # --- Etape 4 : Graphiques ---
    print("[4/5] Generation des graphiques...")
    tracer_histogramme(notes, stats)
    tracer_boxplot(notes, stats, eleves_atypiques)

    # --- Etape 5 : Affichage et redaction ---
    print("[5/5] Redaction de la reponse...")
    print()
    afficher_resultats_console(stats, eleves_atypiques, borne_basse, borne_haute)

    # Redaction pour le proviseur
    texte_proviseur = rediger_reponse_proviseur(stats, eleves_atypiques, borne_basse, borne_haute)
    print(texte_proviseur)

    # Sauvegarde du texte dans un fichier
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    chemin_texte = os.path.join(DOSSIER_SORTIE, "reponse_proviseur_q1.txt")
    with open(chemin_texte, "w", encoding="utf-8") as f:
        f.write(texte_proviseur)
    print(f"[OK] Texte sauvegarde dans : {chemin_texte}")

    print("\n" + "=" * 70)
    print("[OK] Question 1 terminee avec succes !")
    print(f"     Fichiers generes dans le dossier : {DOSSIER_SORTIE}/")
    print("     - histogramme_notes.png")
    print("     - boxplot_notes.png")
    print("     - reponse_proviseur_q1.txt")
    print("=" * 70 + "\n")