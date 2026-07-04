#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

FICHIER_DONNEES = "data/eleves.csv"


# ----------------------------------------------------------------------
# 1. Chargement des donnees
# ----------------------------------------------------------------------
def charger_donnees(chemin=FICHIER_DONNEES):
    """Lit le fichier CSV et renvoie un DataFrame pandas."""
    df = pd.read_csv(chemin)
    return df


# ----------------------------------------------------------------------
# 2. Correlation entre les deux variables (Membre 5)
# ----------------------------------------------------------------------
def calculer_correlation(df, col_x="heures_travail", col_y="note_evaluation"):
  
    correlation, p_value = scipy_stats.pearsonr(df[col_x], df[col_y])
    return correlation, p_value


# ----------------------------------------------------------------------
# 3. Regression lineaire (Membre 5)
# ----------------------------------------------------------------------
def calculer_regression(df, col_x="heures_travail", col_y="note_evaluation"):
   
    pente, ordonnee, r_value, p_value, erreur_std = scipy_stats.linregress(
        df[col_x], df[col_y]
    )
    r_carre = r_value ** 2
    return {
        "pente": pente,
        "ordonnee_origine": ordonnee,
        "r_carre": r_carre,
        "erreur_std_pente": erreur_std,
        "p_value": p_value,
    }


def predire_note(heures, regression):
   
    return regression["ordonnee_origine"] + regression["pente"] * heures


# ----------------------------------------------------------------------
# 4. Analyse des residus pour juger de la fiabilite (Membre 6)
# ----------------------------------------------------------------------
def calculer_residus(df, regression, col_x="heures_travail", col_y="note_evaluation"):
   
    notes_predites = predire_note(df[col_x], regression)
    residus = df[col_y] - notes_predites
    ecart_type_residus = residus.std(ddof=1)
    return residus, ecart_type_residus


def determiner_zone_incertaine(df, col_x="heures_travail"):
    
    q1 = df[col_x].quantile(0.05)
    q3 = df[col_x].quantile(0.95)
    minimum = df[col_x].min()
    maximum = df[col_x].max()
    return minimum, q1, q3, maximum


# ----------------------------------------------------------------------
# 5. Graphique (nuage de points + droite de regression)
# ----------------------------------------------------------------------
def tracer_nuage_points(df, regression, col_x="heures_travail", col_y="note_evaluation",
                         fichier="nuage_points_regression.png"):
    plt.figure(figsize=(8, 6))
    plt.scatter(df[col_x], df[col_y], alpha=0.6, color="#4C72B0", label="Eleves")

    x_vals = np.linspace(df[col_x].min(), df[col_x].max(), 100)
    y_vals = predire_note(x_vals, regression)
    plt.plot(x_vals, y_vals, color="#C44E52", linewidth=2, label="Droite de regression")

    plt.title("Note d'evaluation en fonction des heures de travail personnel")
    plt.xlabel("Heures de travail personnel / semaine")
    plt.ylabel("Note d'evaluation / 20")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fichier, dpi=150)
    plt.close()
    print(f"[Info] Graphique sauvegarde : {fichier}")


# ----------------------------------------------------------------------
# 6. Redaction de la reponse pour le proviseur (Membre 6)
# ----------------------------------------------------------------------
def generer_texte_pour_proviseur(correlation, p_value_corr, regression,
                                  ecart_type_residus, bornes_x):
    minimum, q05, q95, maximum = bornes_x

    force = "forte" if abs(correlation) >= 0.5 else ("moderee" if abs(correlation) >= 0.3 else "faible")
    signification = (
        "cette relation est statistiquement significative"
        if p_value_corr < 0.05
        else "cette relation n'est pas statistiquement significative"
    )

    texte = (
        f"Oui, il existe bien un lien entre le temps de travail personnel et le resultat "
        f"a l'evaluation dans nos donnees : le coefficient de correlation est de "
        f"{correlation:.2f}, ce qui traduit une relation {force} entre les deux mesures "
        f"({signification}, p = {p_value_corr:.4f}).\n\n"
        f"En pratique, chaque heure de travail supplementaire par semaine est associee, "
        f"en moyenne, a une variation d'environ {regression['pente']:.2f} point sur la note. "
        f"Le modele explique {regression['r_carre']*100:.0f}% de la variation des notes "
        f"observees (R² = {regression['r_carre']:.2f}) : le reste s'explique par d'autres "
        f"facteurs non mesures ici (methode de travail, stress, aide exterieure, etc.).\n\n"
        f"Concretement, une estimation typique se trompe en moyenne d'environ "
        f"{ecart_type_residus:.1f} point sur 20 par rapport a la vraie note. C'est une "
        f"marge d'erreur a garder en tete avant d'utiliser cette estimation pour une "
        f"decision individuelle.\n\n"
        f"L'estimation reste raisonnable pour des eleves qui travaillent entre "
        f"{q05:.1f} et {q95:.1f} heures par semaine, soit la zone ou nous avons le plus "
        f"de donnees. En dehors de cette fourchette (moins de {minimum:.1f}h ou plus de "
        f"{maximum:.1f}h), l'estimation devient nettement plus incertaine, car on "
        f"extrapole a partir de tres peu d'eleves observes dans ces cas.\n\n"
        f"Limite importante : une bonne correlation ne signifie pas que le temps de "
        f"travail est la seule cause d'une bonne note, ni que forcer un eleve a travailler "
        f"plus d'heures garantirait mecaniquement une meilleure note. Cette estimation "
        f"doit rester un indicateur d'appui, pas une decision automatique."
    )
    return texte


# ----------------------------------------------------------------------
# 7. Affichage console des resultats bruts
# ----------------------------------------------------------------------
def afficher_resultats(correlation, p_value_corr, regression, ecart_type_residus, bornes_x):
    print("=" * 70)
    print("QUESTION 2 - Relation entre note d'evaluation et heures de travail")
    print("=" * 70)
    print(f"Coefficient de correlation (Pearson) : {correlation:.3f}")
    print(f"P-value de la correlation            : {p_value_corr:.4f}")
    print("-" * 70)
    print(f"Pente de la regression                : {regression['pente']:.3f}")
    print(f"Ordonnee a l'origine                   : {regression['ordonnee_origine']:.3f}")
    print(f"R² (variance expliquee)                : {regression['r_carre']:.3f}")
    print(f"P-value de la regression               : {regression['p_value']:.4f}")
    print("-" * 70)
    print(f"Ecart-type des residus (marge d'erreur type) : {ecart_type_residus:.2f}")
    minimum, q05, q95, maximum = bornes_x
    print(f"Zone fiable des heures de travail : [{q05:.1f} ; {q95:.1f}]")
    print(f"Bornes extremes observees          : [{minimum:.1f} ; {maximum:.1f}]")
    print("=" * 70)


# ----------------------------------------------------------------------
# Point d'entree
# ----------------------------------------------------------------------
if __name__ == "__main__":
    df = charger_donnees()

    correlation, p_value_corr = calculer_correlation(df)
    regression = calculer_regression(df)
    residus, ecart_type_residus = calculer_residus(df, regression)
    bornes_x = determiner_zone_incertaine(df)

    afficher_resultats(correlation, p_value_corr, regression, ecart_type_residus, bornes_x)
    tracer_nuage_points(df, regression)

    print("\n--- Texte pour le proviseur ---\n")
    print(generer_texte_pour_proviseur(correlation, p_value_corr, regression,
                                        ecart_type_residus, bornes_x))
