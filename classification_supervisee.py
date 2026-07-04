#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)

# ======================================================================
# CONSTANTES (regroupees ici, aucune valeur en dur ailleurs dans le code)
# ======================================================================
DOSSIER_SCRIPT = os.path.dirname(os.path.abspath(__file__))
CHEMIN_DONNEES = os.path.join(DOSSIER_SCRIPT, "data", "eleves.csv")
DOSSIER_SORTIE = os.path.join(DOSSIER_SCRIPT, "prototype")

COLONNES_ENTREE = ["note_evaluation", "heures_travail"]
COLONNE_CIBLE = "orientation"
CLASSE_POSITIVE = "scientifique"

TAILLE_TEST = 0.25
GRAINE_ALEATOIRE = 42
NOM_MODELE = "Regression logistique"


# ======================================================================
# SECTION 1 - Chargement des donnees
# ======================================================================
def charger_donnees(chemin=CHEMIN_DONNEES):
    
    if not os.path.exists(chemin):
        raise FileNotFoundError(
            f"Fichier de donnees introuvable : '{chemin}'.\n"
            f"Verifie que 'data/eleves.csv' existe bien a cote de ce script, "
            f"et que M2 a regenere le fichier apres correction du seuil."
        )
    df = pd.read_csv(chemin)

    colonnes_attendues = COLONNES_ENTREE + [COLONNE_CIBLE]
    manquantes = [c for c in colonnes_attendues if c not in df.columns]
    if manquantes:
        raise ValueError(
            f"Colonnes manquantes dans '{chemin}' : {manquantes}."
        )
    return df


# ======================================================================
# SECTION 2 - Preparation des donnees (split + standardisation)
# ======================================================================
def preparer_donnees(df):
   
    X = df[COLONNES_ENTREE].values
    y = (df[COLONNE_CIBLE] == CLASSE_POSITIVE).astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TAILLE_TEST, random_state=GRAINE_ALEATOIRE, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test, scaler, X, y


# ======================================================================
# SECTION 3 - Entrainement du classifieur retenu
# ======================================================================
def entrainer_modele(X_train_s, y_train):
    
    modele = LogisticRegression()
    modele.fit(X_train_s, y_train)
    return modele


# ======================================================================
# SECTION 4 - Evaluation du modele
# ======================================================================
def evaluer_modele(modele, X_test_s, y_test):
   
    y_pred = modele.predict(X_test_s)
    matrice = confusion_matrix(y_test, y_pred)
    vn, fp, fn, vp = matrice.ravel()

    return {
        "matrice_confusion": matrice,
        "vp": int(vp), "vn": int(vn), "fp": int(fp), "fn": int(fn),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "rappel": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "taille_test": len(y_test),
    }


# ======================================================================
# SECTION 5 - Interpretation des coefficients du modele
# ======================================================================
def interpreter_coefficients(modele, scaler):
   
    coefficients_std = modele.coef_[0]
    ecarts_types = scaler.scale_

    phrases = []
    for nom, coef_std, ecart_type in zip(COLONNES_ENTREE, coefficients_std, ecarts_types):
        effet_par_unite = coef_std / ecart_type
        variation_pct = (effet_par_unite / 4) * 100
        sens = "augmente" if effet_par_unite > 0 else "diminue"
        phrases.append(
            f"Chaque unite supplementaire de '{nom}' {sens} la probabilite "
            f"predite d'orientation scientifique d'environ "
            f"{abs(variation_pct):.1f} points de pourcentage (autour du seuil de decision)."
        )
    return phrases


# ======================================================================
# SECTION 6 - Explications en langage clair
# ======================================================================
def generer_explications(resultats):
    
    vp, vn, fp, fn = resultats["vp"], resultats["vn"], resultats["fp"], resultats["fn"]
    n = resultats["taille_test"]
    nb_erreurs = fp + fn

    return {
        "matrice": (
            f"Sur {n} eleves testes (jamais vus pendant l'apprentissage), le "
            f"modele a bien classe {vp + vn} eleves et s'est trompe pour "
            f"{nb_erreurs} d'entre eux : {fn} eleves reellement 'scientifique' "
            f"ont ete predits 'litteraire', et {fp} eleves reellement "
            f"'litteraire' ont ete predits 'scientifique'."
        ),
        "accuracy": (
            f"Accuracy = {resultats['accuracy']:.1%} : sur 100 eleves, le "
            f"modele devine correctement l'orientation d'environ "
            f"{resultats['accuracy']*100:.0f} d'entre eux."
        ),
        "precision": (
            f"Precision = {resultats['precision']:.1%} : quand le modele "
            f"recommande la filiere scientifique, il a raison dans environ "
            f"{resultats['precision']*100:.0f}% des cas."
        ),
        "rappel": (
            f"Rappel = {resultats['rappel']:.1%} : parmi tous les eleves "
            f"reellement faits pour la filiere scientifique, le modele en "
            f"repere environ {resultats['rappel']*100:.0f}%."
        ),
        "conclusion": (
            "Ces resultats montrent une tendance utile mais imparfaite : le "
            "modele doit rester un outil d'aide a la decision pour le conseil "
            "de classe, jamais une decision automatique et definitive, car "
            f"{nb_erreurs} eleve(s) sur {n} recevraient ici une recommandation erronee."
        ),
    }


# ======================================================================
# SECTION 7 - Graphiques (generation, sauvegarde, affichage interactif)
# ======================================================================
def _attendre_entree(message):
    """Affiche un message et bloque jusqu'a ce que l'utilisateur appuie sur Entree."""
    print(f"\n{message}")
    input(">>> Appuie sur Entree pour afficher le graphique... ")


def tracer_regression_logistique(modele, scaler, X_brut, y_brut, dossier=DOSSIER_SORTIE):
   
    os.makedirs(dossier, exist_ok=True)

    x_min, x_max = X_brut[:, 0].min() - 1, X_brut[:, 0].max() + 1
    y_min, y_max = X_brut[:, 1].min() - 1, X_brut[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grille = np.c_[xx.ravel(), yy.ravel()]
    grille_std = scaler.transform(grille)
    Z = modele.predict(grille_std).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.contourf(xx, yy, Z, alpha=0.15, colors=["#c05621", "#2b6cb0"])

    couleurs = np.where(y_brut == 1, "#2b6cb0", "#c05621")
    ax.scatter(X_brut[:, 0], X_brut[:, 1], c=couleurs, edgecolor="white", s=45)

    ax.set_xlabel("Note d'evaluation (/20)")
    ax.set_ylabel("Heures de travail personnel / semaine")
    ax.set_title("Régression logistique : données réelles et frontière de décision",
                 fontsize=12, fontweight="bold")

    from matplotlib.patches import Patch
    legende = [
        Patch(facecolor="#2b6cb0", label="Scientifique (réel)"),
        Patch(facecolor="#c05621", label="Littéraire (réel)"),
    ]
    ax.legend(handles=legende, loc="upper left", fontsize=9)

    explication = (
        "Chaque point represente un eleve. Les zones colorees en fond montrent\n"
        "ce que le modele predirait pour n'importe quel nouvel eleve a cet endroit."
    )
    fig.text(0.5, -0.03, explication, ha="center", va="top", fontsize=8.8, color="#333333")

    plt.tight_layout()
    chemin = os.path.join(dossier, "1_regression_logistique.png")
    plt.savefig(chemin, dpi=150, bbox_inches="tight")

    _attendre_entree(
        "GRAPHIQUE 1/3 - Régression logistique sur les données réelles.\n"
        "Ce graphique montre chaque élève (note, heures) coloré selon son "
        "orientation réelle, avec la frontière apprise par le modèle en fond."
    )
    plt.show()
    plt.close()
    return chemin


def tracer_matrice_confusion(resultats, explications, dossier=DOSSIER_SORTIE):
   
    os.makedirs(dossier, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6.5))
    sns.heatmap(
        resultats["matrice_confusion"], annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Littéraire", "Scientifique"],
        yticklabels=["Littéraire", "Scientifique"],
        ax=ax, annot_kws={"fontsize": 16},
    )
    ax.set_xlabel("Prédiction du modèle")
    ax.set_ylabel("Orientation réelle")
    ax.set_title(f"Matrice de confusion - {NOM_MODELE}", fontsize=12, fontweight="bold", pad=12)

    texte = "\n".join([explications["matrice"], "", explications["accuracy"]])
    fig.text(0.5, -0.05, texte, wrap=True, ha="center", va="top", fontsize=8.5, color="#333333")

    plt.tight_layout()
    chemin = os.path.join(dossier, "2_matrice_confusion.png")
    plt.savefig(chemin, dpi=150, bbox_inches="tight")

    _attendre_entree(
        "GRAPHIQUE 2/3 - Matrice de confusion.\n"
        "Ce graphique croise les predictions du modele avec la realite, "
        "pour voir precisement ou il reussit et ou il se trompe."
    )
    plt.show()
    plt.close()
    return chemin


def tracer_barres_matrice(resultats, dossier=DOSSIER_SORTIE):
   
    os.makedirs(dossier, exist_ok=True)

    categories = [
        "Vrais Positifs\n(Sci. bien predits)",
        "Vrais Negatifs\n(Litt. bien predits)",
        "Faux Positifs\n(Litt. predits Sci.)",
        "Faux Negatifs\n(Sci. predits Litt.)",
    ]
    valeurs = [resultats["vp"], resultats["vn"], resultats["fp"], resultats["fn"]]
    couleurs = ["#2b6cb0", "#276749", "#c05621", "#9b2c2c"]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    barres = ax.bar(categories, valeurs, color=couleurs, edgecolor="white")

    for barre, valeur in zip(barres, valeurs):
        ax.text(barre.get_x() + barre.get_width() / 2, barre.get_height() + 0.3,
                str(valeur), ha="center", fontsize=13, fontweight="bold")

    ax.set_ylabel("Nombre d'élèves")
    ax.set_title("Répartition en barres de la matrice de confusion",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(valeurs) + 5)

    plt.xticks(fontsize=8.8)
    plt.tight_layout()
    chemin = os.path.join(dossier, "3_barres_matrice_confusion.png")
    plt.savefig(chemin, dpi=150, bbox_inches="tight")

    _attendre_entree(
        "GRAPHIQUE 3/3 - Répartition en barres.\n"
        "Cette vue en barres facilite la comparaison visuelle rapide entre "
        "les bonnes predictions et les deux types d'erreurs."
    )
    plt.show()
    plt.close()
    return chemin


# ======================================================================
# SECTION 8 - Sauvegarde du resume texte
# ======================================================================
def sauvegarder_resume(resultats, explications, coefficients_explication, dossier=DOSSIER_SORTIE):
    """Ecrit le fichier texte recapitulatif complet pour le rapport."""
    os.makedirs(dossier, exist_ok=True)
    chemin = os.path.join(dossier, "resume.txt")

    with open(chemin, "w", encoding="utf-8") as f:
        f.write("QUESTION 4 - CLASSIFICATION SUPERVISEE - RESUME\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Modele evalue : {NOM_MODELE}\n\n")

        f.write("--- CHIFFRES BRUTS ---\n")
        f.write(f"Accuracy  : {resultats['accuracy']:.3f}\n")
        f.write(f"Precision : {resultats['precision']:.3f}\n")
        f.write(f"Rappel    : {resultats['rappel']:.3f}\n")
        f.write(f"F1-score  : {resultats['f1']:.3f}\n\n")

        f.write("Matrice de confusion (lignes = reel, colonnes = predit) :\n")
        f.write("                  Predit Litteraire   Predit Scientifique\n")
        f.write(f"Reel Litteraire        {resultats['vn']:>4}                {resultats['fp']:>4}\n")
        f.write(f"Reel Scientifique      {resultats['fn']:>4}                {resultats['vp']:>4}\n\n")

        f.write("--- EXPLICATIONS EN LANGAGE CLAIR ---\n\n")
        f.write(explications["matrice"] + "\n\n")
        f.write(explications["accuracy"] + "\n")
        f.write(explications["precision"] + "\n")
        f.write(explications["rappel"] + "\n\n")
        f.write("Fiabilite et risque pedagogique :\n")
        f.write(explications["conclusion"] + "\n\n")

        f.write("--- INTERPRETATION DES COEFFICIENTS ---\n\n")
        for phrase in coefficients_explication:
            f.write(f"- {phrase}\n")

    return chemin


# ======================================================================
# SECTION 9 - Point d'entree principal
# ======================================================================
def main():
    print("\n" + "=" * 70)
    print("INF232 - THEME D - QUESTION 4 : CLASSIFICATION SUPERVISEE")
    print("=" * 70)

    try:
        df = charger_donnees()
    except (FileNotFoundError, ValueError) as erreur:
        print(f"\n[ERREUR] {erreur}")
        sys.exit(1)

    print(f"[OK] {len(df)} eleves charges depuis '{CHEMIN_DONNEES}'")

    X_train_s, X_test_s, y_train, y_test, scaler, X_brut, y_brut = preparer_donnees(df)
    print(f"[OK] Split effectue : {len(y_train)} entrainement / {len(y_test)} test")

    modele = entrainer_modele(X_train_s, y_train)
    print(f"[OK] Modele entraine : {NOM_MODELE}")

    resultats = evaluer_modele(modele, X_test_s, y_test)
    explications = generer_explications(resultats)
    coefficients_explication = interpreter_coefficients(modele, scaler)

    print("\n" + "-" * 70)
    print(f"Accuracy={resultats['accuracy']:.3f}  Precision={resultats['precision']:.3f}  "
          f"Rappel={resultats['rappel']:.3f}  F1={resultats['f1']:.3f}")
    print("-" * 70)

    os.makedirs(DOSSIER_SORTIE, exist_ok=True)

    tracer_regression_logistique(modele, scaler, X_brut, y_brut)
    tracer_matrice_confusion(resultats, explications)
    tracer_barres_matrice(resultats)

    sauvegarder_resume(resultats, explications, coefficients_explication)

    print("\n" + "=" * 70)
    print("[OK] Termine. Tous les resultats sont dans le dossier :")
    print(f"     {DOSSIER_SORTIE}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
