#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import subprocess
import sys
import os

DOSSIER_SCRIPT = os.path.dirname(os.path.abspath(__file__))

ETAPES = [
    ("data_generation.py", "Generation des donnees"),
    ("question1.py", "Question 1 - Statistique univariee"),
    ("question2.py", "Question 2 - Statistique bivariee"),
    ("question3.py", "Question 3 - Clustering non supervise"),
    ("classification_supervisee.py", "Question 4 - Classification supervisee"),
]


def lancer_script(nom_fichier, description):
    chemin = os.path.join(DOSSIER_SCRIPT, nom_fichier)

    print("\n" + "=" * 70)
    print(f"ETAPE : {description}")
    print(f"Script : {nom_fichier}")
    print("=" * 70)

    if not os.path.exists(chemin):
        print(f"[ERREUR] Fichier introuvable : {chemin}")
        return False

    resultat = subprocess.run([sys.executable, chemin], cwd=DOSSIER_SCRIPT)

    if resultat.returncode != 0:
        print(f"\n[ERREUR] '{nom_fichier}' a echoue (code {resultat.returncode}).")
        return False

    print(f"\n[OK] '{nom_fichier}' termine avec succes.")
    return True


def main():
    print("\n" + "#" * 70)
    print("# INF232 - THEME D - LANCEMENT COMPLET DU PIPELINE")
    print("#" * 70)

    print(
        "\nATTENTION : classification_supervisee.py affiche 3 graphiques "
        "interactifs et attend un appui sur Entree entre chacun. "
        "Reste devant l'ecran pendant cette derniere etape."
    )

    for nom_fichier, description in ETAPES:
        succes = lancer_script(nom_fichier, description)
        if not succes:
            print("\n[ARRET] Le pipeline s'est arrete a cause d'une erreur.")
            sys.exit(1)

    print("\n" + "#" * 70)
    print("# TOUTES LES ETAPES SE SONT TERMINEES AVEC SUCCES")
    print("#" * 70)
    print(
        "\nResultats generes :\n"
        "  - data/eleves.csv\n"
        "  - resultats_question1/ (histogramme, boxplot, texte proviseur)\n"
        "  - nuage_points_regression.png (Q2)\n"
        "  - coude_clusters.png, silhouette_clusters.png, "
        "visualisation_clusters.png (Q3)\n"
        "  - prototype/ (matrices et resume Q4)\n"
    )


if __name__ == "__main__":
    main()