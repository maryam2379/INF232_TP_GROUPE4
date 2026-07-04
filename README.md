# INF232 - Theme D : Etablissement Scolaire Secondaire

Projet de statistique et analyse de donnees applique aux performances scolaires.

## Structure du projet

```
.
├── classification_supervisee.py
├── data
│   └── eleves.csv
├── data_generation.py
├── nuage_points_regression.png
├── question1.py
├── question2.py
├── question3.py
└── README.md
                          
```

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

## Mode d'emploi

1. Generer les donnees :
```bash
python data_generation.py
```

2. Lancer les analyses (chaque script est independant) :
```bash
python question1.py           # Q1 : Statistique univariee
python question2.py    # Q2 : Statistique bivariee
python question3.py  # Q3 : Clustering
python classification_supervisee.py    # Q4 : Classification supervisee
```
3. Lancer une fois:
```bash
python main.py
``` 

Chaque script lit `data/eleves.csv` et produit ses propres resultats et graphiques.

## Choix du langage

Python a ete choisi pour ses bibliotheques scientifiques (NumPy, Pandas,
Matplotlib, Scikit-learn) qui couvrent l'ensemble des methodes statistiques
requises par le sujet.
