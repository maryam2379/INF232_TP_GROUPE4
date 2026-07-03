# Thème D : Établissement Scolaire Secondaire (Groupe 04)

Ce projet a été réalisé dans le cadre des travaux pratiques de l'UE INF232 (Statistiques et Analyse de Données). Il implémente une chaîne complète d'analyse de données (statistique univariée, bivariée, clustering non supervisé et classification supervisée) appliquée aux performances et à l'orientation des élèves de terminale d'un établissement secondaire.

## 📋 Structure du Projet

```
├── data
│   ├── eleves.csv
│   └── README.md
└── data_generation.py

```

## 🛠️ Installation et Prérequis

L'application est développée en Python 3. Pour installer les dépendances nécessaires à l'exécution de l'ensemble de la chaîne, exécutez la commande suivante :

~~~bash 
pip install numpy pandas matplotlib seaborn scikit-learn
~~~

## 🚀 Mode d'emploi (Exécution de l'Application)

L'exécution complète de l'application se déroule en deux étapes simples :

### 1. Génération des données personnalisées

Générez le jeu de données déterministe basé sur le nom du chef de groupe (Mfopit Mvu Maryam) en lançant :

~~~bash 
python data_generation.py
~~~

### 2. Lancement de l'analyse globale

Pour exécuter l'application unifiée intégrant les traitements et les visualisations graphiques de toutes les équipes (univarié, bivarié, clustering, classification), lancez :

~~~bash
python main.py
~~~