# Classification supervisée — Orientation scolaire (INF232, Thème D)

## Contexte

Ce projet s'inscrit dans le cadre du TP INF232 (Établissement scolaire secondaire, Thème D). La direction d'un établissement scolaire souhaite mieux comprendre le profil de ses élèves de terminale et évaluer dans quelle mesure une orientation (filière scientifique ou filière littéraire) peut être anticipée à partir de deux indicateurs simples : le résultat obtenu à une évaluation interne, et le nombre d'heures de travail personnel effectuées chaque semaine.

Ce dépôt correspond à la partie du projet portant sur la classification supervisée (Question 4 du sujet). Les trois autres volets du TP (statistique univariée, statistique bivariée, classification non supervisée) sont traités dans d'autres fichiers du groupe et ne sont pas couverts ici.

## Problème traité

À partir des deux mesures disponibles pour chaque élève, on cherche à construire un modèle capable de prédire l'orientation la plus probable. Ce n'est pas un exercice de prédiction parfaite : l'objectif est de mesurer à quel point cette prédiction est fiable, de comprendre ses limites, et de discuter du risque qu'il y aurait à s'appuyer dessus pour une vraie décision d'orientation.

## Approche retenue

Le choix s'est porté sur un seul classifieur, la régression logistique, plutôt que sur plusieurs modèles concurrents. Ce choix a été fait pour trois raisons : la relation entre les deux variables d'entrée et l'orientation semble progressive plutôt que basée sur des seuils brutaux, le modèle reste stable sur un échantillon de taille modeste (200 élèves) sans nécessiter de réglage particulier, et ses coefficients peuvent être traduits directement en une phrase compréhensible, ce qui est utile pour justifier les résultats devant un public non spécialiste comme un conseil pédagogique.

Les données sont séparées en un ensemble d'entraînement et un ensemble de test, ce dernier n'étant jamais utilisé pendant l'apprentissage du modèle, afin d'obtenir une évaluation honnête de sa capacité à généraliser à de nouveaux élèves.

## Fonctionnalités

- Chargement et validation du fichier de données du groupe
- Séparation entraînement/test et mise à l'échelle des variables
- Entraînement d'un modèle de régression logistique
- Évaluation par matrice de confusion, accuracy, précision, rappel et F1-score
- Traduction des coefficients du modèle en explications lisibles
- Génération d'explications en langage courant pour chaque métrique
- Trois visualisations : la frontière de décision du modèle sur les données réelles, la matrice de confusion, et une représentation en barres de cette même matrice
- Sauvegarde automatique des graphiques et d'un résumé texte destiné au rapport

## Technologies utilisées

Python 3, avec les bibliothèques pandas, scikit-learn, matplotlib et seaborn.

## Installation

```
pip install pandas scikit-learn matplotlib seaborn
```

## Utilisation

Le script attend un fichier `data/eleves.csv` situé au même niveau que lui. Une fois cette condition remplie, il suffit de l'exécuter :

```
python3 classification_supervisee.py
```

Le déroulement complet (chargement, entraînement, évaluation, graphiques, sauvegarde) est automatique et ne demande aucune autre intervention que de fermer chaque fenêtre de graphique au fur et à mesure qu'elle s'affiche.

## Résultats produits

L'exécution crée un dossier `prototype/` contenant les trois graphiques au format image ainsi qu'un fichier texte récapitulant les résultats chiffrés et leur interprétation. Ce dossier est pensé pour être réutilisé directement dans la rédaction du rapport final du groupe.

## Limites du modèle

Le modèle ne s'appuie que sur deux variables et ne peut donc pas rendre compte de la complexité réelle d'un choix d'orientation, qui dépend aussi de la motivation, des aptitudes spécifiques ou du contexte de chaque élève. Les données utilisées sont par ailleurs générées artificiellement pour les besoins du TP, et non observées dans un établissement réel. Pour ces raisons, les résultats produits par ce script doivent être compris comme une estimation d'aide à la décision, et non comme une recommandation d'orientation à appliquer telle quelle.

## Auteur

Kamdem Ndeffo Franck Emmanuel — Membre 9, Équipe 5, Groupe 4 — Université de Yaoundé I, L2 Informatique Systèmes et Réseaux.
