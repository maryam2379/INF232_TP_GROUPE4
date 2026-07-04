#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INF232 - Theme D - Question 3
Equipe 4 / Membre 7 : Application du clustering (k-means)

Ce script applique une classification non supervisée (k-means) sur les deux
mesures disponibles (note d'évaluation et heures de travail) pour identifier
des groupes naturels d'élèves aux profils proches.

Deux méthodes objectives sont utilisées pour déterminer le nombre optimal de
groupes :
    1. Méthode du coude (inertie)
    2. Indice de silhouette

Le script lit toujours le même fichier eleves.csv (figé par la graine du
groupe), donc le résultat est garanti identique à chaque exécution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# PARAMETRES CONFIGURABLES

FICHIER_DONNEES = "data/eleves.csv"
MAX_CLUSTERS = 10  # Nombre maximum de clusters à tester
RANDOM_STATE = 42  # Pour la reproductibilité de k-means


# 1. CHARGEMENT DES DONNEES

def charger_donnees(chemin=FICHIER_DONNEES):
    """Lit le fichier CSV et renvoie un DataFrame pandas."""
    df = pd.read_csv("data/eleves.csv")
    return df


# 2. PREPARATION DES DONNEES POUR LE CLUSTERING

def preparer_donnees_clustering(df):
    """
    Extrait les deux colonnes numériques (note_evaluation, heures_travail)
    et les standardise (moyenne = 0, écart-type = 1) pour que les deux
    variables aient la même importance dans le calcul des distances.
    
    Le standard scaling est nécessaire car les notes sont sur 20 et les
    heures de travail sur environ 25 : sans standardisation, la variable
    avec la plus grande variance dominerait la distance.
    """
    # Sélectionner les deux colonnes numériques
    X = df[['note_evaluation', 'heures_travail']].copy()
    
    # Standardisation (centrage et réduction)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler


# 3. METHODE DU COUDE - DETERMINATION DU NOMBRE OPTIMAL DE CLUSTERS

def methode_du_coude(X_scaled, max_clusters=MAX_CLUSTERS):
    """
    Calcule l'inertie (somme des carrés des distances intra-cluster) pour
    différents nombres de clusters. On cherche le "coude" : le point où
    l'ajout d'un cluster supplémentaire ne réduit plus significativement
    l'inertie.
    """
    inerties = []
    k_range = range(1, max_clusters + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        kmeans.fit(X_scaled)
        inerties.append(kmeans.inertia_)
    
    return k_range, inerties


def tracer_courbe_coude(k_range, inerties, fichier="coude_clusters.png"):
    """
    Trace la courbe de l'inertie en fonction du nombre de clusters et
    sauvegarde l'image.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inerties, 'bo-', color='#4C72B0', linewidth=2, markersize=8)
    plt.xlabel('Nombre de clusters (k)', fontsize=12)
    plt.ylabel('Inertie (somme des distances intra-cluster)', fontsize=12)
    plt.title("Méthode du coude - Détermination du nombre optimal de clusters", fontsize=14)
    plt.xticks(k_range)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fichier, dpi=150)
    plt.close()
    print(f"[Info] Graphique sauvegardé : {fichier}")
    return fichier


# 4. INDICE DE SILHOUETTE - CONFIRMATION DU NOMBRE OPTIMAL

def calculer_indices_silhouette(X_scaled, max_clusters=MAX_CLUSTERS):
    """
    Calcule l'indice de silhouette pour chaque nombre de clusters (à partir
    de k=2, car la silhouette n'est pas définie pour k=1). L'indice de
    silhouette mesure la cohésion des clusters : plus il est proche de 1,
    mieux les points sont regroupés.
    """
    silhouette_scores = []
    k_range = range(2, max_clusters + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
    
    return k_range, silhouette_scores


def tracer_silhouette(k_range, silhouette_scores, fichier="silhouette_clusters.png"):
    """
    Trace l'indice de silhouette en fonction du nombre de clusters et
    sauvegarde l'image.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, 's-', color='#C44E52', linewidth=2, markersize=8)
    plt.xlabel('Nombre de clusters (k)', fontsize=12)
    plt.ylabel('Indice de silhouette', fontsize=12)
    plt.title("Indice de silhouette - Validation du nombre de clusters", fontsize=14)
    plt.xticks(k_range)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fichier, dpi=150)
    plt.close()
    print(f"[Info] Graphique sauvegardé : {fichier}")
    return fichier


def determiner_nombre_optimal(k_range, silhouette_scores):
    """
    Détermine le nombre optimal de clusters comme celui qui maximise
    l'indice de silhouette.
    """
    meilleur_index = np.argmax(silhouette_scores)
    meilleur_k = k_range[meilleur_index]
    meilleur_score = silhouette_scores[meilleur_index]
    return meilleur_k, meilleur_score

# 5. EXECUTION DU CLUSTERING AVEC LE NOMBRE OPTIMAL DE CLUSTERS

def appliquer_clustering(X_scaled, n_clusters):
    """
    Applique l'algorithme k-means avec le nombre optimal de clusters
    déterminé précédemment.
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(X_scaled)
    centres = kmeans.cluster_centers_
    return labels, centres, kmeans


def visualiser_clusters(df, X_scaled, labels, centres, scaler,
                        fichier="visualisation_clusters.png"):
    """
    Visualise les clusters dans le plan (note, heures) avec les centres.
    Les points sont colorés par cluster et les centres sont marqués par
    une étoile blanche.
    
    Attention : Pour l'affichage, on utilise les données originales
    (non standardisées) pour que les axes soient lisibles.
    """
    # Récupérer les données originales (non standardisées)
    X_original = df[['note_evaluation', 'heures_travail']].values
    
    # Désstandardiser les centres pour les afficher sur les mêmes axes
    centres_original = scaler.inverse_transform(centres)
    
    plt.figure(figsize=(10, 8))
    
    # Palette de couleurs pour les clusters
    couleurs = ['#4C72B0', '#C44E52', '#55A868', '#8172B2', '#CCB974', '#64B5CD']
    
    # Tracer chaque cluster
    for cluster_id in range(len(centres)):
        mask = (labels == cluster_id)
        plt.scatter(X_original[mask, 0], X_original[mask, 1],
                   c=couleurs[cluster_id % len(couleurs)], 
                   label=f'Cluster {cluster_id + 1}',
                   alpha=0.6, s=60, edgecolors='white', linewidth=0.5)
    
    # Tracer les centres des clusters
    plt.scatter(centres_original[:, 0], centres_original[:, 1],
               c='white', s=200, marker='*', edgecolors='black',
               linewidth=1.5, zorder=10, label='Centres des clusters')
    
    plt.xlabel("Note d'évaluation (/20)", fontsize=12)
    plt.ylabel("Heures de travail personnel (/semaine)", fontsize=12)
    plt.title(f"Visualisation des {len(centres)} clusters d'élèves", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fichier, dpi=150)
    plt.close()
    print(f"[Info] Graphique sauvegardé : {fichier}")
    return fichier


# 6. STATISTIQUES DESCRIPTIVES PAR CLUSTER (pour le Membre 8)

def statistiques_par_cluster(df, labels):
    """
    Calcule les statistiques descriptives (moyenne, écart-type) pour
    chaque cluster sur les deux variables.
    """
    df_copy = df.copy()
    df_copy['cluster'] = labels
    
    stats = df_copy.groupby('cluster').agg({
        'note_evaluation': ['mean', 'std', 'min', 'max', 'count'],
        'heures_travail': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    return stats


def repartition_orientation_par_cluster(df, labels):
    """
    Calcule la répartition des orientations (scientifique/littéraire)
    dans chaque cluster.
    """
    df_copy = df.copy()
    df_copy['cluster'] = labels
    
    repartition = df_copy.groupby(['cluster', 'orientation']).size().unstack(fill_value=0)
    
    # Ajouter les pourcentages
    pourcentages = repartition.div(repartition.sum(axis=1), axis=0) * 100
    
    return repartition, pourcentages


def afficher_statistiques_clusters(stats, repartition, pourcentages):
    """
    Affiche un résumé des statistiques par cluster.
    """
    print("=" * 70)
    print("STATISTIQUES PAR CLUSTER")
    print("=" * 70)
    print("\n--- Statistiques descriptives (moyenne ± écart-type) ---")
    for cluster_id in stats.index:
        print(f"\nCluster {cluster_id + 1} ({int(stats.loc[cluster_id, ('note_evaluation', 'count')])} élèves) :")
        print(f"  Note moyenne      : {stats.loc[cluster_id, ('note_evaluation', 'mean')]:.2f} ± {stats.loc[cluster_id, ('note_evaluation', 'std')]:.2f}")
        print(f"  Note min / max    : {stats.loc[cluster_id, ('note_evaluation', 'min')]:.2f} / {stats.loc[cluster_id, ('note_evaluation', 'max')]:.2f}")
        print(f"  Heures moyenne    : {stats.loc[cluster_id, ('heures_travail', 'mean')]:.2f} ± {stats.loc[cluster_id, ('heures_travail', 'std')]:.2f}")
        print(f"  Heures min / max  : {stats.loc[cluster_id, ('heures_travail', 'min')]:.2f} / {stats.loc[cluster_id, ('heures_travail', 'max')]:.2f}")
    
    print("\n--- Répartition des orientations par cluster ---")
    print("\nEffectifs :")
    print(repartition.to_string())
    print("\nPourcentages :")
    print(pourcentages.round(1).to_string())


# 7. FONCTION PRINCIPALE

def analyser_clustering(df):
    """
    Fonction "chef d'orchestre" qui exécute toutes les étapes du clustering.
    """
    print("\n" + "=" * 70)
    print("QUESTION 3 - Classification non supervisée (clustering)")
    print("=" * 70)
    
    # --- Étape 1 : Préparation des données ---
    print("\n[1/6] Préparation des données pour le clustering...")
    X_scaled, scaler = preparer_donnees_clustering(df)
    print(f"      {len(X_scaled)} élèves, 2 variables standardisées")
    
    # --- Étape 2 : Méthode du coude ---
    print("\n[2/6] Calcul de l'inertie (méthode du coude)...")
    k_range, inerties = methode_du_coude(X_scaled)
    tracer_courbe_coude(k_range, inerties)
    print("      Courbe du coude tracée et sauvegardée")
    
    # --- Étape 3 : Indice de silhouette ---
    print("\n[3/6] Calcul de l'indice de silhouette...")
    k_range_sil, silhouette_scores = calculer_indices_silhouette(X_scaled)
    tracer_silhouette(k_range_sil, silhouette_scores)
    
    # Détermination du nombre optimal
    meilleur_k, meilleur_score = determiner_nombre_optimal(k_range_sil, silhouette_scores)
    print(f"      Nombre optimal de clusters : k = {meilleur_k}")
    print(f"      Indice de silhouette max    : {meilleur_score:.3f}")
    
    # --- Étape 4 : Application du clustering ---
    print(f"\n[4/6] Application de k-means avec k = {meilleur_k}...")
    labels, centres, kmeans = appliquer_clustering(X_scaled, meilleur_k)
    print(f"      Clustering terminé, {meilleur_k} clusters identifiés")
    
    # --- Étape 5 : Visualisation des clusters ---
    print("\n[5/6] Visualisation des clusters...")
    visualiser_clusters(df, X_scaled, labels, centres, scaler)
    print("      Visualisation sauvegardée")
    
    # --- Étape 6 : Statistiques par cluster ---
    print("\n[6/6] Calcul des statistiques par cluster...")
    stats = statistiques_par_cluster(df, labels)
    repartition, pourcentages = repartition_orientation_par_cluster(df, labels)
    afficher_statistiques_clusters(stats, repartition, pourcentages)
    
    # --- Résumé final ---
    print("\n" + "=" * 70)
    print("RÉSUMÉ DES CLUSTERS IDENTIFIÉS")
    print("=" * 70)
    for cluster_id in range(meilleur_k):
        n_eleves = int(stats.loc[cluster_id, ('note_evaluation', 'count')])
        note_moy = stats.loc[cluster_id, ('note_evaluation', 'mean')]
        heures_moy = stats.loc[cluster_id, ('heures_travail', 'mean')]
        pct_scientifique = pourcentages.loc[cluster_id, 'scientifique'] if 'scientifique' in pourcentages.columns else 0
        
        # Détection automatique du profil
        if note_moy >= 14 and heures_moy >= 14:
            profil = "Bosseurs excellents"
        elif note_moy >= 14 and heures_moy < 14:
            profil = "Élèves doués (peu de travail)"
        elif 10 <= note_moy < 14 and heures_moy >= 12:
            profil = "Travailleurs moyens"
        elif 10 <= note_moy < 14 and heures_moy < 12:
            profil = "Élèves moyens"
        elif note_moy < 10 and heures_moy >= 10:
            profil = "En difficulté malgré un effort"
        else:
            profil = "En grande difficulté"
        
        print(f"\nCluster {cluster_id + 1} : {profil}")
        print(f"  Effectif          : {n_eleves} élèves ({n_eleves/len(df)*100:.1f}%)")
        print(f"  Note moyenne      : {note_moy:.2f}/20")
        print(f"  Heures de travail : {heures_moy:.1f}h/semaine")
        print(f"  % scientifique    : {pct_scientifique:.1f}%")
    
    print("\n" + "=" * 70)
    
    # Retourner les résultats pour le Membre 8
    return {
        'labels': labels,
        'centres': centres,
        'meilleur_k': meilleur_k,
        'stats': stats,
        'repartition': repartition,
        'pourcentages': pourcentages
    }


# POINT D'ENTREE PRINCIPAL

if __name__ == "__main__":
    df = charger_donnees()
    resultats = analyser_clustering(df)
    
    print("\n[OK] Analyse de clustering terminée avec succès !")
    print("     Les fichiers graphiques suivants ont été générés :")
    print("     - coude_clusters.png")
    print("     - silhouette_clusters.png")
    print("     - visualisation_clusters.png")
    print("\n[Note] Le Membre 8 utilisera ces résultats pour caractériser")
    print("       les profils et rédiger la réponse au proviseur.")
