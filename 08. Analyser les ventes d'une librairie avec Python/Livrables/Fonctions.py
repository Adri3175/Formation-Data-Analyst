import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.express as px
import numpy as np
from datetime import datetime
from scipy.stats import chi2_contingency, spearmanr, pearsonr, shapiro, kruskal, mannwhitneyu, kstest
# ========================================
# Fonctions pour le calcul des indicateurs
# ========================================
def calcul_indicateurs(donnees):
    """
    Calcule et affiche les principaux indicateurs des données fournies.

    Paramètres :
    - donnees : DataFrame contenant les colonnes "price" et "session_id".
    
    Affiche :
    - Chiffre d'affaires total
    - Nombre de commandes
    - Nombre de produits vendus
    - Nombre moyen de produits par commande
    - Panier moyen
    """
    # Vérification des données
    if donnees.empty:
        print("Le DataFrame est vide. Impossible de calculer les indicateurs.")
        return
    
    colonnes_necessaires = {"price", "session_id"}
    if not colonnes_necessaires.issubset(donnees.columns):
        print(f"Les colonnes nécessaires sont manquantes. Colonnes attendues : {colonnes_necessaires}")
        return

    # Calculs
    try:
        ca_total = donnees["price"].sum()
        nombre_commandes = donnees["session_id"].nunique()
        nombre_produits_vendus = donnees["session_id"].count()
        produits_par_commande = nombre_produits_vendus / nombre_commandes
        panier_moyen = ca_total / nombre_commandes

        # Affichage
        print(f"- Le CA total est de {ca_total:.2f} €")
        print(f"- Nombre de commandes : {nombre_commandes}")
        print(f"- Nombre de produits vendus : {nombre_produits_vendus}")
        print(f"- Nombre moyen de produits vendus par commande : {produits_par_commande:.2f}")
        print(f"- Panier moyen : {panier_moyen:.2f} €")

    except Exception as e:
        print(f"Une erreur s'est produite lors du calcul des indicateurs : {e}")
# ========================================
# Fonctions pour l'analyse par année
# ========================================
def calcul_indicateurs_par_annee(donnees, debut_annee1, fin_annee1, debut_annee2, fin_annee2):
    """
    Calcule et affiche les indicateurs pour deux périodes commerciales et les évolutions entre elles.

    Paramètres :
    - donnees : DataFrame pandas contenant les colonnes "année", "mois", "price", "session_id".
    - debut_annee1, fin_annee1 : tuple (année, mois) définissant la période de la première année.
    - debut_annee2, fin_annee2 : tuple (année, mois) définissant la période de la deuxième année.

    Affiche :
    - Les indicateurs pour chaque année.
    - Les évolutions entre les deux périodes.
    """
    # Filtrer les données pour la première année
    donnees_annee1 = donnees[
        ((donnees["année"] == debut_annee1[0]) & (donnees["mois"] >= debut_annee1[1])) |
        ((donnees["année"] == fin_annee1[0]) & (donnees["mois"] <= fin_annee1[1]))
    ]

    # Filtrer les données pour la deuxième année
    donnees_annee2 = donnees[
        ((donnees["année"] == debut_annee2[0]) & (donnees["mois"] >= debut_annee2[1])) |
        ((donnees["année"] == fin_annee2[0]) & (donnees["mois"] <= fin_annee2[1]))
    ]

    # Fonction pour calculer les indicateurs
    def calcul_indicateurs2(donnees):
        ca_total = donnees["price"].sum()
        nombre_commandes = donnees["session_id"].nunique()
        nombre_produits_vendus = donnees["session_id"].count()
        produits_par_commande = nombre_produits_vendus / nombre_commandes if nombre_commandes > 0 else 0
        panier_moyen = ca_total / nombre_commandes if nombre_commandes > 0 else 0
        return ca_total, nombre_commandes, nombre_produits_vendus, produits_par_commande, panier_moyen

    # Calcul des indicateurs pour chaque année
    indicateurs_annee1 = calcul_indicateurs2(donnees_annee1)
    indicateurs_annee2 = calcul_indicateurs2(donnees_annee2)

    # Afficher les résultats
    print(f"Première année commerciale ({debut_annee1[1]}/{debut_annee1[0]} - {fin_annee1[1]}/{fin_annee1[0]}):")
    print(f"- CA total : {indicateurs_annee1[0]:.2f} €")
    print(f"- Nombre de commandes : {indicateurs_annee1[1]}")
    print(f"- Nombre de produits vendus : {indicateurs_annee1[2]}")
    print(f"- Nombre moyen de produits vendus par commande : {indicateurs_annee1[3]:.2f}")
    print(f"- Panier moyen : {indicateurs_annee1[4]:.2f} €")

    print(f"\nSeconde année commerciale ({debut_annee2[1]}/{debut_annee2[0]} - {fin_annee2[1]}/{fin_annee2[0]}):")
    print(f"- CA total : {indicateurs_annee2[0]:.2f} €")
    print(f"- Nombre de commandes : {indicateurs_annee2[1]}")
    print(f"- Nombre de produits vendus : {indicateurs_annee2[2]}")
    print(f"- Nombre moyen de produits vendus par commande : {indicateurs_annee2[3]:.2f}")
    print(f"- Panier moyen : {indicateurs_annee2[4]:.2f} €")

    # Calcul des évolutions
    def calcul_evolution(val1, val2):
        return ((val2 - val1) / val1 * 100) if val1 != 0 else 0

    evolutions = [
        calcul_evolution(indicateurs_annee1[i], indicateurs_annee2[i])
        for i in range(len(indicateurs_annee1))
    ]

    print("\nÉvolutions entre les deux années :")
    print(f"- Évolution du CA total : {evolutions[0]:.2f}%")
    print(f"- Évolution du nombre de commandes : {evolutions[1]:.2f}%")
    print(f"- Évolution du nombre de produits vendus : {evolutions[2]:.2f}%")
    print(f"- Évolution du nombre moyen de produits par commande : {evolutions[3]:.2f}%")
    print(f"- Évolution du panier moyen : {evolutions[4]:.2f}%")
# ========================================
# Fonction pour calculer les évolutions mensuelles sur deux années
# ========================================
def evolutions_mensuelles_sur_deux_annees(donnees):
    """
    Fonction pour calculer et afficher l'évolution mensuelle des indicateurs (CA, ventes, sessions, clients)
    sur la période disponible dans les données (de mars 2021 à février 2023).

    :param donnees: DataFrame contenant les données de ventes
    """
    # Créer la colonne 'date' pour faciliter le filtrage
    donnees['date'] = pd.to_datetime(
        donnees['année'].astype(str) + '-' +
        donnees['mois'].astype(str) + '-' +
        donnees['jour'].astype(str), 
        errors='coerce'
    )

    # Créer la colonne 'année_mois'
    donnees['année_mois'] = pd.to_datetime(donnees['année'].astype(str) + "-" + donnees['mois'].astype(str))

    # Filtrer les données pour la période de mars 2021 à février 2023
    mask = ((donnees['année_mois'] >= pd.to_datetime('2021-03-01')) & 
            (donnees['année_mois'] <= pd.to_datetime('2023-02-28')))
    donnees_filtrees = donnees[mask]

    # Calculer les métriques mensuelles
    ca_mensuel = donnees_filtrees.groupby(["année_mois"])["price"].sum()  
    ventes_mensuelles = donnees_filtrees.groupby(["année_mois"])["price"].count()  
    sessions_mensuelles = donnees_filtrees.groupby(["année_mois"])["session_id"].nunique()  
    clients_mensuels = donnees_filtrees.groupby(["année_mois"])["client_id"].nunique()  

    # Calculer les moyennes mobiles sur 3 mois
    ca_mensuel_moyenne_mobile = ca_mensuel.rolling(window=3).mean()
    ventes_mensuelles_moyenne_mobile = ventes_mensuelles.rolling(window=3).mean()
    sessions_mensuelles_moyenne_mobile = sessions_mensuelles.rolling(window=3).mean()
    clients_mensuels_moyenne_mobile = clients_mensuels.rolling(window=3).mean()

    # Créer les graphiques
    plt.figure(figsize=(15, 20))

    # Extraire les dates uniques pour l'affichage des étiquettes
    dates = ca_mensuel.index.strftime('%Y-%m')  # Convertir les dates au format "année-mois"

    # Graphique 1 : Évolution du CA mensuel
    plt.subplot(4, 1, 1)  # Premier graphique (4 lignes, 1 colonne, position 1)
    plt.plot(ca_mensuel.index, ca_mensuel.values, marker='o', color='#1f77b4', linewidth=2.5, label='CA réel')
    plt.plot(ca_mensuel_moyenne_mobile.index, ca_mensuel_moyenne_mobile.values, color='#ff7f0e', linestyle='--', linewidth=2.5, label='CA (moyenne mobile 3 mois)')
    plt.xticks(ticks=ca_mensuel.index, labels=dates, rotation=45, ha='right')
    plt.xlabel('Mois')
    plt.ylabel('CA (€)')
    plt.title('Évolution du CA mensuel', fontweight='bold')
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)

    # Graphique 2 : Évolution du nombre de ventes mensuelles
    plt.subplot(4, 1, 2)  # Deuxième graphique (4 lignes, 1 colonne, position 2)
    plt.plot(ventes_mensuelles.index, ventes_mensuelles.values, marker='o', color='#ff7f0e', linewidth=2.5, label='Ventes réelles')
    plt.plot(ventes_mensuelles_moyenne_mobile.index, ventes_mensuelles_moyenne_mobile.values, color='#1f77b4', linestyle='--', linewidth=2.5, label='Ventes (moyenne mobile 3 mois)')
    plt.xticks(ticks=ventes_mensuelles.index, labels=dates, rotation=45, ha='right')
    plt.xlabel('Mois')
    plt.ylabel('Nombre de ventes')
    plt.title('Évolution du nombre de ventes mensuelles', fontweight='bold')
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)

    # Graphique 3 : Évolution du nombre de sessions mensuelles
    plt.subplot(4, 1, 3)  # Troisième graphique (4 lignes, 1 colonne, position 3)
    plt.plot(sessions_mensuelles.index, sessions_mensuelles.values, marker='o', color='#2ca02c', linewidth=2.5, label='Sessions réelles')
    plt.plot(sessions_mensuelles_moyenne_mobile.index, sessions_mensuelles_moyenne_mobile.values, color='#d62728', linestyle='--', linewidth=2.5, label='Sessions (moyenne mobile 3 mois)')
    plt.xticks(ticks=sessions_mensuelles.index, labels=dates, rotation=45, ha='right')
    plt.xlabel('Mois')
    plt.ylabel('Nombre de sessions')
    plt.title('Évolution du nombre de sessions mensuelles', fontweight='bold')
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)

    # Graphique 4 : Évolution du nombre de clients mensuels
    plt.subplot(4, 1, 4)  # Quatrième graphique (4 lignes, 1 colonne, position 4)
    plt.plot(clients_mensuels.index, clients_mensuels.values, marker='o', color='#9467bd', linewidth=2.5, label='Clients réels')
    plt.plot(clients_mensuels_moyenne_mobile.index, clients_mensuels_moyenne_mobile.values, color='#8c564b', linestyle='--', linewidth=2.5, label='Clients (moyenne mobile 3 mois)')
    plt.xticks(ticks=clients_mensuels.index, labels=dates, rotation=45, ha='right')
    plt.xlabel('Mois')
    plt.ylabel('Nombre de clients uniques')
    plt.title('Évolution du nombre de clients mensuels', fontweight='bold')
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)

    # Afficher les graphiques
    plt.tight_layout()
    plt.show()
# ========================================
# Fonction pour afficher les évolutions mensuelles pour une année civile
# ========================================
def evolutions_mensuelles_annee_civile(donnees):
    """
    Fonction pour afficher l'évolution mensuelle des indicateurs (CA, ventes, sessions, clients)
    pour chaque année civile.

    :param donnees: DataFrame contenant les données de ventes
    """
    # Préparer les données pour le CA
    ca_mensuel_somme = donnees.groupby(['année', 'mois'])['price'].sum().reset_index()
    chaque_mois = np.arange(1, 13)
    donnees_ca_alignees = []

    for year in ca_mensuel_somme['année'].unique():
        sous_ensemble = ca_mensuel_somme[ca_mensuel_somme['année'] == year]
        sous_ensemble = sous_ensemble.set_index('mois').reindex(chaque_mois, fill_value=0).reset_index()
        sous_ensemble['année'] = year
        donnees_ca_alignees.append(sous_ensemble)

    donnees_ca_alignees = pd.concat(donnees_ca_alignees)

    # Préparer les données pour le nombre de ventes
    ventes_mensuelles = donnees.groupby(['année', 'mois'])['price'].count().reset_index()
    donnees_ventes_alignees = []

    for year in ventes_mensuelles['année'].unique():
        sous_ensemble = ventes_mensuelles[ventes_mensuelles['année'] == year]
        sous_ensemble = sous_ensemble.set_index('mois').reindex(chaque_mois, fill_value=0).reset_index()
        sous_ensemble['année'] = year
        donnees_ventes_alignees.append(sous_ensemble)

    donnees_ventes_alignees = pd.concat(donnees_ventes_alignees)

    # Préparer les données pour les sessions
    sessions_mensuelles = donnees.groupby(['année', 'mois'])['session_id'].nunique().reset_index()
    donnees_sessions_alignees = []

    for year in sessions_mensuelles['année'].unique():
        sous_ensemble = sessions_mensuelles[sessions_mensuelles['année'] == year]
        sous_ensemble = sous_ensemble.set_index('mois').reindex(chaque_mois, fill_value=0).reset_index()
        sous_ensemble['année'] = year
        donnees_sessions_alignees.append(sous_ensemble)

    donnees_sessions_alignees = pd.concat(donnees_sessions_alignees)

    # Préparer les données pour les clients
    clients_mensuels = donnees.groupby(['année', 'mois'])['client_id'].nunique().reset_index()
    donnees_clients_alignees = []

    for year in clients_mensuels['année'].unique():
        sous_ensemble = clients_mensuels[clients_mensuels['année'] == year]
        sous_ensemble = sous_ensemble.set_index('mois').reindex(chaque_mois, fill_value=0).reset_index()
        sous_ensemble['année'] = year
        donnees_clients_alignees.append(sous_ensemble)

    donnees_clients_alignees = pd.concat(donnees_clients_alignees)

    # Palette de couleurs personnalisée
    couleurs_personnalisees = ['#ff7f0e', '#1f77b4', '#2ca02c']
    bar_width = 0.32

    # Créer les graphiques
    plt.figure(figsize=(18, 20))

    # Graphique 1 : CA par mois et année
    plt.subplot(4, 1, 1)
    x = chaque_mois
    years = ca_mensuel_somme['année'].unique()

    for i, year in enumerate(years):
        sous_ensemble = donnees_ca_alignees[donnees_ca_alignees['année'] == year]
        plt.bar(
            x + i * bar_width,
            sous_ensemble['price'],
            width=bar_width,
            color=couleurs_personnalisees[i % len(couleurs_personnalisees)],
            label=str(year)
        )

    plt.xticks(
        ticks=x + bar_width * (len(years) - 1) / 2,
        labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    )
    plt.xlabel('Mois')
    plt.ylabel('CA (€)')
    plt.title('CA pour chaque mois', fontweight='bold')
    plt.legend(title='Année', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Graphique 2 : Nombre de ventes par mois et année
    plt.subplot(4, 1, 2)

    x = chaque_mois
    years = ventes_mensuelles['année'].unique()

    for i, year in enumerate(years):
        sous_ensemble = donnees_ventes_alignees[donnees_ventes_alignees['année'] == year]
        plt.bar(
            x + i * bar_width,
            sous_ensemble['price'],
            width=bar_width,
            color=couleurs_personnalisees[i % len(couleurs_personnalisees)],
            label=str(year)
        )

    plt.xticks(
        ticks=x + bar_width * (len(years) - 1) / 2,
        labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    )
    plt.xlabel('Mois')
    plt.ylabel('Nombre de ventes')
    plt.title('Nombre de ventes pour chaque mois', fontweight='bold')
    plt.legend(title='Année', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Graphique 3 : Nombre de sessions par mois et année
    plt.subplot(4, 1, 3)

    x = chaque_mois
    years = sessions_mensuelles['année'].unique()

    for i, year in enumerate(years):
        sous_ensemble = donnees_sessions_alignees[donnees_sessions_alignees['année'] == year]
        plt.bar(
            x + i * bar_width,
            sous_ensemble['session_id'],
            width=bar_width,
            color=couleurs_personnalisees[i % len(couleurs_personnalisees)],
            label=str(year)
        )

    plt.xticks(
        ticks=x + bar_width * (len(years) - 1) / 2,
        labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    )
    plt.xlabel('Mois')
    plt.ylabel('Nombre de sessions')
    plt.title('Nombre de sessions pour chaque mois', fontweight='bold')
    plt.legend(title='Année', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Graphique 4 : Nombre de clients par mois et année
    plt.subplot(4, 1, 4)

    x = chaque_mois
    years = clients_mensuels['année'].unique()

    for i, year in enumerate(years):
        sous_ensemble = donnees_clients_alignees[donnees_clients_alignees['année'] == year]
        plt.bar(
            x + i * bar_width,
            sous_ensemble['client_id'],
            width=bar_width,
            color=couleurs_personnalisees[i % len(couleurs_personnalisees)],
            label=str(year)
        )

    plt.xticks(
        ticks=x + bar_width * (len(years) - 1) / 2,
        labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    )
    plt.xlabel('Mois')
    plt.ylabel('Nombre de clients')
    plt.title('Nombre de clients pour chaque mois', fontweight='bold')
    plt.legend(title='Année', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Afficher les graphiques
    plt.tight_layout()
    plt.show()
# ========================================
# Fonction pour calculer les évolutions quotidiennes
# ========================================
def evolution_quotidienne_moyenne_mobile(donnees):
    # Création de la colonne "date"
    donnees['date'] = pd.to_datetime(
        donnees['année'].astype(str) + '-' +
        donnees['mois'].astype(str) + '-' +
        donnees['jour'].astype(str), 
        errors='coerce'
    )

    # Calcul des métriques journalières
    ca_quotidien = donnees.groupby("date")["price"].sum()  # Chiffre d'affaires quotidien
    ventes_quotidiennes = donnees.groupby("date")["price"].count()  # Nombre de ventes quotidiennes
    sessions_quotidiennes = donnees.groupby("date")["session_id"].nunique()  # Nombre de sessions quotidiennes
    clients_quotidiens = donnees.groupby("date")["client_id"].nunique()  # Nombre de clients quotidiens

    # Calcul des moyennes mobiles sur 7 jours
    ca_quotidien_moyenne_mobile = ca_quotidien.rolling(window=7).mean()
    ventes_quotidiennes_moyenne_mobile = ventes_quotidiennes.rolling(window=7).mean()
    sessions_quotidiennes_moyenne_mobile = sessions_quotidiennes.rolling(window=7).mean()
    clients_quotidiens_moyenne_mobile = clients_quotidiens.rolling(window=7).mean()

    # Extraire les mois uniques pour l'affichage sur l'axe X
    mois_labels = ca_quotidien.index.to_series().dt.to_period("M").unique()
    mois_positions = [ca_quotidien.index.get_loc(ca_quotidien[ca_quotidien.index.to_period("M") == mois].index[0]) for mois in mois_labels]

    # Création des graphiques
    plt.figure(figsize=(15, 20))

    # Graphique 1 : Évolution du CA quotidien
    plt.subplot(4, 1, 1)  # Premier graphique
    plt.plot(ca_quotidien.index, ca_quotidien.values, marker='o', color='#1f77b4', linewidth=1.5, label='CA réel')
    plt.plot(ca_quotidien_moyenne_mobile.index, ca_quotidien_moyenne_mobile.values, color='#ff7f0e', linestyle='--', linewidth=2.5, label='CA (moyenne mobile 7 jours)')
    plt.xticks(ticks=ca_quotidien.index[mois_positions], labels=[str(m) for m in mois_labels], rotation=45, ha='right')
    plt.xlabel('Date')
    plt.ylabel('CA (€)')
    plt.title('Évolution du CA quotidien', fontweight='bold')
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)

    # Graphique 2 : Évolution du nombre de ventes quotidiennes
    plt.subplot(4, 1, 2)  # Deuxième graphique
    plt.plot(ventes_quotidiennes.index, ventes_quotidiennes.values, marker='o', color='#ff7f0e', linewidth=1.5, label='Ventes réelles')
    plt.plot(ventes_quotidiennes_moyenne_mobile.index, ventes_quotidiennes_moyenne_mobile.values, color='#1f77b4', linestyle='--', linewidth=2.5, label='Ventes (moyenne mobile 7 jours)')
    plt.xticks(ticks=ventes_quotidiennes.index[mois_positions], labels=[str(m) for m in mois_labels], rotation=45, ha='right')
    plt.xlabel('Date')
    plt.ylabel('Nombre de ventes')
    plt.title('Évolution du nombre de ventes quotidiennes', fontweight='bold')
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)

    # Graphique 3 : Évolution du nombre de sessions quotidiennes
    plt.subplot(4, 1, 3)  # Troisième graphique
    plt.plot(sessions_quotidiennes.index, sessions_quotidiennes.values, marker='o', color='#2ca02c', linewidth=1.5, label='Sessions réelles')
    plt.plot(sessions_quotidiennes_moyenne_mobile.index, sessions_quotidiennes_moyenne_mobile.values, color='#d62728', linestyle='--', linewidth=2.5, label='Sessions (moyenne mobile 7 jours)')
    plt.xticks(ticks=sessions_quotidiennes.index[mois_positions], labels=[str(m) for m in mois_labels], rotation=45, ha='right')
    plt.xlabel('Date')
    plt.ylabel('Nombre de sessions')
    plt.title('Évolution du nombre de sessions quotidiennes', fontweight='bold')
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)

    # Graphique 4 : Évolution du nombre de clients quotidiens
    plt.subplot(4, 1, 4)  # Quatrième graphique
    plt.plot(clients_quotidiens.index, clients_quotidiens.values, marker='o', color='#9467bd', linewidth=1.5, label='Clients réels')
    plt.plot(clients_quotidiens_moyenne_mobile.index, clients_quotidiens_moyenne_mobile.values, color='#8c564b', linestyle='--', linewidth=2.5, label='Clients (moyenne mobile 7 jours)')
    plt.xticks(ticks=clients_quotidiens.index[mois_positions], labels=[str(m) for m in mois_labels], rotation=45, ha='right')
    plt.xlabel('Date')
    plt.ylabel('Nombre de clients uniques')
    plt.title('Évolution du nombre de clients quotidiens', fontweight='bold')
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)

    # Afficher les graphiques
    plt.tight_layout()
    plt.show()
# ========================================
# Fonction pour afficher un zoom sur le mois d'octobre 2021
# ========================================
def zoom_octobre_2021(donnees):
    # Filtrer les données pour septembre, octobre et novembre 2021, ainsi que septembre, octobre et novembre 2022
    mois_2021 = donnees[(donnees['année'] == 2021) & (donnees['mois'].isin([9, 10, 11]))]
    mois_2022 = donnees[(donnees['année'] == 2022) & (donnees['mois'].isin([9, 10, 11]))].copy()

    # Normaliser les dates de 2022 pour correspondre aux jours de septembre, octobre et novembre 2021
    mois_2022['date_normalisee'] = mois_2022['date'].apply(lambda x: x.replace(year=2021))

    # Calculer les métriques pour septembre, octobre et novembre 2021
    ca_journalier_2021 = mois_2021.groupby("date")["price"].sum().reset_index()
    ventes_journalieres_2021 = mois_2021.groupby("date")["price"].count().reset_index()

    # Calculer les métriques pour septembre, octobre et novembre 2022 avec les dates normalisées
    ca_journalier_2022 = mois_2022.groupby("date_normalisee")["price"].sum().reset_index()
    ventes_journalieres_2022 = mois_2022.groupby("date_normalisee")["price"].count().reset_index()

    # Calculer la moyenne mobile sur 7 jours pour 2022 (exemple de fenêtre 7 jours)
    ca_journalier_2022['ca_moyenne_mobile'] = ca_journalier_2022['price'].rolling(window=7, min_periods=1).mean()
    ventes_journalieres_2022['ventes_moyenne_mobile'] = ventes_journalieres_2022['price'].rolling(window=7, min_periods=1).mean()

    # Ajouter le calcul du nombre de sessions et du nombre de clients pour 2021 et 2022
    sessions_journalieres_2021 = mois_2021.groupby("date")["session_id"].nunique().reset_index() 
    clients_journalieres_2021 = mois_2021.groupby("date")["client_id"].nunique().reset_index()  

    sessions_journalieres_2022 = mois_2022.groupby("date_normalisee")["session_id"].nunique().reset_index()
    clients_journalieres_2022 = mois_2022.groupby("date_normalisee")["client_id"].nunique().reset_index()

    # Calculer les moyennes mobiles sur 7 jours pour les sessions et les clients en 2022
    sessions_journalieres_2022['sessions_moyenne_mobile'] = sessions_journalieres_2022['session_id'].rolling(window=7, min_periods=1).mean()
    clients_journalieres_2022['clients_moyenne_mobile'] = clients_journalieres_2022['client_id'].rolling(window=7, min_periods=1).mean()

    # Créer une figure avec quatre graphiques, tous alignés verticalement (1 colonne, 4 lignes)
    plt.figure(figsize=(18, 24))  # Plus grand pour s'assurer que les graphiques sont espacés correctement

    # Graphique 1 : Évolution du CA quotidien (septembre, octobre, novembre 2021 vs moyenne mobile 2022)
    plt.subplot(4, 1, 1)
    plt.plot(
        ca_journalier_2021['date'], 
        ca_journalier_2021['price'], 
        marker='o', 
        color='#1f77b4', 
        linewidth=2.5, 
        label="CA quotidien (Sept, Oct, Nov 2021)"
    )
    plt.plot(
        ca_journalier_2022['date_normalisee'], 
        ca_journalier_2022['ca_moyenne_mobile'], 
        marker='o', 
        color='#ff7f0e', 
        linewidth=2.5, 
        linestyle='--', 
        label="Moyenne mobile CA quotidien (Sept, Oct, Nov 2022)"
    )
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Chiffre d\'Affaires (€)', fontsize=12)
    plt.title('Évolution du CA quotidien (Sept, Oct, Nov 2021 vs Moyenne mobile 2022)', fontweight='bold', fontsize=14)
    plt.xticks(
        ticks=ca_journalier_2021['date'][::7],  # Choisir un point tous les 7 jours pour l'axe X
        labels=ca_journalier_2021['date'].dt.strftime('%Y-%m-%d')[::7],  # Afficher la date sous format 'yyyy-mm-dd'
        rotation=45,
        ha='right'
    )
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)
    plt.legend(fontsize=10)

    # Graphique 2 : Évolution du nombre de ventes quotidiennes (septembre, octobre, novembre 2021 vs moyenne mobile 2022)
    plt.subplot(4, 1, 2)
    plt.plot(
        ventes_journalieres_2021['date'], 
        ventes_journalieres_2021['price'], 
        marker='o', 
        color='#ff7f0e', 
        linewidth=2.5, 
        label="Nombre de ventes quotidiennes (Sept, Oct, Nov 2021)"
    )
    plt.plot(
        ventes_journalieres_2022['date_normalisee'], 
        ventes_journalieres_2022['ventes_moyenne_mobile'], 
        marker='o', 
        color='#1f77b4', 
        linewidth=2.5, 
        linestyle='--', 
        label="Moyenne mobile Ventes quotidiennes (Sept, Oct, Nov 2022)"
    )
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Nombre de ventes', fontsize=12)
    plt.title('Évolution du nombre de ventes quotidiennes (Sept, Oct, Nov 2021 vs Moyenne mobile 2022)', fontweight='bold', fontsize=14)
    plt.xticks(
        ticks=ventes_journalieres_2021['date'][::7],  # Choisir un point tous les 7 jours pour l'axe X
        labels=ventes_journalieres_2021['date'].dt.strftime('%Y-%m-%d')[::7],  # Afficher la date sous format 'yyyy-mm-dd'
        rotation=45,
        ha='right'
    )
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)
    plt.legend(fontsize=10)

    # Graphique 3 : Évolution du nombre de sessions quotidiennes avec moyenne mobile
    plt.subplot(4, 1, 3)
    plt.plot(
        sessions_journalieres_2021['date'], 
        sessions_journalieres_2021['session_id'], 
        marker='o', 
        color='#2ca02c', 
        linewidth=2.5, 
        label="Nombre de sessions quotidiennes (Sept, Oct, Nov 2021)"
    )
    plt.plot(
        sessions_journalieres_2022['date_normalisee'], 
        sessions_journalieres_2022['sessions_moyenne_mobile'], 
        marker='o', 
        color='#d62728', 
        linewidth=2.5, 
        linestyle='--', 
        label="Moyenne mobile Sessions quotidiennes (Sept, Oct, Nov 2022)"
    )
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Nombre de sessions', fontsize=12)
    plt.title('Évolution du nombre de sessions quotidiennes (Sept, Oct, Nov 2021 vs Moyenne mobile 2022)', fontweight='bold', fontsize=14)
    plt.xticks(
        ticks=sessions_journalieres_2021['date'][::7],  # Choisir un point tous les 7 jours pour l'axe X
        labels=sessions_journalieres_2021['date'].dt.strftime('%Y-%m-%d')[::7],  # Afficher la date sous format 'yyyy-mm-dd'
        rotation=45,
        ha='right'
    )
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)
    plt.legend(fontsize=10)

    # Graphique 4 : Évolution du nombre de clients quotidiens avec moyenne mobile
    plt.subplot(4, 1, 4)
    plt.plot(
        clients_journalieres_2021['date'], 
        clients_journalieres_2021['client_id'], 
        marker='o', 
        color='#9467bd', 
        linewidth=2.5, 
        label="Nombre de clients quotidiens (Sept, Oct, Nov 2021)"
    )
    plt.plot(
        clients_journalieres_2022['date_normalisee'], 
        clients_journalieres_2022['clients_moyenne_mobile'], 
        marker='o', 
        color='#8c564b', 
        linewidth=2.5, 
        linestyle='--', 
        label="Moyenne mobile Clients quotidiens (Sept, Oct, Nov 2022)"
    )
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Nombre de clients', fontsize=12)
    plt.title('Évolution du nombre de clients quotidiens (Sept, Oct, Nov 2021 vs Moyenne mobile 2022)', fontweight='bold', fontsize=14)
    plt.xticks(
        ticks=clients_journalieres_2021['date'][::7],  # Choisir un point tous les 7 jours pour l'axe X
        labels=clients_journalieres_2021['date'].dt.strftime('%Y-%m-%d')[::7],  # Afficher la date sous format 'yyyy-mm-dd'
        rotation=45,
        ha='right'
    )
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)
    plt.legend(fontsize=10)

    # Ajuster l'espacement entre les graphiques
    plt.tight_layout()
    plt.show()
# ========================================
# Fonction pour afficher les TOP10
# ========================================
def top_10(donnees, type_resultat="clients"):
    # Palette de couleurs
    palette = sns.color_palette("Set2", 3)

    if type_resultat == "clients":
        # ---- Top 10 Clients ----
        
        # CA par client
        ca_par_client_categ = donnees.groupby(["client_id", "categ"])["price"].sum().unstack(fill_value=0)
        top_10_clients_ca = ca_par_client_categ.sum(axis=1).sort_values(ascending=False).head(10)
        ca_par_client_categ_top10 = ca_par_client_categ.loc[top_10_clients_ca.index]
        
        # Produits achetés par client
        produits_par_client_categ = donnees.groupby(["client_id", "categ"])["id_prod"].count().unstack(fill_value=0)
        top_10_clients_produits = produits_par_client_categ.sum(axis=1).sort_values(ascending=False).head(10)
        produits_par_client_categ_top10 = produits_par_client_categ.loc[top_10_clients_produits.index]
        
        # Sessions par client
        sessions_par_client_categ = donnees.groupby(["client_id", "categ"])["session_id"].nunique().unstack(fill_value=0)
        top_10_clients_sessions = sessions_par_client_categ.sum(axis=1).sort_values(ascending=False).head(10)
        sessions_par_client_categ_top10 = sessions_par_client_categ.loc[top_10_clients_sessions.index]

        # Visualisation pour les Clients (disposition verticale)
        fig, axes = plt.subplots(3, 1, figsize=(15, 15))  # 3 lignes, 1 colonne

        # CA par client
        ca_par_client_categ_top10.plot(kind="bar", stacked=True, ax=axes[0], color=palette, edgecolor="black")
        axes[0].set_title("Top 10 des clients par chiffre d'affaires", fontsize=16, fontweight="bold")
        axes[0].set_xlabel("ID du client", fontsize=12)
        axes[0].set_ylabel("Chiffre d'affaires (€)", fontsize=12)

        # Produits achetés par client
        produits_par_client_categ_top10.plot(kind="bar", stacked=True, ax=axes[1], color=palette, edgecolor="black")
        axes[1].set_title("Top 10 des clients par nombre de produits achetés", fontsize=16, fontweight="bold")
        axes[1].set_xlabel("ID du client", fontsize=12)
        axes[1].set_ylabel("Nombre de produits achetés", fontsize=12)

        # Sessions par client
        sessions_par_client_categ_top10.plot(kind="bar", stacked=True, ax=axes[2], color=palette, linewidth=0.8, edgecolor="black")
        axes[2].set_title("Top 10 des clients par nombre de sessions", fontsize=16, fontweight="bold")
        axes[2].set_xlabel("ID du client", fontsize=12)
        axes[2].set_ylabel("Nombre de sessions", fontsize=12)

        plt.tight_layout()  # Ajuste les espacements pour éviter les chevauchements
        plt.show()

        # Retourner les DataFrames
        return {
            "top_10_clients_ca": ca_par_client_categ_top10,
            "top_10_clients_produits": produits_par_client_categ_top10,
            "top_10_clients_sessions": sessions_par_client_categ_top10
        }

    elif type_resultat == "produits":
        # ---- Top 10 Produits ----

        # CA par produit
        ca_par_produit = donnees.groupby(["id_prod", "categ"])["price"].sum().unstack(fill_value=0)
        top_10_produits_ca = ca_par_produit.sum(axis=1).sort_values(ascending=False).head(10)
        ca_par_produit_top10 = ca_par_produit.loc[top_10_produits_ca.index]

        # Ventes par produit
        ventes_par_produit = donnees.groupby(["id_prod", "categ"])["id_prod"].count().unstack(fill_value=0)
        top_10_produits_ventes = ventes_par_produit.sum(axis=1).sort_values(ascending=False).head(10)
        ventes_par_produit_top10 = ventes_par_produit.loc[top_10_produits_ventes.index]

        # Produits les plus chers
        prix_moyen_par_produit = donnees.groupby(["id_prod", "categ"])["price"].mean().unstack(fill_value=0)
        top_10_produits_chers = prix_moyen_par_produit.mean(axis=1).sort_values(ascending=False).head(10)
        prix_moyen_par_produit_top10 = prix_moyen_par_produit.loc[top_10_produits_chers.index]

        # Visualisation pour les Produits (disposition verticale)
        fig, axes = plt.subplots(3, 1, figsize=(15, 15))  # 3 lignes, 1 colonne

        # CA par produit
        ca_par_produit_top10.plot(kind="bar", stacked=True, ax=axes[0], color=palette, edgecolor="black")
        axes[0].set_title("Top 10 des produits par chiffre d'affaires", fontsize=16, fontweight="bold")
        axes[0].set_xlabel("ID du produit", fontsize=12)
        axes[0].set_ylabel("Chiffre d'affaires (€)", fontsize=12)

        # Ventes par produit
        ventes_par_produit_top10.plot(kind="bar", stacked=True, ax=axes[1], color=palette, edgecolor="black")
        axes[1].set_title("Top 10 des produits par nombre de ventes", fontsize=16, fontweight="bold")
        axes[1].set_xlabel("ID du produit", fontsize=12)
        axes[1].set_ylabel("Nombre de ventes", fontsize=12)

        # Produits les plus chers
        prix_moyen_par_produit_top10.plot(kind="bar", stacked=True, ax=axes[2], color=palette, edgecolor="black")
        axes[2].set_title("Top 10 des produits les plus chers", fontsize=16, fontweight="bold")
        axes[2].set_xlabel("ID du produit", fontsize=12)
        axes[2].set_ylabel("Prix moyen (€)", fontsize=12)

        plt.tight_layout()  # Ajuste les espacements pour éviter les chevauchements
        plt.show()

        # Retourner les DataFrames
        return {
            "top_10_produits_ca": ca_par_produit_top10,
            "top_10_produits_ventes": ventes_par_produit_top10,
            "top_10_produits_chers": prix_moyen_par_produit_top10
        }

    else:
        raise ValueError("Le type de résultat doit être 'clients' ou 'produits'")
# ========================================
# Fonction pour afficher le TOP10 produits
# ========================================
def top_produits(donnees):
    # Palette de couleurs
    palette = sns.color_palette("Set2", 3)  # Palette de 3 couleurs

    # Données pour le top 10 des produits par CA
    ca_par_produit = donnees.groupby(["id_prod", "categ"])["price"].sum().unstack(fill_value=0)
    top_10_produits_ca = ca_par_produit.sum(axis=1).sort_values(ascending=False).head(10)
    ca_par_produit_top10 = ca_par_produit.loc[top_10_produits_ca.index]

    # Données pour le top 10 des produits par nombre de ventes
    ventes_par_produit = donnees.groupby(["id_prod", "categ"])["id_prod"].count().unstack(fill_value=0)
    top_10_produits_ventes = ventes_par_produit.sum(axis=1).sort_values(ascending=False).head(10)
    ventes_par_produit_top10 = ventes_par_produit.loc[top_10_produits_ventes.index]

    # Données pour le top 10 des produits les plus chers
    prix_moyen_par_produit = donnees.groupby(["id_prod", "categ"])["price"].mean().unstack(fill_value=0)
    top_10_produits_chers = prix_moyen_par_produit.mean(axis=1).sort_values(ascending=False).head(10)
    prix_moyen_par_produit_top10 = prix_moyen_par_produit.loc[top_10_produits_chers.index]

    # Visualisation

    # 1. Chiffre d'affaires par produit
    ax = ca_par_produit_top10.plot(kind="bar", stacked=True, figsize=(15, 5), color=palette, edgecolor="black")
    plt.title("Top 10 des produits par chiffre d'affaires (avec nombre de ventes)", fontsize=16, fontweight="bold")
    plt.xlabel("ID du produit", fontsize=12)
    plt.ylabel("Chiffre d'affaires (€)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Catégorie", fontsize=10)
    for container in ax.containers:
        ax.bar_label(container, label_type="center", fmt="%.0f", color='white', fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 2. Nombre de ventes par produit (avec prix moyen affiché)
    ax = ventes_par_produit_top10.plot(kind="bar", stacked=True, figsize=(15, 5), color=palette, edgecolor="black")
    plt.title("Top 10 des produits par nombre de ventes (avec prix moyen)", fontsize=16, fontweight="bold")
    plt.xlabel("ID du produit", fontsize=12)
    plt.ylabel("Nombre de ventes", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Catégorie", fontsize=10)

    # Ajouter le prix moyen au milieu des barres
    prix_moyens = prix_moyen_par_produit.loc[ventes_par_produit_top10.index].mean(axis=1)
    for bar, prix in zip(ax.patches, np.tile(prix_moyens, len(palette))):  # Répéter les prix pour chaque catégorie
        if bar.get_height() > 0:  # Éviter les divisions par zéro
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f"{prix:.2f} €",
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    plt.tight_layout()
    plt.show()

    # 3. Produits les plus chers (avec nombre de ventes affiché)
    ax = prix_moyen_par_produit_top10.plot(kind="bar", stacked=True, figsize=(15, 5), color=palette, edgecolor="black")
    plt.title("Top 10 des produits les plus chers (avec nombre de ventes)", fontsize=16, fontweight="bold")
    plt.xlabel("ID du produit", fontsize=12)
    plt.ylabel("Prix moyen (€)", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    # Ajouter le nombre de ventes au centre des barres
    nombre_ventes = ventes_par_produit.sum(axis=1).loc[prix_moyen_par_produit_top10.index]

    # Calculer la position des labels dans les barres empilées
    bar_index = 0
    for i, bar in enumerate(ax.patches):
        if bar.get_height() > 0:
            height = bar.get_height()
            width = bar.get_width()
            x_pos = bar.get_x() + width / 2
            y_pos = bar.get_y() + height / 2  # Calculer la position du centre de la barre empilée
            
            # Afficher les valeurs dans les barres (en noir et en gras)
            ax.text(x_pos, y_pos, f"{nombre_ventes.iloc[bar_index]:.0f}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
            
            bar_index += 1

    plt.tight_layout()
    plt.show()
# ========================================
# Fonction pour afficher le FLOP 10 produits
# ========================================
def flop_produits(donnees):
    # Palette de couleurs
    palette = sns.color_palette("Set2", 3)  # Palette de 3 couleurs

    # Données pour le flop 10 des produits par CA
    ca_par_produit = donnees.groupby(["id_prod", "categ"])["price"].sum().unstack(fill_value=0)
    flop_10_produits_ca = ca_par_produit.sum(axis=1).sort_values(ascending=True).head(10)
    ca_par_produit_flop10 = ca_par_produit.loc[flop_10_produits_ca.index]

    # Données pour le flop 10 des produits par nombre de ventes
    ventes_par_produit = donnees.groupby(["id_prod", "categ"])["id_prod"].count().unstack(fill_value=0)
    flop_10_produits_ventes = ventes_par_produit.sum(axis=1).sort_values(ascending=True).head(10)
    ventes_par_produit_flop10 = ventes_par_produit.loc[flop_10_produits_ventes.index]

    # Données pour le flop 10 des produits les moins chers
    prix_moyen_par_produit = donnees.groupby(["id_prod", "categ"])["price"].mean().unstack(fill_value=0)
    flop_10_produits_chers = prix_moyen_par_produit.mean(axis=1).sort_values(ascending=True).head(10)
    prix_moyen_par_produit_flop10 = prix_moyen_par_produit.loc[flop_10_produits_chers.index]

    # Visualisation

    # 1. Chiffre d'affaires par produit
    ax = ca_par_produit_flop10.plot(kind="bar", stacked=True, figsize=(15, 5), color=palette, edgecolor="black")
    plt.title("Flop 10 des produits par chiffre d'affaires (avec nombre de ventes)", fontsize=16, fontweight="bold")
    plt.xlabel("ID du produit", fontsize=12)
    plt.ylabel("Chiffre d'affaires (€)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Catégorie", fontsize=10)
    for container in ax.containers:
        ax.bar_label(container, label_type="center", fmt="%.0f", color='white', fontweight="bold")  # Affiche les valeurs en blanc et gras au centre des barres
    plt.tight_layout()
    plt.show()

    # 2. Nombre de ventes par produit (avec prix moyen affiché)
    ax = ventes_par_produit_flop10.plot(kind="bar", stacked=True, figsize=(15, 5), color=palette, edgecolor="black")
    plt.title("Flop 10 des produits par nombre de ventes (avec prix moyen)", fontsize=16, fontweight="bold")
    plt.xlabel("ID du produit", fontsize=12)
    plt.ylabel("Nombre de ventes", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Catégorie", fontsize=10)

    # Ajouter le prix moyen au milieu des barres
    prix_moyens = prix_moyen_par_produit.loc[ventes_par_produit_flop10.index].mean(axis=1)
    for bar, prix in zip(ax.patches, np.tile(prix_moyens, len(palette))):  # Répéter les prix pour chaque catégorie
        if bar.get_height() > 0:  # Éviter les divisions par zéro
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f"{prix:.2f} €",
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold")  # Texte en blanc et gras
    plt.tight_layout()
    plt.show()

    # 3. Flop 10 des produits les moins chers (avec nombre de ventes affiché)
    produits_flop10 = donnees.groupby("id_prod")["price"].sum().sort_values(ascending=True).head(10)

    # Tracer le graphique des prix par produit
    prix_par_produit_flop10 = donnees.groupby("id_prod")["price"].sum().loc[produits_flop10.index]

    # Tracer le graphique du Flop 10 des produits les moins chers
    ax = prix_par_produit_flop10.plot(kind="bar", stacked=True, figsize=(15, 5), color=palette, edgecolor="black")
    plt.title("Flop 10 des produits les moins chers (avec nombre de ventes)", fontsize=16, fontweight="bold")
    plt.xlabel("ID du produit", fontsize=12)
    plt.ylabel("Prix total (€)", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    # Ajouter le nombre de ventes dans les barres
    nombre_ventes = donnees.groupby("id_prod")["session_id"].count().loc[produits_flop10.index]
    for bar, ventes in zip(ax.patches, nombre_ventes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f"{ventes:.0f}",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")  # Texte en blanc et gras

    plt.tight_layout()
    plt.show()
# ========================================
# Fonction pour afficher le FLOP 10 clients
# ========================================
def flop_clients(donnees):
    # Palette de couleurs
    palette = sns.color_palette("Set2", 3)  # Palette de 3 couleurs

    # Flop 10 clients par CA (avec détail par catégorie)
    ca_par_client_categ = donnees.groupby(["client_id", "categ"])["price"].sum().unstack(fill_value=0)
    flop_10_clients_ca = ca_par_client_categ.sum(axis=1).sort_values(ascending=True).head(10)
    ca_par_client_categ_flop10 = ca_par_client_categ.loc[flop_10_clients_ca.index]

    # Flop 10 clients par produits achetés (avec détail par catégorie)
    produits_par_client_categ = donnees.groupby(["client_id", "categ"])["id_prod"].count().unstack(fill_value=0)
    flop_10_clients_produits = produits_par_client_categ.sum(axis=1).sort_values(ascending=True).head(10)
    produits_par_client_categ_flop10 = produits_par_client_categ.loc[flop_10_clients_produits.index]

    # Flop 10 clients par sessions (avec détail par catégorie)
    sessions_par_client_categ = donnees.groupby(["client_id", "categ"])["session_id"].nunique().unstack(fill_value=0)
    flop_10_clients_sessions = sessions_par_client_categ.sum(axis=1).sort_values(ascending=True).head(10)
    sessions_par_client_categ_flop10 = sessions_par_client_categ.loc[flop_10_clients_sessions.index]

    # Visualisation

    # CA par client
    ax = ca_par_client_categ_flop10.plot(kind="bar", stacked=True, figsize=(15, 5), color=palette, edgecolor="black")
    plt.title("Flop 10 des clients par chiffre d'affaires (détail par catégorie)", fontsize=16, fontweight="bold")
    plt.xlabel("ID du client", fontsize=12)
    plt.ylabel("Chiffre d'affaires (€)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Catégorie", fontsize=10)
    plt.tight_layout()
    plt.show()

    # Produits achetés par client
    ax = produits_par_client_categ_flop10.plot(kind="bar", stacked=True, figsize=(15, 5), color=palette, edgecolor="black")
    plt.title("Flop 10 des clients par nombre de produits achetés (détail par catégorie)", fontsize=16, fontweight="bold")
    plt.xlabel("ID du client", fontsize=12)
    plt.ylabel("Nombre de produits achetés", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Catégorie", fontsize=10)
    plt.tight_layout()
    plt.show()

    # Sessions par client
    ax = sessions_par_client_categ_flop10.plot(kind="bar", stacked=True, figsize=(15, 5), color=palette, edgecolor="black")
    plt.title("Flop 10 des clients par nombre de sessions (détail par catégorie)", fontsize=16, fontweight="bold")
    plt.xlabel("ID du client", fontsize=12)
    plt.ylabel("Nombre de sessions", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Catégorie", fontsize=10)
    plt.tight_layout()
    plt.show()
# ========================================
# Fonction pour afficher les graphiques des clients BtoB 
# ========================================
def evolution_clients_b2b(donnees, clients_b2b):
    # Filtrer les données pour les clients BtoB
    donnees_b2b = donnees[donnees['client_id'].isin(clients_b2b)].copy()  # Ajouter .copy() pour éviter le problème

    # Créer la colonne 'année_mois' pour regrouper par mois et année en utilisant .loc
    donnees_b2b.loc[:, 'année_mois'] = pd.to_datetime(donnees_b2b['année'].astype(str) + '-' + donnees_b2b['mois'].astype(str))

    # Extraire les dates uniques pour l'affichage des étiquettes
    dates_b2b = donnees_b2b['année_mois'].dt.strftime('%Y-%m')  # Convertir les dates au format "année-mois"

    # Créer les graphiques pour les clients BtoB
    plt.figure(figsize=(15, 12))

    # Graphique 1 : Évolution du CA mensuel pour les clients BtoB
    plt.subplot(3, 1, 1)  # Premier graphique (3 lignes, 1 colonne, position 1)
    for client in clients_b2b:
        # Filtrer les données pour le client
        ca_client_b2b = donnees_b2b[donnees_b2b['client_id'] == client].groupby(["année_mois"])["price"].sum()
        plt.plot(ca_client_b2b.index, ca_client_b2b.values, marker='o', label=f'CA - Client {client}')

    # Affichage de toutes les dates sur l'axe horizontal
    plt.xticks(donnees_b2b['année_mois'].unique(), rotation=45, ha='right')  # Utiliser toutes les dates uniques
    plt.xlabel('Mois')
    plt.ylabel('CA (€)')
    plt.title('Évolution du CA mensuel (Clients BtoB)', fontweight='bold')
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)

    # Graphique 2 : Évolution du nombre de ventes mensuelles pour les clients BtoB
    plt.subplot(3, 1, 2)  # Deuxième graphique (3 lignes, 1 colonne, position 2)
    for client in clients_b2b:
        # Filtrer les données pour le client
        ventes_client_b2b = donnees_b2b[donnees_b2b['client_id'] == client].groupby(["année_mois"])["price"].count()
        plt.plot(ventes_client_b2b.index, ventes_client_b2b.values, marker='o', label=f'Ventes - Client {client}')

    # Affichage de toutes les dates sur l'axe horizontal
    plt.xticks(donnees_b2b['année_mois'].unique(), rotation=45, ha='right')  # Utiliser toutes les dates uniques
    plt.xlabel('Mois')
    plt.ylabel('Nombre de ventes')
    plt.title('Évolution du nombre de ventes mensuelles (Clients BtoB)', fontweight='bold')
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)

    # Graphique 3 : Évolution du nombre de sessions mensuelles pour les clients BtoB
    plt.subplot(3, 1, 3)  # Troisième graphique (3 lignes, 1 colonne, position 3)
    for client in clients_b2b:
        # Filtrer les données pour le client
        sessions_client_b2b = donnees_b2b[donnees_b2b['client_id'] == client].groupby(["année_mois"])["session_id"].nunique()
        plt.plot(sessions_client_b2b.index, sessions_client_b2b.values, marker='o', label=f'Sessions - Client {client}')

    # Affichage de toutes les dates sur l'axe horizontal
    plt.xticks(donnees_b2b['année_mois'].unique(), rotation=45, ha='right')  # Utiliser toutes les dates uniques
    plt.xlabel('Mois')
    plt.ylabel('Nombre de sessions')
    plt.title('Évolution du nombre de sessions mensuelles (Clients BtoB)', fontweight='bold')
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)

    # Afficher les graphiques
    plt.tight_layout()
    plt.show()
# ========================================
# Fonction pour afficher les graphiques pour les clients BtoB
# ========================================
def repartition_ca_ventes(donnees, donnees_b2b, clients_b2b):
    # Calcul du CA total des clients BtoB
    ca_b2b = donnees_b2b["price"].sum()

    # Calcul du CA total des autres clients (en excluant les clients BtoB)
    donnees_autres_clients = donnees[~donnees['client_id'].isin(clients_b2b)]
    ca_autres_clients = donnees_autres_clients["price"].sum()

    # Calcul du nombre de ventes total pour les clients BtoB
    ventes_b2b = donnees_b2b["price"].count()

    # Calcul du nombre de ventes total pour les autres clients (en excluant les clients BtoB)
    ventes_autres_clients = donnees_autres_clients["price"].count()

    # Créer les deux graphiques en secteur côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Graphique en secteur pour le CA
    axes[0].pie([ca_b2b, ca_autres_clients], labels=['Clients BtoB', 'Autres Clients'], 
                colors=['#ff7f0e', '#1f77b4'], autopct='%1.1f%%', startangle=90, 
                explode=(0.1, 0), wedgeprops={'edgecolor': 'black'}, textprops={'color': 'white', 'fontweight': 'bold'})
    axes[0].set_title("Répartition du Chiffre d'Affaires", fontweight='bold')
    axes[0].axis('equal')  # Assurer que le graphique soit circulaire

    # Graphique en secteur pour le nombre de ventes
    axes[1].pie([ventes_b2b, ventes_autres_clients], labels=['Clients BtoB', 'Autres Clients'], 
                colors=['#ff7f0e', '#1f77b4'], autopct='%1.1f%%', startangle=90, 
                explode=(0.1, 0), wedgeprops={'edgecolor': 'black'}, textprops={'color': 'white', 'fontweight': 'bold'})
    axes[1].set_title("Répartition du Nombre de Ventes", fontweight='bold')
    axes[1].axis('equal')  # Assurer que le graphique soit circulaire

    # Afficher les graphiques
    plt.tight_layout()
    plt.show()
# ========================================
# Fonction pour tracer la courbe de Lorenz pour le chiffre d'affaires
# ========================================
def courbe_lorenz_chiffre_affaires(donnees):
    # Calculer le total du chiffre d'affaires pour chaque client
    ca_par_client = donnees.groupby('client_id')['price'].sum()

    # Trier les clients par chiffre d'affaires
    ca_par_client = ca_par_client.sort_values(ascending=True)

    # Calculer la proportion cumulée du chiffre d'affaires
    ca_cumulatif = ca_par_client.cumsum()
    ca_total_clients = ca_par_client.sum()
    ca_cumulatif_proportion = ca_cumulatif / ca_total_clients

    # Calculer la proportion cumulée des clients
    clients_cumulatif_proportion = np.arange(1, len(ca_par_client) + 1) / len(ca_par_client)

    # Tracer la courbe de Lorenz
    plt.figure(figsize=(8, 8))
    plt.plot(clients_cumulatif_proportion, ca_cumulatif_proportion, label='Courbe de Lorenz', color='#1f77b4', linewidth=2)
    plt.plot([0, 1], [0, 1], label='Égalité parfaite', color='red', linestyle='--', linewidth=2)

    # Ajouter des labels et une légende
    plt.title("Courbe de Lorenz pour le Chiffre d'Affaires", fontweight='bold')
    plt.xlabel("Proportion des Clients")
    plt.ylabel("Proportion Cumulée du Chiffre d'Affaires")
    plt.legend(loc="best")
    plt.grid(True)

    # Afficher le graphique
    plt.show()
# ========================================
# Fonction pour tracer la courbe de Pareto pour le chiffre d'affaires
# ========================================
def principe_pareto_chiffre_affaires(donnees):
    # Étape 1 : Calculer le CA par client
    ca_par_client = donnees.groupby('client_id')['price'].sum()

    # Étape 2 : Trier les données par CA croissant
    ca_par_client = ca_par_client.sort_values(ascending=False)

    # Étape 3 : Calculer le pourcentage cumulé des revenus
    ca_total_clients2 = ca_par_client.sum()
    ca_cumule = ca_par_client.cumsum()
    pourcentage_cumule = ca_cumule / ca_total_clients2 * 100

    # Étape 4 : Calculer la contribution des clients (en pourcentage)
    clients_cumule = np.arange(1, len(ca_par_client) + 1) / len(ca_par_client) * 100

    # Étape 5 : Créer le graphique
    plt.figure(figsize=(12, 8))

    # Courbe de Pareto (CA cumulé)
    plt.plot(clients_cumule, pourcentage_cumule, label='CA cumulé', color='b', linewidth=2)

    # Ligne des 80/20
    plt.axvline(x=20, color='r', linestyle='--', label='Ligne des 20% (80/20)')
    plt.axhline(y=80, color='g', linestyle='--', label='Ligne des 80% (80/20)')

    # Ajouter une échelle et des annotations
    plt.fill_between(clients_cumule, pourcentage_cumule, color='blue', alpha=0.1, label='Zone cumulée')
    plt.xlabel('Pourcentage des clients (%)', fontsize=12)
    plt.ylabel('Pourcentage cumulé du CA (%)', fontsize=12)
    plt.title('Principe de Pareto pour le CA des clients', fontsize=14, fontweight='bold')

    # Ajouter des ticks et une grille
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(alpha=0.4)
    plt.legend(fontsize=10)

    # Afficher le graphique
    plt.tight_layout()
    plt.show()
# ========================================
# Fonction pour afficher les analyses chiffrées 
# ========================================
# Fonction pour calculer les indicateurs (avec un paramètre pour la catégorie de regroupement)
def calculer_indicateurs(donnees_filtrees, groupe_colonne):
    ca_total_groupes = donnees_filtrees["price"].sum()
    nombre_commandes = donnees_filtrees["session_id"].nunique()
    nombre_produits_vendus = donnees_filtrees["session_id"].count()
    produits_par_commande = nombre_produits_vendus / nombre_commandes if nombre_commandes > 0 else 0
    panier_moyen = ca_total_groupes / nombre_commandes if nombre_commandes > 0 else 0
    return {
        "CA total (€)": ca_total_groupes,
        "Nombre de commandes": nombre_commandes,
        "Nombre de produits vendus": nombre_produits_vendus,
        "Produits par commande": produits_par_commande,
        "Panier moyen (€)": panier_moyen,
    }

# Fonction pour afficher les résultats pour un regroupement (par "categ" ou "sex")
def afficher_indicateurs_par_groupe(donnees_premiere_annee, donnees_seconde_annee, groupe_colonne):
    # Définir les groupes uniques (par exemple, "categ" ou "sex")
    groupes = donnees_premiere_annee[groupe_colonne].unique()

    # Initialisation des dictionnaires pour stocker les résultats
    resultats_premiere_annee = {}
    resultats_seconde_annee = {}

    # Calculer les indicateurs pour chaque groupe
    for groupe in groupes:
        # Filtrer les données pour chaque groupe et chaque année
        donnees_groupe_premiere_annee = donnees_premiere_annee[donnees_premiere_annee[groupe_colonne] == groupe]
        donnees_groupe_seconde_annee = donnees_seconde_annee[donnees_seconde_annee[groupe_colonne] == groupe]
        
        # Stocker les résultats par groupe
        resultats_premiere_annee[groupe] = calculer_indicateurs(donnees_groupe_premiere_annee, groupe_colonne)
        resultats_seconde_annee[groupe] = calculer_indicateurs(donnees_groupe_seconde_annee, groupe_colonne)

    # Afficher les résultats pour chaque groupe et chaque année commerciale
    for groupe in groupes:
        print(f"\n--- {groupe_colonne.capitalize()} {groupe} ---")
        print("Première année commerciale (mars 2021 - février 2022):")
        for indicateur, valeur in resultats_premiere_annee[groupe].items():
            print(f"- {indicateur} : {valeur:.2f}" if isinstance(valeur, float) else f"- {indicateur} : {valeur}")
        
        print("\nSeconde année commerciale (mars 2022 - février 2023):")
        for indicateur, valeur in resultats_seconde_annee[groupe].items():
            print(f"- {indicateur} : {valeur:.2f}" if isinstance(valeur, float) else f"- {indicateur} : {valeur}")

        # Calculer les évolutions
        evolutions = {
            indicateur: (
                (resultats_seconde_annee[groupe][indicateur] - resultats_premiere_annee[groupe][indicateur]) / 
                resultats_premiere_annee[groupe][indicateur] * 100
            ) if resultats_premiere_annee[groupe][indicateur] > 0 else 0
            for indicateur in resultats_premiere_annee[groupe]
        }
        
        print("\nÉvolutions entre les deux années:")
        for indicateur, valeur in evolutions.items():
            print(f"- Évolution {indicateur} : {valeur:.2f}%")
# ========================================
# Fonction pour afficher les répartitions
# ========================================
# Fonction pour analyser la répartition par groupe (categ ou sex)
def analyser_repartition_par_groupe(donnees, group_col):
    # Calculer les métriques par groupe (categ ou sex)
    ca_par_groupe = donnees.groupby(group_col)["price"].sum()
    ventes_par_groupe = donnees.groupby(group_col)["id_prod"].count()
    sessions_par_groupe = donnees.groupby(group_col)["session_id"].nunique()
    clients_par_groupe = donnees.groupby(group_col)["client_id"].nunique()

    # Palette vive et harmonieuse
    palette_vive = sns.color_palette("Set2", n_colors=len(ca_par_groupe.unique()))

    # Fonction pour créer un camembert avec des couleurs harmonieuses
    def creer_camembert(data, title, colors, ax):
        labels = [f"{group}" for group in data.index]
        sizes = data.values
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('equal')  # Assure un affichage circulaire

    # Créer la figure et les sous-graphiques sur une seule ligne
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 1 ligne, 4 colonnes

    # Affichage des graphiques
    creer_camembert(ca_par_groupe, "CA", palette_vive, axes[0])
    creer_camembert(ventes_par_groupe, "Ventes", palette_vive, axes[1])
    creer_camembert(sessions_par_groupe, "Sessions", palette_vive, axes[2])
    creer_camembert(clients_par_groupe, "Clients", palette_vive, axes[3])

    # Ajuster l'espace entre les graphiques
    plt.tight_layout()

    # Afficher les graphiques
    plt.show()
# ========================================
# Fonction pour afficher l'évolution des répartitions
# ========================================
def evolution_repartition_par_groupe(donnees, group_col):
    """
    Cette fonction génère des graphiques d'évolution du chiffre d'affaires (CA)
    et du nombre de ventes mensuelles, pour chaque groupe spécifié par la colonne `group_col`.
    
    Paramètres :
    - donnees : DataFrame contenant les données de vente
    - group_col : str, la colonne à utiliser pour grouper les données (par exemple, 'categ' ou 'sex')
    """
    
    # Initialisation de la figure
    plt.figure(figsize=(15, 12))

    # Graphique 1 : Évolution du chiffre d'affaires mensuel
    plt.subplot(2, 1, 1)
    for group in donnees[group_col].unique():
        # Filtrer les données pour chaque groupe
        data_group = donnees[donnees[group_col] == group]
        
        # Assurer que 'année_mois' est en format datetime
        data_group.loc[:, 'année_mois'] = pd.to_datetime(data_group['année_mois'], format='%Y-%m')
        
        # Grouper par mois et calculer le CA mensuel
        ca_mensuel = data_group.groupby("année_mois")["price"].sum()
        
        # Tracer la courbe pour le groupe
        plt.plot(
            ca_mensuel.index, 
            ca_mensuel.values, 
            marker='o', 
            linewidth=2.5, 
            label=f'{group_col} {group}'
        )

    plt.title(f'Évolution du CA mensuel par {group_col}', fontweight='bold', fontsize=14)
    plt.xlabel('Mois', fontsize=12)
    plt.ylabel('Chiffre d\'Affaires (€)', fontsize=12)

    # Afficher toutes les étiquettes sur l'axe X
    plt.xticks(ca_mensuel.index, rotation=45, ha='right')

    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)
    plt.legend(title=f'{group_col}s', fontsize=10)

    # Graphique 2 : Évolution du nombre de ventes mensuelles
    plt.subplot(2, 1, 2)
    for group in donnees[group_col].unique():
        # Filtrer les données pour chaque groupe
        data_group = donnees[donnees[group_col] == group]
        
        # Assurer que 'année_mois' est en format datetime
        data_group.loc[:, 'année_mois'] = pd.to_datetime(data_group['année_mois'], format='%Y-%m')
        
        # Grouper par mois et calculer le nombre de ventes mensuelles
        ventes_mensuelles = data_group.groupby("année_mois")["price"].count()
        
        # Tracer la courbe pour le groupe
        plt.plot(
            ventes_mensuelles.index, 
            ventes_mensuelles.values, 
            marker='o', 
            linewidth=2.5, 
            label=f'{group_col} {group}'
        )

    plt.title(f'Évolution du nombre de ventes mensuelles par {group_col}', fontweight='bold', fontsize=14)
    plt.xlabel('Mois', fontsize=12)
    plt.ylabel('Nombre de ventes', fontsize=12)

    # Afficher toutes les étiquettes sur l'axe X
    plt.xticks(ventes_mensuelles.index, rotation=45, ha='right')

    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)
    plt.legend(title=f'{group_col}s', fontsize=10)

    # Ajustement et affichage
    plt.tight_layout()
    plt.show()
# ========================================
# Fonction pour afficher l'évolution par sexe
# ========================================
def analyse_evolution_par_sexe(donnees):
    """
    Analyse de l'évolution du chiffre d'affaires (CA) et du nombre de ventes mensuelles
    par sexe avec moyennes mobiles, et génère les graphiques associés.

    Paramètres :
    - donnees : DataFrame contenant les données de vente (doit inclure les colonnes 'sex', 'année_mois', 'price').
    """

    # Générer la palette "Set2" pour deux catégories
    palette_vive = sns.color_palette("Set2", 2)
    couleur_femme = palette_vive[0]  # Première couleur (femmes)
    couleur_homme = palette_vive[1]  # Deuxième couleur (hommes)

    # Fonction pour éclaircir une couleur
    def lighten_color(color, factor=0.7):
        rgb = np.array(mcolors.to_rgb(color))
        return tuple(np.clip(rgb + (1 - rgb) * (1 - factor), 0, 1))

    # Initialisation de la figure
    plt.figure(figsize=(15, 12))

    # Graphique 1 : Évolution du chiffre d'affaires mensuel par sexe avec moyennes mobiles
    plt.subplot(2, 1, 1)
    for sexe in ["f", "m"]:
        # Filtrer les données pour chaque sexe
        data_sexe = donnees[donnees['sex'] == sexe]
        
        # Assurer que 'année_mois' est en format datetime
        data_sexe.loc[:, 'année_mois'] = pd.to_datetime(data_sexe['année_mois'], format='%Y-%m')
        
        # Grouper par mois et calculer le CA mensuel
        ca_mensuel = data_sexe.groupby("année_mois")["price"].sum()
        
        # Calculer la moyenne mobile sur 3 mois
        ca_mensuel_mm = ca_mensuel.rolling(window=3).mean()
        
        # Définir les couleurs
        couleur = couleur_femme if sexe == "f" else couleur_homme
        
        # Tracer la courbe pour le sexe
        plt.plot(
            ca_mensuel.index, 
            ca_mensuel.values, 
            marker='o', 
            linewidth=2.5, 
            label=f'Sexe {sexe.upper()} - CA',
            color=couleur
        )
        
        # Tracer la moyenne mobile avec la couleur plus claire
        plt.plot(
            ca_mensuel_mm.index, 
            ca_mensuel_mm.values, 
            marker='', 
            linestyle='--', 
            linewidth=2.5, 
            label=f'Sexe {sexe.upper()} - Moyenne mobile (3 mois)',
            color=lighten_color(couleur)
        )

    plt.title('Évolution du CA mensuel par sexe avec moyennes mobiles', fontweight='bold', fontsize=14)
    plt.xlabel('Mois', fontsize=12)
    plt.ylabel('Chiffre d\'Affaires (€)', fontsize=12)

    # Afficher toutes les étiquettes sur l'axe X
    plt.xticks(ca_mensuel.index, rotation=45, ha='right')

    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)
    plt.legend(title='Sexes', fontsize=10)

    # Graphique 2 : Évolution du nombre de ventes mensuelles par sexe avec moyennes mobiles
    plt.subplot(2, 1, 2)
    for sexe in ["f", "m"]:
        # Filtrer les données pour chaque sexe
        data_sexe = donnees[donnees['sex'] == sexe]
        
        # Assurer que 'année_mois' est en format datetime
        data_sexe.loc[:, 'année_mois'] = pd.to_datetime(data_sexe['année_mois'], format='%Y-%m')
        
        # Grouper par mois et calculer le nombre de ventes mensuelles
        ventes_mensuelles = data_sexe.groupby("année_mois")["price"].count()
        
        # Calculer la moyenne mobile sur 3 mois
        ventes_mensuelles_mm = ventes_mensuelles.rolling(window=3).mean()
        
        # Définir les couleurs
        couleur = couleur_femme if sexe == "f" else couleur_homme
        
        # Tracer la courbe pour le sexe
        plt.plot(
            ventes_mensuelles.index, 
            ventes_mensuelles.values, 
            marker='o', 
            linewidth=2.5, 
            label=f'Sexe {sexe.upper()} - Ventes',
            color=couleur
        )
        
        # Tracer la moyenne mobile avec la couleur plus claire
        plt.plot(
            ventes_mensuelles_mm.index, 
            ventes_mensuelles_mm.values, 
            marker='', 
            linestyle='--', 
            linewidth=2.5, 
            label=f'Sexe {sexe.upper()} - Moyenne mobile (3 mois)',
            color=lighten_color(couleur)
        )

    plt.title('Évolution du nombre de ventes mensuelles par sexe avec moyennes mobiles', fontweight='bold', fontsize=14)
    plt.xlabel('Mois', fontsize=12)
    plt.ylabel('Nombre de ventes', fontsize=12)

    # Afficher toutes les étiquettes sur l'axe X
    plt.xticks(ventes_mensuelles.index, rotation=45, ha='right')

    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)
    plt.legend(title='Sexes', fontsize=10)

    # Ajustement et affichage
    plt.tight_layout()
    plt.show()
# ========================================
# Fonction pour afficher l'évolution par categ pour octobre 2021
# ========================================
def analyse_evolution_octobre_2021(donnees):
    """
    Analyse de l'évolution du chiffre d'affaires (CA) et du nombre de ventes quotidiennes
    par catégorie pour le mois d'octobre 2021 et génère les graphiques associés.

    Paramètres :
    - donnees : DataFrame contenant les données de vente (doit inclure les colonnes 'année', 'mois', 'categ', 'date', 'price').
    """

    # Filtrage des données pour le mois d'octobre (octobre 2021)
    octobre_2021 = donnees[(donnees['année'] == 2021) & (donnees['mois'] == 10)]

    # Initialisation de la figure
    plt.figure(figsize=(15, 12))

    # Initialisation des catégories
    categories = [0, 1, 2]

    # Graphique 1 : Évolution du CA quotidien par catégorie (octobre 2021)
    plt.subplot(2, 1, 1)

    # Tracer les courbes pour octobre 2021 (toutes catégories)
    for cat in categories:
        # Filtrer les données pour chaque catégorie en octobre 2021
        data_cat_2021 = octobre_2021[octobre_2021['categ'] == cat]
        ca_journalier_2021 = data_cat_2021.groupby("date")["price"].sum()

        # Tracer la courbe pour la catégorie
        plt.plot(
            ca_journalier_2021.index, 
            ca_journalier_2021.values, 
            marker='o', 
            linewidth=2.5, 
            label=f'Catégorie {cat} (2021)'
        )

    # Configurer l'axe x pour afficher toutes les étiquettes (1er au 31 octobre)
    plt.xticks(ticks=ca_journalier_2021.index, labels=[d.strftime('%Y-%m-%d') for d in ca_journalier_2021.index], rotation=45, ha='right')
    plt.title('Évolution du CA quotidien par catégorie (Octobre 2021)', fontweight='bold', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Chiffre d\'Affaires (€)', fontsize=12)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)
    plt.legend(title='Catégories', fontsize=10)

    # Graphique 2 : Évolution du nombre de ventes quotidiennes par catégorie (octobre 2021)
    plt.subplot(2, 1, 2)

    # Tracer les courbes pour octobre 2021 (toutes catégories)
    for cat in categories:
        # Filtrer les données pour chaque catégorie en octobre 2021
        data_cat_2021 = octobre_2021[octobre_2021['categ'] == cat]
        ventes_journalieres_2021 = data_cat_2021.groupby("date")["price"].count()

        # Tracer la courbe pour la catégorie
        plt.plot(
            ventes_journalieres_2021.index, 
            ventes_journalieres_2021.values, 
            marker='o', 
            linewidth=2.5, 
            label=f'Catégorie {cat} (2021)'
        )

    # Configurer l'axe x pour afficher toutes les étiquettes (1er au 31 octobre)
    plt.xticks(ticks=ventes_journalieres_2021.index, labels=[d.strftime('%Y-%m-%d') for d in ventes_journalieres_2021.index], rotation=45, ha='right')

    plt.title('Évolution du nombre de ventes quotidiennes par catégorie (Octobre 2021)', fontweight='bold', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Nombre de ventes', fontsize=12)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)
    plt.legend(title='Catégories', fontsize=10)

    # Ajustement et affichage
    plt.tight_layout()
    plt.show()
# ========================================
# Fonction pour afficher les analyses chiffrées selon l'âge 
# ========================================
def calcul_indicateurs3(donnees):
    """
    Fonction d'exemple pour calculer les indicateurs sur un sous-ensemble de données.
    Vous devrez adapter cette fonction pour inclure vos calculs spécifiques.
    """
    # Exemple d'indicateur, ajustez selon vos besoins
    indicateur = donnees["price"].sum()  # Exemple : somme du chiffre d'affaires
    return {"CA_total": indicateur}

def analyser_comportement_clients(donnees):
    """
    Analyse le comportement des clients en fonction de leur tranche d'âge et calcule les évolutions 
    des indicateurs entre deux années commerciales successives (mars-février).
    
    Paramètres :
    - donnees : DataFrame contenant les données des ventes (doit inclure les colonnes 'année', 'mois', 'birth', 'price', etc.).
    """
    
    # Calculer l'âge des clients
    annee_actuelle = datetime.now().year
    donnees.loc[:, "age"] = annee_actuelle - donnees["birth"]

    # Créer les tranches d'âge
    bins = [0, 17, 29, 39, 49, 59, 120]
    labels = ["<18", "18-29", "30-39", "40-49", "50-59", "60+"]
    donnees.loc[:, "groupe_age"] = pd.cut(donnees["age"], bins=bins, labels=labels, right=True)

    # Séparer les données par années commerciales
    donnees_premiere_annee = donnees[
        (donnees["année"] == 2021) & (donnees["mois"] >= 3) |
        (donnees["année"] == 2022) & (donnees["mois"] <= 2)
    ]

    donnees_seconde_annee = donnees[
        (donnees["année"] == 2022) & (donnees["mois"] >= 3) |
        (donnees["année"] == 2023) & (donnees["mois"] <= 2)
    ]

    # Définir les tranches d'âge
    tranches_age = donnees_premiere_annee['groupe_age'].unique()

    # Initialisation des dictionnaires pour stocker les résultats
    resultats_premiere_annee = {}
    resultats_seconde_annee = {}

    # Calculer les indicateurs pour chaque tranche d'âge
    for tranche in tranches_age:
        # Filtrer les données pour chaque tranche
        donnees_age_premiere_annee = donnees_premiere_annee[donnees_premiere_annee["groupe_age"] == tranche]
        donnees_age_seconde_annee = donnees_seconde_annee[donnees_seconde_annee["groupe_age"] == tranche]
        
        # Stocker les résultats par tranche d'âge
        resultats_premiere_annee[tranche] = calcul_indicateurs3(donnees_age_premiere_annee)
        resultats_seconde_annee[tranche] = calcul_indicateurs3(donnees_age_seconde_annee)

    # Afficher les résultats pour chaque tranche d'âge et chaque année commerciale
    for tranche in tranches_age:
        print(f"\n--- Tranche d'âge {tranche} ---")
        print("Première année commerciale (mars 2021 - février 2022):")
        for indicateur, valeur in resultats_premiere_annee[tranche].items():
            print(f"- {indicateur} : {valeur:.2f}" if isinstance(valeur, float) else f"- {indicateur} : {valeur}")
        
        print("\nSeconde année commerciale (mars 2022 - février 2023):")
        for indicateur, valeur in resultats_seconde_annee[tranche].items():
            print(f"- {indicateur} : {valeur:.2f}" if isinstance(valeur, float) else f"- {indicateur} : {valeur}")

        # Calculer les évolutions
        evolutions = {
            indicateur: (
                (resultats_seconde_annee[tranche][indicateur] - resultats_premiere_annee[tranche][indicateur]) / 
                resultats_premiere_annee[tranche][indicateur] * 100
            ) if resultats_premiere_annee[tranche][indicateur] > 0 else 0
            for indicateur in resultats_premiere_annee[tranche]
        }
        
        print("\nÉvolutions entre les deux années:")
        for indicateur, valeur in evolutions.items():
            print(f"- Évolution {indicateur} : {valeur:.2f}%")
# ========================================
# Fonction pour afficher la répartition par age
# ========================================
def repartition_par_age(donnees):
    # Calculer les indicateurs clés (en dédupliquant les clients)
    ca_par_age = donnees.groupby('groupe_age', observed=True)['price'].sum()
    clients_par_age = donnees.groupby('groupe_age', observed=True)['client_id'].nunique()
    ventes_par_age = donnees.groupby('groupe_age', observed=True)['id_prod'].count()
    ca_moyen_par_client = ca_par_age / clients_par_age

    # Visualisations
    palette_vive = sns.color_palette("Set2", len(ca_par_age))

    # a) Répartition des clients par tranche d'âge (client_id unique)
    # b) CA par tranche d'âge (camembert)
    # c) Panier moyen par tranche d'âge
    # d) CA par tranche d'âge et par sexe (barres empilées)

    # Créer une figure avec 2 lignes et 2 colonnes pour les graphiques côte à côte
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Graphique a) Répartition des clients par tranche d'âge
    clients_par_age.plot(kind='pie', autopct='%1.1f%%', colors=palette_vive, startangle=140, ax=axes[0, 0])
    axes[0, 0].set_title('Répartition des clients par tranche d\'âge', fontsize=16, fontweight='bold')
    axes[0, 0].set_ylabel('')

    # Graphique b) Répartition du CA par tranche d'âge
    ca_par_age.plot(kind='pie', autopct='%1.1f%%', colors=palette_vive, startangle=140, ax=axes[0, 1])
    axes[0, 1].set_title('Répartition du CA par tranche d\'âge', fontsize=16, fontweight='bold')
    axes[0, 1].set_ylabel('')

    # Graphique c) Panier moyen par tranche d'âge
    ca_moyen_par_client.plot(kind='bar', color=palette_vive, edgecolor='black', ax=axes[1, 0])
    axes[1, 0].set_title('Panier moyen par tranche d\'âge', fontsize=16, fontweight='bold')
    axes[1, 0].set_xlabel('Tranches d\'âge', fontsize=12)
    axes[1, 0].set_ylabel('CA moyen (€)', fontsize=12)

    # Graphique d) CA par tranche d'âge et par sexe (barres empilées)
    ca_sexe_age = donnees.groupby(['groupe_age', 'sex'], observed=True)['price'].sum().unstack()
    ca_sexe_age.plot(kind='bar', stacked=True, colormap='Set2', edgecolor='black', ax=axes[1, 1])
    axes[1, 1].set_title('CA par tranche d\'âge et par sexe', fontsize=16, fontweight='bold')
    axes[1, 1].set_xlabel('Tranches d\'âge', fontsize=12)
    axes[1, 1].set_ylabel('Chiffre d\'Affaires (€)', fontsize=12)
    axes[1, 1].legend(title='Sexe', fontsize=10)

    # Ajuster l'espacement
    plt.tight_layout()
    plt.show()

# ========================================
# Fonction pour afficher la répartition par quantile d'âge sans les 21 ans
# ========================================
def repartition_par_quantile_age(donnees):
    # Retirer les clients âgés de 21 ans
    donnees_sans_21_ans = donnees[donnees["age"] != 21].copy()

    # Calculer l'âge unique par client (en supposant que chaque client a un âge constant)
    clients_uniques = donnees_sans_21_ans.drop_duplicates(subset=["client_id"])[["client_id", "age"]]

    # Vérification des statistiques descriptives des âges
    print("Statistiques descriptives des âges (clients uniques, sans 21 ans) :")
    print(clients_uniques["age"].describe())

    # Créer les quantiles d'âge en équilibrant le nombre de clients uniques par quantile
    try:
        clients_uniques["groupe_quantile_age"] = pd.qcut(
            clients_uniques["age"], 
            q=10, 
            labels=["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10"]
        )
    except ValueError as e:
        print("Erreur lors de la création des quantiles :", e)
        print("Il peut y avoir des doublons dans les âges ou une distribution non uniforme.")
        return

    # Ajouter les quantiles à la table principale via un mapping
    donnees_sans_21_ans = donnees_sans_21_ans.merge(
        clients_uniques[["client_id", "groupe_quantile_age"]],
        on="client_id",
        how="left"
    )

    # Vérification de la répartition des quantiles
    quantile_counts = clients_uniques["groupe_quantile_age"].value_counts().sort_index()
    print("\nNombre de clients uniques par quantile d'âge :")
    print(quantile_counts)

    # Calcul des indicateurs clés
    ca_par_quantile = donnees_sans_21_ans.groupby('groupe_quantile_age', observed=True)['price'].sum()
    clients_par_quantile = clients_uniques['groupe_quantile_age'].value_counts().sort_index()
    ca_moyen_par_client = ca_par_quantile / clients_par_quantile

    # Visualisation
    palette_vive = sns.color_palette("Set2", len(ca_par_quantile))

    # Création de la figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Graphique 1 : Répartition des clients uniques par quantile d'âge
    quantile_counts.plot(kind='pie', autopct='%1.1f%%', colors=palette_vive, startangle=140, ax=axes[0])
    axes[0].set_title('Répartition des clients uniques par quantile d\'âge', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('')

    # Graphique 2 : Répartition du CA par quantile d'âge
    ca_par_quantile.plot(kind='pie', autopct='%1.1f%%', colors=palette_vive, startangle=140, ax=axes[1])
    axes[1].set_title('Répartition du CA par quantile d\'âge', fontsize=16, fontweight='bold')
    axes[1].set_ylabel('')

    # Graphique 3 : Panier moyen par quantile d'âge
    ca_moyen_par_client.plot(kind='bar', color=palette_vive, edgecolor='black', ax=axes[2])
    axes[2].set_title('Panier moyen par quantile d\'âge', fontsize=16, fontweight='bold')
    axes[2].set_xlabel('Quantiles d\'âge', fontsize=12)
    axes[2].set_ylabel('CA moyen (€)', fontsize=12)

    # Ajustement de l'espacement
    plt.tight_layout()
    plt.show()

    # Affichage des limites d'âge pour chaque quantile
    print("\nLimites d'âge pour chaque quantile :")
    print(clients_uniques.groupby("groupe_quantile_age")["age"].agg(["min", "max"]))

# ========================================
# Fonction pour l'analyse Lien entre genre et catégorie d'achat
# ========================================
def analyse_table_contingence(donnees):
    """
    Cette fonction effectue une analyse de table de contingence pour les variables 'sex' et 'categ',
    en affichant la table de contingence, la table de contingence en pourcentage et les résultats
    d'un test du Chi-2.

    Arguments:
    donnees : DataFrame contenant les données avec les colonnes 'sex' et 'categ'.
    """
    # Table de contingence
    table_contingence = pd.crosstab(donnees['sex'], donnees['categ'])
    print("Table de contingence (comptages) :\n", table_contingence)
    
    # Table de contingence exprimée en pourcentage
    table_contingence_pourcentage = table_contingence.div(table_contingence.sum(axis=1), axis=0) * 100
    print("\nTable de contingence (en pourcentage) :\n", table_contingence_pourcentage)
    
    # Test du Chi-2
    chi2, p_val, dof, expected = chi2_contingency(table_contingence)
    print("\nRésultats du test du Chi-2 :")
    print(f"Statistique du Chi-2 : {chi2:.2f}")
    print(f"Degrés de liberté (dof) : {dof}")
    print("Matrice des fréquences attendues :\n", pd.DataFrame(expected, 
          index=table_contingence.index, columns=table_contingence.columns))
    print(f"p-value : {p_val:.4f}")
    
    # Interprétation des résultats
    alpha = 0.05
    if p_val < alpha:
        print("\nConclusion :")
        print("Nous rejetons H0. Il existe une association statistiquement significative entre le genre et la catégorie d'achat.")
    else:
        print("\nConclusion :")
        print("Nous ne rejetons pas H0. Aucune association significative entre le genre et la catégorie d'achat.")

# ========================================
# Fonction pour l'analyse tableau Lien entre age et catégorie des achats
# ========================================
def analyse_table_contingence_style(donnees):
    """
    Cette fonction effectue une analyse de table de contingence pour les variables 'sex' et 'categ',
    puis applique un style à la table en pourcentage et ajoute une colonne 'Total' pour la somme des pourcentages.
    
    Arguments:
    donnees : DataFrame contenant les données avec les colonnes 'sex' et 'categ'.
    
    Retourne :
    styled_table : Table de contingence stylisée avec les pourcentages et une colonne 'Total'.
    """
    # Table de contingence
    table_contingence = pd.crosstab(donnees['sex'], donnees['categ'])

    # Table de contingence exprimée en pourcentage
    table_contingence_pourcentage = table_contingence.div(table_contingence.sum(axis=1), axis=0) * 100

    # Ajouter une colonne "Total" (somme des pourcentages pour chaque ligne)
    table_contingence_pourcentage['Total'] = table_contingence_pourcentage.sum(axis=1)

    # Appliquer un style avec coloration, bordures et formatage des nombres
    styled_table = table_contingence_pourcentage.style \
        .background_gradient(cmap='coolwarm', axis=None) \
        .set_properties(**{
            'border': '2px solid black',         # Bordures plus épaisses
            'padding': '10px',                   # Espacement dans les cellules
            'text-align': 'center',              # Alignement centré
            'font-weight': 'bold',               # Texte en gras
            'font-size': '14px'                  # Augmenter la taille de la police
        }) \
        .set_table_styles([
            {'selector': 'thead th', 'props': [('background-color', '#f0f0f0'), ('font-size', '16px'), ('padding', '10px')]},  # Style des en-têtes
            {'selector': 'tbody td', 'props': [('font-size', '14px'), ('padding', '10px')]},  # Style des cellules
            {'selector': 'thead', 'props': [('background-color', '#d9d9d9')]},  # Couleur de fond de l'en-tête
        ]) \
        .format("{:.2f}%")  # Formatage global en pourcentage avec 2 décimales

    return styled_table

# ========================================
# Fonction pour l'analyse Lien entre age et montant des chats
# ========================================
def analyse_lien_age_montant(donnees_sans_21_ans, montant_total_par_age):
    """
    Cette fonction effectue un test de normalité (Kolmogorov-Smirnov) sur l'âge et le montant total des achats,
    puis teste la corrélation entre l'âge et le montant total des achats en fonction de la normalité des distributions.
    
    Arguments :
    donnees_sans_21_ans : DataFrame contenant les données des clients sans 21 ans avec une colonne 'age'.
    montant_total_par_age : Série contenant les montants totaux des achats par âge.
    
    Retourne :
    Aucune valeur retour. Affiche les résultats des tests de normalité et de corrélation.
    """
    # Test de normalité (Kolmogorov-Smirnov) sur l'âge et le montant total des achats
    stat_age, p_value_age = kstest(donnees_sans_21_ans['age'], 'norm', args=(np.mean(donnees_sans_21_ans['age']), np.std(donnees_sans_21_ans['age'])))
    stat_montant, p_value_montant = kstest(montant_total_par_age, 'norm', args=(np.mean(montant_total_par_age), np.std(montant_total_par_age)))

    # Affichage des résultats des tests de normalité
    print(f"Test de normalité pour l'âge : Statistique de KS = {stat_age}, p-value = {p_value_age}")
    print(f"Test de normalité pour le montant total : Statistique de KS = {stat_montant}, p-value = {p_value_montant}")

    # Interprétation des résultats de normalité
    if p_value_age > 0.05:
        print("La distribution de l'âge est normale (on ne rejette pas l'hypothèse nulle).")
    else:
        print("La distribution de l'âge n'est pas normale (on rejette l'hypothèse nulle).")

    if p_value_montant > 0.05:
        print("La distribution du montant total des achats est normale (on ne rejette pas l'hypothèse nulle).")
    else:
        print("La distribution du montant total des achats n'est pas normale (on rejette l'hypothèse nulle).")

    # Hypothèses :
    # H0 : Il n'y a pas de relation entre l'âge et le montant total des achats.
    # H1 : Il existe une relation entre l'âge et le montant total des achats.

    # Test de corrélation
    # Assurez-vous que les données sont correctement agglomérées
    if p_value_age > 0.05 and p_value_montant > 0.05:  # Si les deux distributions sont normales
        # Test de corrélation de Pearson
        correlation_stat, p_value_corr = pearsonr(montant_total_par_age.index, montant_total_par_age.values)
        print(f"Test de corrélation de Pearson : Statistique = {correlation_stat}, p-value = {p_value_corr}")
        if p_value_corr < 0.05:
            print("Il existe une relation significative entre l'âge et le montant total des achats (H1).")
        else:
            print("Il n'y a pas de relation significative entre l'âge et le montant total des achats (H0).")
    else:
        # Test de corrélation de Spearman (si non normalité)
        correlation_stat, p_value_corr = spearmanr(montant_total_par_age.index, montant_total_par_age.values)
        print(f"Test de corrélation de Spearman : Statistique = {correlation_stat}, p-value = {p_value_corr}")
        if p_value_corr < 0.05:
            print("Il existe une relation significative entre l'âge et le montant total des achats (H1).")
        else:
            print("Il n'y a pas de relation significative entre l'âge et le montant total des achats (H0).")

# ========================================
# Fonction pour l'analyse Lien entre age et panier moyen
# ========================================
def analyse_lien_age_panier(donnees_sans_21_ans, age_avg_price):
    """
    Cette fonction effectue un test de normalité (Kolmogorov-Smirnov) sur l'âge et le panier moyen,
    puis teste la corrélation entre l'âge et le panier moyen en fonction de la normalité des distributions.
    
    Arguments :
    donnees_sans_21_ans : DataFrame contenant les données des clients sans 21 ans avec une colonne 'age'.
    age_avg_price : DataFrame contenant les âges et les prix moyens d'achat ('age' et 'price').
    
    Retourne :
    Aucune valeur retour. Affiche les résultats des tests de normalité et de corrélation.
    """
    # Test de normalité (Kolmogorov-Smirnov) sur l'âge et le panier moyen
    stat_age, p_value_age = kstest(donnees_sans_21_ans['age'], 'norm')
    stat_panier, p_value_panier = kstest(age_avg_price['price'], 'norm')

    # Affichage des résultats des tests de normalité
    print(f"Test de normalité pour l'âge : Statistique de Kolmogorov-Smirnov = {stat_age}, p-value = {p_value_age}")
    print(f"Test de normalité pour le panier moyen : Statistique de Kolmogorov-Smirnov = {stat_panier}, p-value = {p_value_panier}")

    # Interprétation des résultats de normalité
    if p_value_age > 0.05:
        print("La distribution de l'âge est normale (on ne rejette pas l'hypothèse nulle).")
    else:
        print("La distribution de l'âge n'est pas normale (on rejette l'hypothèse nulle).")

    if p_value_panier > 0.05:
        print("La distribution du panier moyen est normale (on ne rejette pas l'hypothèse nulle).")
    else:
        print("La distribution du panier moyen n'est pas normale (on rejette l'hypothèse nulle).")

    # Hypothèses :
    # H0 : Il n'y a pas de relation entre l'âge et le panier moyen.
    # H1 : Il existe une relation entre l'âge et le panier moyen.

    # Test de corrélation
    # Assurez-vous que les données sont correctement agglomérées
    if p_value_age > 0.05 and p_value_panier > 0.05:  # Si les deux distributions sont normales
        # Test de corrélation de Pearson
        correlation_stat, p_value_corr = pearsonr(age_avg_price['age'], age_avg_price['price'])
        print(f"Test de corrélation de Pearson : Statistique = {correlation_stat}, p-value = {p_value_corr}")
        if p_value_corr < 0.05:
            print("Il existe une relation significative entre l'âge et le panier moyen (H1).")
        else:
            print("Il n'y a pas de relation significative entre l'âge et le panier moyen (H0).")
    else:
        # Test de corrélation de Spearman (si non normalité)
        correlation_stat, p_value_corr = spearmanr(age_avg_price['age'], age_avg_price['price'])
        print(f"Test de corrélation de Spearman : Statistique = {correlation_stat}, p-value = {p_value_corr}")
        if p_value_corr < 0.05:
            print("Il existe une relation significative entre l'âge et le panier moyen (H1).")
        else:
            print("Il n'y a pas de relation significative entre l'âge et le panier moyen (H0).")

# ========================================
# Fonction pour les boites a moustache panier moyen
# ========================================
def boxplot_panier_moyen(donnees_sans_21_ans):
    """
    Cette fonction génère des boîtes à moustaches (boxplots) pour visualiser la distribution du panier moyen 
    en fonction des tranches d'âge et des groupes d'âge (moins de 33 ans / plus de 33 ans).
    
    Arguments :
    donnees_sans_21_ans : DataFrame contenant les données des clients sans 21 ans, avec les colonnes 'client_id', 'price' et 'age'.
    
    Retourne :
    Aucun retour, mais affiche deux graphiques : un boxplot par tranche d'âge et un autre par groupe d'âge.
    """
    # Calculer le panier moyen par client
    panier_moyen_par_client = donnees_sans_21_ans.groupby('client_id')['price'].mean().reset_index()
    panier_moyen_par_client.rename(columns={'price': 'panier_moyen_par_client'}, inplace=True)

    # Fusionner avec le DataFrame d'origine pour inclure les autres colonnes
    donnees_sans_21_ans = donnees_sans_21_ans.merge(panier_moyen_par_client, on='client_id', how='left')

    # Créer les tranches d'âge
    bins = [18, 29, 39, 49, 59, 100]  # Définir les bornes des tranches d'âge
    labels = ['18-29', '30-39', '40-49', '50-59', '60+']  # Étiquettes des tranches d'âge

    # Ajouter une colonne "Tranche d'âge"
    donnees_sans_21_ans['Tranche d\'âge'] = pd.cut(donnees_sans_21_ans['age'], bins=bins, labels=labels, right=False)

    # Créer le boxplot pour les tranches d'âge
    fig1 = px.box(
        donnees_sans_21_ans,
        x="Tranche d'âge",
        y="panier_moyen_par_client",
        color="Tranche d'âge",
        title="Panier moyen par tranches d'âge",
        labels={"Tranche d'âge": "Tranche d'âge", "panier_moyen_par_client": "Panier moyen"},
        category_orders={"Tranche d'âge": labels},  # Ordonner les catégories sur l'axe X
        width=800,
        height=600
    )

    # Mettre à jour la mise en page pour centrer le titre
    fig1.update_layout(
        title={
            'text': "Panier moyen par tranches d'âge",
            'x': 0.5,  # Position horizontale (0 = gauche, 1 = droite, 0.5 = centre)
            'xanchor': 'center'  # Ancre du titre
        }
    )

    # Afficher le graphique
    fig1.show()

    # Créer les groupes "plus de 33 ans" et "moins de 33 ans"
    donnees_sans_21_ans['Groupe d\'âge'] = donnees_sans_21_ans['age'].apply(lambda x: 'Moins de 33 ans' if x < 33 else 'Plus de 33 ans')

    # Créer le boxplot pour les groupes d'âge
    fig2 = px.box(
        donnees_sans_21_ans,
        x="Groupe d'âge",
        y="panier_moyen_par_client",
        color="Groupe d'âge",
        title="Panier moyen par groupe d'âge (moins de 33 ans / plus de 33 ans)",
        labels={"Groupe d'âge": "Groupe d'âge", "panier_moyen_par_client": "Panier moyen"},
        category_orders={"Groupe d'âge": ['Moins de 33 ans', 'Plus de 33 ans']},  # Ordonner les catégories sur l'axe X
        width=800,
        height=600
    )

    # Mettre à jour la mise en page pour centrer le titre
    fig2.update_layout(
        title={
            'text': "Panier moyen par groupe d'âge (moins de 33 ans / plus de 33 ans)",
            'x': 0.5,  # Position horizontale (0 = gauche, 1 = droite, 0.5 = centre)
            'xanchor': 'center'  # Ancre du titre
        }
    )

    # Afficher le graphique
    fig2.show()

# ========================================
# Fonction pour l'analyse Lien entre age et fréquence d'achat
# ========================================
def test_normalite_corr_age_frequence_achats(donnees_sans_21_ans, nb_achats):
    """
    Cette fonction effectue un test de normalité (Kolmogorov-Smirnov) sur l'âge et la fréquence des achats,
    puis réalise un test de corrélation (Pearson ou Spearman selon la normalité) entre l'âge et la fréquence des achats.
    
    Arguments :
    donnees_sans_21_ans : DataFrame contenant les données des clients sans 21 ans, avec les colonnes 'age'.
    nb_achats : Série ou DataFrame contenant la fréquence des achats (par exemple, nombre d'achats par client).
    
    Retourne :
    Aucun retour, mais affiche les résultats des tests de normalité et de corrélation.
    """
    # Test de normalité (Kolmogorov-Smirnov) sur l'âge et la fréquence des achats
    stat_age, p_value_age = kstest(donnees_sans_21_ans['age'], 'norm')
    stat_nb_achats, p_value_nb_achats = kstest(nb_achats, 'norm')

    # Affichage des résultats des tests de normalité
    print(f"Test de normalité pour l'âge : Statistique de Kolmogorov-Smirnov = {stat_age}, p-value = {p_value_age}")
    print(f"Test de normalité pour la fréquence des achats : Statistique de Kolmogorov-Smirnov = {stat_nb_achats}, p-value = {p_value_nb_achats}")

    # Interprétation des résultats de normalité
    if p_value_age > 0.05:
        print("La distribution de l'âge est normale (on ne rejette pas l'hypothèse nulle).")
    else:
        print("La distribution de l'âge n'est pas normale (on rejette l'hypothèse nulle).")

    if p_value_nb_achats > 0.05:
        print("La distribution de la fréquence des achats est normale (on ne rejette pas l'hypothèse nulle).")
    else:
        print("La distribution de la fréquence des achats n'est pas normale (on rejette l'hypothèse nulle).")

    # Hypothèses :
    # H0 : Il n'y a pas de relation entre l'âge et la fréquence des achats.
    # H1 : Il existe une relation entre l'âge et la fréquence des achats.

    # Test de corrélation
    # Assurez-vous que les données sont correctement agglomérées
    if p_value_age > 0.05 and p_value_nb_achats > 0.05:  # Si les deux distributions sont normales
        # Test de corrélation de Pearson
        correlation_stat, p_value_corr = pearsonr(nb_achats.index, nb_achats.values)
        print(f"Test de corrélation de Pearson : Statistique = {correlation_stat}, p-value = {p_value_corr}")
        if p_value_corr < 0.05:
            print("Il existe une relation significative entre l'âge et la fréquence des achats (H1).")
        else:
            print("Il n'y a pas de relation significative entre l'âge et la fréquence des achats (H0).")
    else:
        # Test de corrélation de Spearman (si non normalité)
        correlation_stat, p_value_corr = spearmanr(nb_achats.index, nb_achats.values)
        print(f"Test de corrélation de Spearman : Statistique = {correlation_stat}, p-value = {p_value_corr}")
        if p_value_corr < 0.05:
            print("Il existe une relation significative entre l'âge et la fréquence des achats (H1).")
        else:
            print("Il n'y a pas de relation significative entre l'âge et la fréquence des achats (H0).")

# ========================================
# Fonction pour graphiques Lien entre age et catégorie d'achat
# ========================================
def graphique_age_categorie(donnees_sans_21_ans, donnees):
    """
    Cette fonction génère deux graphiques en barres empilées montrant la quantité achetée par âge et par catégorie,
    en utilisant les données des clients (excluant les clients de moins de 21 ans) et les données d'âge et de catégorie.

    Arguments :
    donnees_sans_21_ans : DataFrame contenant les données des clients sans 21 ans.
    donnees : DataFrame contenant les données des clients, y compris les âges.

    Affiche deux graphiques en barres empilées.
    """
    # Calculer le nombre d'achats pour chaque combinaison d'âge et de catégorie
    df_age_categorie = donnees_sans_21_ans.groupby(['age', 'categ'])['session_id'].count().reset_index(name='nombre_achat')
    
    # Créer le premier graphique en barres empilées
    fig_age_categorie_barres = px.bar(df_age_categorie, x='age', y='nombre_achat', color='categ',
                                      labels={'categ': 'Catégorie', 'nombre_achat': 'Quantité achetée', 'age': 'Âge'},
                                      title="Quantité achetée par tranches d'âge et par catégorie", 
                                      category_orders={'age': sorted(df_age_categorie['age'].unique())})
    
    # Mise en forme du graphique
    fig_age_categorie_barres.update_layout(
        plot_bgcolor="white",
        xaxis_title="Âge",
        yaxis_title="Quantité achetée",
        barmode='stack'  # Empiler les barres pour chaque tranche d'âge
    )
    # Personnaliser les axes
    fig_age_categorie_barres.update_yaxes(gridcolor='#E0E2E5', showline=True, linewidth=1, linecolor='#E0E2E5')
    fig_age_categorie_barres.update_xaxes(showline=True, linewidth=1, linecolor='#E0E2E5', ticks='outside', nticks=14)
    fig_age_categorie_barres.show()

    # Créer les tranches d'âge
    bins = [0, 17, 29, 39, 49, 59, 120]
    labels = ["<18", "18-29", "30-39", "40-49", "50-59", "60+"]
    donnees.loc[:, "groupe_age"] = pd.cut(donnees["age"], bins=bins, labels=labels, right=True)

    # Calculer le nombre d'achats pour chaque combinaison de groupe d'âge et de catégorie
    df_age_categorie = donnees_sans_21_ans.groupby(['groupe_age', 'categ'], observed=True)['session_id'].count().reset_index(name='nombre_achat')
    
    # Créer le second graphique en barres empilées
    fig_age_categorie_barres = px.bar(df_age_categorie, x='groupe_age', y='nombre_achat', color='categ',
                                      labels={'categ': 'Catégorie', 'nombre_achat': 'Quantité achetée', 'groupe_age': 'Tranche d\'âge'},
                                      title="Quantité achetée par âge et par catégorie", 
                                      category_orders={'groupe_age': labels})
    
    # Mise en forme du graphique
    fig_age_categorie_barres.update_layout(
        plot_bgcolor="white",
        xaxis_title="Tranche d'âge",
        yaxis_title="Quantité achetée",
        barmode='stack'  # Empiler les barres pour chaque tranche d'âge
    )
    # Personnaliser les axes
    fig_age_categorie_barres.update_yaxes(gridcolor='#E0E2E5', showline=True, linewidth=1, linecolor='#E0E2E5')
    fig_age_categorie_barres.update_xaxes(showline=True, linewidth=1, linecolor='#E0E2E5', ticks='outside', nticks=14)
    fig_age_categorie_barres.show()

# ========================================
# Fonction pour analyse lien age - catégorie d'achat
# ========================================
def analyse_age_categorie(donnees):
    """
    Cette fonction effectue l'analyse de la relation entre l'âge et la catégorie d'achat,
    y compris la création des tranches d'âge, le calcul de la table de contingence,
    l'exécution du test du Chi-2 et l'affichage des résultats.

    Arguments :
    donnees : DataFrame contenant les données des clients avec une colonne "age" et "categ".

    Affiche les résultats de l'analyse.
    """
    # Créer les tranches d'âge
    bins = [0, 17, 29, 39, 49, 59, 120]
    labels = ["<18", "18-29", "30-39", "40-49", "50-59", "60+"]
    donnees.loc[:, "groupe_age"] = pd.cut(donnees["age"], bins=bins, labels=labels, right=True)

    # Table de contingence
    table_contingence2 = pd.crosstab(donnees['groupe_age'], donnees['categ'])
    print("Table de contingence (comptages) :\n", table_contingence2)

    # Table de contingence exprimée en pourcentage
    table_contingence_pourcentage2 = table_contingence2.div(table_contingence2.sum(axis=1), axis=0) * 100
    print("\nTable de contingence (en pourcentage) :\n", table_contingence_pourcentage2)

    # Test du Chi-2
    chi2, p_val, dof, expected = chi2_contingency(table_contingence2)
    print("\nRésultats du test du Chi-2 :")
    print(f"Statistique du Chi-2 : {chi2:.2f}")
    print(f"Degrés de liberté (dof) : {dof}")
    print("Matrice des fréquences attendues :\n", pd.DataFrame(expected, 
          index=table_contingence2.index, columns=table_contingence2.columns))
    print(f"p-value : {p_val:.4f}")

    # Interprétation des résultats
    alpha = 0.05
    if p_val < alpha:
        print("\nConclusion :")
        print("Nous rejetons H0. Il existe une association statistiquement significative entre l'âge et la catégorie d'achat.")
    else:
        print("\nConclusion :")
        print("Nous ne rejetons pas H0. Aucune association significative entre l'âge et la catégorie d'achat.")

# ========================================
# Fonction pour tableau contingence lien age - catégorie d'achat
# ========================================
def tableau_contingence_age_categorie():
    """
    Cette fonction crée un tableau de contingence pour les tranches d'âge et les catégories d'achat,
    ajoute une colonne Total représentant la somme des trois catégories, applique un style avec 
    coloration, bordures et formatage des nombres, et affiche le tableau stylisé.

    Retourne le tableau stylisé.
    """
    # Données du tableau
    data = {
        'categ 0': [21.955002, 67.499349, 75.167358, 61.317003, 42.792155],
        'categ 1': [39.012463, 26.197802, 24.476808, 38.083248, 56.308438],
        'categ 2': [39.032535, 6.302849, 0.355834, 0.599749, 0.899407]
    }

    index = ['18-29', '30-39', '40-49', '50-59', '60+']

    # Créer un DataFrame à partir des données
    table_contingence_pourcentage2 = pd.DataFrame(data, index=index)

    # Ajouter une colonne "Total" représentant la somme des trois catégories
    table_contingence_pourcentage2['Total'] = table_contingence_pourcentage2.sum(axis=1)

    # Appliquer un style avec coloration, bordures et formatage des nombres
    styled_table2 = table_contingence_pourcentage2.style \
        .background_gradient(cmap='coolwarm', axis=None) \
        .set_properties(**{
            'border': '2px solid black',         # Bordure plus épaisse
            'padding': '15px',                    # Plus d'espace dans les cellules
            'text-align': 'center',               # Aligner le texte au centre
            'font-weight': 'bold',                # Mettre en gras
            'font-size': '16px'                   # Augmenter la taille de la police
        }) \
        .set_table_styles([
            {'selector': 'thead th', 'props': [('background-color', '#f0f0f0'), ('font-size', '16px'), ('padding', '15px')]},  # En-têtes en gris clair et plus gros
            {'selector': 'tbody td', 'props': [('font-size', '16px'), ('padding', '15px')]},  # Taille de police pour les cellules plus grande
            {'selector': 'thead', 'props': [('background-color', '#d9d9d9')]},  # Couleur de fond de l'en-tête
        ]) \
        .format({
            'categ 0': '{:.2f}%',
            'categ 1': '{:.2f}%',
            'categ 2': '{:.2f}%',
            'Total': '{:.2f}%'  # Formatage pour la colonne Total
        })

    # Retourner le tableau stylisé
    return styled_table2