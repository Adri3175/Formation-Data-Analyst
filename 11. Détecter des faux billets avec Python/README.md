# Projet 11 – Détection de faux billets en euros avec du machine learning

## Contexte

J’ai été missionné par l’**ONCFM (Organisation nationale de lutte contre le faux-monnayage)** pour développer une **application de prédiction automatique** capable d’identifier si un billet en euros est authentique ou contrefait, à partir de caractéristiques physiques scannées.

L’objectif : **accélérer la détection sur le terrain grâce au machine learning**.

---

## Ce que j’ai fait

- Préparation des données : 1500 billets (1000 vrais / 500 faux),
- Traitement et visualisation initiale des caractéristiques (distributions, corrélations, outliers…),
- Séparation jeu d'entraînement / test,
- Entraînement et évaluation de plusieurs algorithmes :
  - **Régression logistique**
  - **KNN (k-plus proches voisins)**
  - **Random Forest**
  - **K-means** (en bonus, pour une vision non supervisée),
- Comparaison des performances : précision, recall, F1-score, matrice de confusion,
- Sélection du **modèle le plus robuste**,
- Développement d’une **mini-application dans un notebook** permettant de prédire en direct la nature d’un billet à partir de ses caractéristiques.

---

## Ce que vous trouverez dans ce dossier

- `Inputs/` : les fichiers bruts fournis
- `Livrables/` :
  - Notebook 1 : préparation, exploration, entraînement & évaluation des modèles
  - Notebook 2 : **application de prédiction utilisable directement**
  - Fichier Python avec les fonctions utilisées dans le notebook
  - Présentation finale
- `Missions.pdf` : l’énoncé complet du projet

---

## Ce que ça montre

Ce projet illustre :
- ma capacité à **gérer un projet ML de bout en bout** (de l’analyse au déploiement),
- la comparaison rigoureuse de plusieurs algorithmes classiques de classification,
- mon aptitude à **documenter les choix techniques pour un public mixte**,
- une approche pratique et concrète, orientée **mise en production légère**.