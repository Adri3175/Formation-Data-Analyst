import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# ========================================
# FONCTION Analyse la normalité de la variable 'Marge_basse'
# ========================================

import numpy as np
import seaborn as sns
import scipy.stats as stats

def analyser_normalite_marge_basse(billets):
    """
    Analyse la normalité de la variable 'Marge_basse' :
    - Supprime les valeurs NaN
    - Effectue le test de Shapiro-Wilk
    - Affiche un histogramme avec KDE et un Q-Q plot
    - Retourne la p-value du test de Shapiro-Wilk
    
    Paramètre :
    - billets (pd.DataFrame) : DataFrame contenant la colonne 'Marge_basse'
    
    Retour :
    - p_value (float) : p-value du test de Shapiro-Wilk
    """
    # Supprimer les NaN
    billets_sans_nan = billets.dropna(subset=['Marge_basse']).reset_index(drop=True)
    
    # Test de Shapiro-Wilk
    stat, p_value = stats.shapiro(billets_sans_nan['Marge_basse'])
    
    # Créer une figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Histogramme avec courbe de densité
    sns.histplot(data=billets_sans_nan, x='Marge_basse', kde=True, ax=ax1)
    ax1.set_title('Distribution de Marge_basse')

    # Diagramme Quantile-Quantil (QQ-plot)
    stats.probplot(billets_sans_nan['Marge_basse'], dist="norm", plot=ax2)
    ax2.set_title('Q-Q plot de Marge_basse')

    plt.tight_layout()
    plt.show()

    # Affichage des résultats du test
    print(f"Test de Shapiro-Wilk : Statistique = {stat:.4f}, p-value = {p_value:.4e}")
    print("\nInterprétation :")
    print("Hypothèse nulle (H₀) : La variable suit une distribution normale.")
    print("Hypothèse alternative (H₁) : La variable ne suit pas une distribution normale.")
    print("Seuil de 5% (α = 0.05) :")
    print("Si p-value > 0.05 → On ne rejette pas H₀ → La distribution peut être considérée comme normale.")
    print("Si p-value ≤ 0.05 → On rejette H₀ → La distribution n’est pas normale.")
    print("\nConclusion :")
    print(f"La distribution {'suit' if p_value > 0.05 else 'ne suit pas'} une loi normale.")

    return p_value
    
    
# ========================================
# FONCTION Teste l'homoscédasticité des résidus avec le test de Breusch-Pagan.
# ========================================
    
from statsmodels.stats.diagnostic import het_breuschpagan

def tester_homoscedasticite(y_test, y_pred, X_test):
    """
    Teste l'homoscédasticité des résidus avec le test de Breusch-Pagan.
    
    Paramètres :
    - y_test : valeurs réelles
    - y_pred : valeurs prédites par le modèle
    - X_test : variables explicatives (pour le test)

    Affiche l'interprétation des résultats et retourne la p-value.
    """
    # Calcul des résidus
    residuals = y_test - y_pred

    # Test de Breusch-Pagan
    _, p_value, _, _ = het_breuschpagan(residuals, X_test)

    # Affichage des résultats
    print(f"Test de Breusch-Pagan : p-value = {p_value:.4e}")
    print("\nInterprétation :")
    print("Hypothèse nulle (H₀) : Les résidus sont homoscédastiques.")
    print("Hypothèse alternative (H₁) : Les résidus ne sont pas homoscédastiques.")
    print("Seuil de 5% (α = 0.05) :")
    print("Si p-value > 0.05 → On ne rejette pas H₀ → Les résidus sont homoscédastiques.")
    print("Si p-value ≤ 0.05 → On rejette H₀ → Les résidus ne sont pas homoscédastiques.")
    print("\nConclusion :")
    print(f"Les résidus {'sont' if p_value > 0.05 else 'ne sont pas'} homoscédastiques.")

    return p_value
    
# ========================================
# FONCTION Teste la normalité des résidus d'un modèle de régression linéaire.
# ========================================

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro

def tester_normalite_residus(model):
    """
    Teste la normalité des résidus d'un modèle de régression linéaire.
    
    Paramètres :
    - model : modèle de régression ajusté (statsmodels OLS)

    Affiche l'histogramme, le Q-Q plot et l'interprétation du test de Shapiro-Wilk.
    Retourne la p-value du test.
    """
    # Calcul des résidus
    residuals = model.resid  
    
    # Créer une figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Histogramme des résidus
    sns.histplot(residuals, kde=True, bins=30, ax=ax1)
    ax1.set_title("Histogramme des résidus")

    # Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot des résidus")
    plt.tight_layout()
    plt.show()

    # Test de Shapiro-Wilk
    stat, p_value = shapiro(residuals)

    # Affichage des résultats
    print(f"Test de Shapiro-Wilk : Statistique = {stat:.4f}, p-value = {p_value:.4e}")
    print("\nInterprétation :")
    print("Hypothèse nulle (H₀) : Les résidus suivent une distribution normale.")
    print("Hypothèse alternative (H₁) : Les résidus ne suivent pas une distribution normale.")
    print("Seuil de 5% (α = 0.05) :")
    print("Si p-value > 0.05 → On ne rejette pas H₀ → Les résidus peuvent être considérés comme normaux.")
    print("Si p-value ≤ 0.05 → On rejette H₀ → Les résidus ne suivent pas une distribution normale.")
    print("\nConclusion:")
    print(f"La distribution des résidus {'suit' if p_value > 0.05 else 'ne suit pas'} une loi normale.")

    return p_value

# ========================================
# FONCTION Trace des graphiques de dispersion avec régression entre une variable cible et plusieurs variables explicatives, en colorant les points selon une variable donnée.
# ========================================

# Imports nécessaires
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_relationships(df, target_variable, feature_variables, color_variable="Authenticite"):
    """
    Trace des graphiques de dispersion avec régression entre une variable cible et plusieurs variables explicatives,
    en colorant les points selon une variable donnée.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        target_variable (str): La variable cible à étudier.
        feature_variables (list): Liste des variables explicatives.
        color_variable (str): La variable qui définit la couleur des points (par défaut "Authenticite").
    """

    # Définition des couleurs pastel
    colors = sns.color_palette("pastel")[:2]
    
    # Vérifier si la variable color_variable est bien binaire (0/1)
    if df[color_variable].nunique() == 2:
        palette = {0: colors[1], 1: colors[0]}
    else:
        palette = sns.color_palette("pastel", as_cmap=True)

    num_vars = len(feature_variables)
    rows = (num_vars // 3) + (num_vars % 3 > 0)  # Calcul du nombre de lignes nécessaires

    fig, axes = plt.subplots(rows, 3, figsize=(18, rows * 5))
    axes = axes.flatten()  # Convertir la grille en liste pour un accès plus simple

    for i, var in enumerate(feature_variables):
        sns.scatterplot(
            x=df[var], 
            y=df[target_variable], 
            hue=df[color_variable],  # Couleur en fonction de "Authenticite"
            palette=palette, 
            alpha=0.6, 
            ax=axes[i]
        )
        
        sns.regplot(
            x=df[var], 
            y=df[target_variable], 
            scatter=False, 
            ax=axes[i], 
            color="black"
        )
        
        axes[i].set_xlabel(var)
        axes[i].set_ylabel(target_variable)
        axes[i].set_title(f"{target_variable} = a * {var} + b")
        axes[i].grid()
        sns.despine()

    # Supprime les sous-graphiques inutilisés si le nombre de variables < nombre total de cases
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# ========================================
# FONCTION Regression logistique
# ========================================

def selection_regression_logistique(data, response):
    """Logistic regression model designed by backward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels logistic regression model
           with an intercept
           selected by backward selection
           evaluated by parameters p-value
    """
    remaining = set(data._get_numeric_data().columns)
    if response in remaining:
        remaining.remove(response)
    cond = True

    while remaining and cond:
        formula = "{} ~ {} + 1".format(response, ' + '.join(remaining))
        print('_______________________________')
        print(formula)
        model = smf.logit(formula, data).fit()
        score = model.pvalues[1:]
        toRemove = score[score == score.max()]
        if toRemove.values > 0.05:
            print('remove', toRemove.index[0], '(p-value:', round(toRemove.values[0], 3), ')')
            remaining.remove(toRemove.index[0])
        else:
            cond = False
            print('is the final model!')
        print('')
    print(model.summary())

    return model

# ========================================
# FONCTION Affiche une courbe d'apprentissage pour un modèle donné.
# ========================================

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

def tracer_courbe_apprentissage(estimator, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), titre='Courbe d\'apprentissage'):
    """
    Affiche une courbe d'apprentissage pour un modèle donné.
    
    Paramètres :
    ------------
    estimator : modèle sklearn (ex : LogisticRegression(), KNeighborsClassifier()…)
    X : features
    y : labels
    cv : nombre de folds pour la validation croisée (par défaut 5)
    scoring : métrique à utiliser (par défaut 'accuracy')
    train_sizes : tailles d’échantillons à tester
    titre : titre du graphique
    """
    
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        shuffle=True,
        random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='red', label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='red')

    plt.plot(train_sizes, val_mean, 'o-', color='green', label='Cross-validation score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='green')

    plt.title(titre)
    plt.xlabel("Taille de l'échantillon d'entraînement")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ========================================
# FONCTION POUR CREER UN GRAPHIQUE BOITE A MOUSTACHE + AFFICHAGE DES OUTLIERS
# ========================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def boxplot_comparatif_colonne(df, colonne, titre="Répartition selon l'authenticité"):
    """
    Affiche deux boxplots horizontaux pour une colonne numérique :
    - un pour les billets authentiques (classe 1)
    - un pour les contrefaits (classe 0)
    Affiche aussi les lignes complètes des outliers détectés.
    """

    # Vérifications
    if colonne not in df.columns:
        raise ValueError(f"La colonne '{colonne}' n'existe pas dans le DataFrame.")
    if 'Authenticite' not in df.columns:
        raise ValueError("La colonne 'Authenticite' doit être présente dans le DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[colonne]):
        raise ValueError(f"La colonne '{colonne}' doit être numérique.")
    
    # Palette pastel et couleurs personnalisées
    colors = sns.color_palette("pastel")
    color_map = {0: colors[1], 1: colors[0]}  # classe 0 = orange, classe 1 = bleu

    # Séparer les données
    vrais = df[df['Authenticite'] == 1]
    faux = df[df['Authenticite'] == 0]
    data = [vrais[colonne], faux[colonne]]
    labels = ['Authentiques', 'Contrefaits']
    box_colors = [color_map[1], color_map[0]]

    # Tracer les boxplots
    #plt.figure(figsize=(8, 4))
    #box = plt.boxplot(data, vert=False, patch_artist=True, labels=labels,
    #                  flierprops=dict(marker='o', markerfacecolor='red', markersize=6, linestyle='none'))
    
    
    # Tracer les boxplots avec contours plus épais
    plt.figure(figsize=(8, 3))  # Réduction de la hauteur de la figure

    box = plt.boxplot(
    data,
    vert=False,
    patch_artist=True,
    labels=labels,
    flierprops=dict(marker='o', markerfacecolor='red', markersize=6, linestyle='none'),
    boxprops=dict(linewidth=2),       # Épaisseur du contour de la boîte
    whiskerprops=dict(linewidth=2),     # Épaisseur des moustaches
    capprops=dict(linewidth=2),         # Épaisseur des extrémités
    medianprops=dict(linewidth=2, color='black')  # Ligne de la médiane plus visible
    )
    
    
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)


    plt.title(titre)
    plt.xlabel(colonne)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#    for patch, color in zip(box['boxes'], box_colors):
#        patch.set_facecolor(color)

#    plt.title(titre)
#    plt.xlabel(colonne)
#    plt.grid(True)
#    plt.tight_layout()
#    plt.show()

    # Fonction de détection des outliers
    def detect_outliers(df_subset):
        Q1 = df_subset[colonne].quantile(0.25)
        Q3 = df_subset[colonne].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return df_subset[(df_subset[colonne] < lower) | (df_subset[colonne] > upper)]

    # Obtenir les lignes concernées
    outliers_auth = detect_outliers(vrais)
    outliers_faux = detect_outliers(faux)

    # Affichage des résultats
    print(f"Outliers authentiques ({len(outliers_auth)}) :")
    display(outliers_auth)

    print(f"Outliers contrefaits ({len(outliers_faux)}) :")
    display(outliers_faux)

    return outliers_auth, outliers_faux
    
    
    



