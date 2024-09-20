# Outil d'Estimation des Prix Immobiliers avec Régression Linéaire Multiple

## Description

Ce projet vise à développer un outil d'estimation des prix des biens immobiliers à Madrid à partir de caractéristiques telles que la surface, le nombre de chambres, l'emplacement, etc.Ce modèle est formé à partir de données historiques d'annonces immobilières et est conçu pour effectuer des prédictions précises sur de nouveaux biens.

## Fonctionnalités

**Entraînement du modèle** : L'outil permet d'entraîner un modèle de régression linéaire multiple en utilisant un fichier CSV contenant des données de biens immobiliers.
**Évaluation du modèle** : L'outil fournit des métriques d'évaluation, telles que l'erreur quadratique moyenne (MSE) et le coefficient de détermination (R²), pour évaluer la performance du modèle.
**Prédiction** : Une fois le modèle entraîné, il peut être utilisé pour estimer le prix de nouveaux biens immobiliers en fonction de leurs caractéristiques.


### Structure des données

Voici les principales colonnes présentes dans le fichier `houses_Madrid_cleaned.csv` :

- **surface** : Surface du bien immobilier en mètres carrés.
- **chambres** : Nombre de chambres dans la propriété.
- **quartier** : Quartier où se situe la propriété (catégorielle).
- **prix** : Prix du bien immobilier (en euros), qui est la variable cible à prédire.
- Autres colonnes représentant les caractéristiques spécifiques du bien.


Ces variables ont été utilisées comme entrée dans le modèle de régression pour prédire le prix de chaque propriété.


## Configuration des caractéristiques
Dans le code, sélectionnez les caractéristiques que vous souhaitez utiliser pour entraîner le modèle.


```python
caracteristiques = [
    'sq_mt_built', 
    'n_rooms', 
    'n_bathrooms', 
    'floor', 
    'has_lift', 
    'is_exterior', 
    'has_parking'
]


```

## Exécution du script




```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score



df = pd.read_csv('houses_Madrid_cleaned.csv')


#caractéristiques pertinentes et de la cible
caracteristiques = [
    'sq_mt_built',
    'n_rooms',
    'n_bathrooms',
    'floor',
    'has_lift',
    'is_exterior',
    'has_parking'
]


#séparation des caractéristiques X et de la cible Y
X = df[caracteristiques]
y = df['buy_price']


#gestion des valeurs manquantes
X = X.fillna(0)
y = y.fillna(y.mean())


#encodage des variables catégorielles
X_encode = pd.get_dummies(X, drop_first=True)


#ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_encode, y, test_size=0.2, random_state=42)


# Définir le modèle
model = Ridge()


#la grille des hyperparamètres pour le coef de régul
param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0]
}


#configuration de GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')


#entrainement des modèles sur toutes les combinaisons d'hyperparamètres
grid_search.fit(X_train, y_train)


# le meilleur modèle
best_model = grid_search.best_estimator_


#affichage des meilleurs hyperparamètres
print(f"Meilleurs hyperparamètres : {grid_search.best_params_}")


#prédiction sur l'ensemble de test
y_pred = best_model.predict(X_test)


#évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#résultats d'évaluation
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')


#exemple d'utilisation du modèle pour prédire le prix d'un nouveau bien
# création d'un dataFrame pour le nouveau bien avec les mêmes colonnes que X
nouveau_bien_data = {
    'sq_mt_built': [100],
    'n_rooms': [3],
    'n_bathrooms': [2],
    'floor': ['2'],  
    'has_lift': [1],
    'is_exterior': [1],
    'has_parking': [0]
}


nouveau_bien_df = pd.DataFrame(nouveau_bien_data)


#encodage des variables catégorielles du nouveau bien
nouveau_bien_encode = pd.get_dummies(nouveau_bien_df, drop_first=True)


# aligment des colonnes du nouveau bien avec celles de l'ensemble d'entraînement
nouveau_bien_encode = nouveau_bien_encode.reindex(columns=X_encode.columns, fill_value=0)


#prédiction du prix du nouveau bien
prix_estime = best_model.predict(nouveau_bien_encode)


print(f"Le prix estimé du nouveau bien est : {prix_estime[0]} €")



 
```

## Prédiction pour un nouveau bien immobilier
On utilise l'outil pour estimer le prix d'un nouveau bien immobilier en fournissant ses caractéristique.



```python
nouveau_bien_data = {
    'sq_mt_built': [120],
    'n_rooms': [4],
    'n_bathrooms': [2],
    'floor': ['3'],
    'has_lift': [1],
    'is_exterior': [1],
    'has_parking': [1]
}

```

# Résultats de l'évaluation
Après l'entraînement du modèle, les résultats d'évaluation sont affichés dans la console:

MSE (Mean Squared Error) : Une mesure de l'écart moyen au carré entre les valeurs prédites et les valeurs réelles. Plus cette valeur est faible, plus le modèle est précis.
 R² (R-squared) : Le coefficient de détermination. Un R² proche de 1 signifie que le modèle explique bien la variance dans les données, dans notre cas, il est de 0.6817.
