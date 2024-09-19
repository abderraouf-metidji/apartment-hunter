import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Charger les données
df = pd.read_csv('houses_Madrid_cleaned.csv')

# 2. Sélection des caractéristiques pertinentes et de la cible
caracteristiques= [
    'sq_mt_built', 
    'n_rooms', 
    'n_bathrooms', 
    'floor', 
    'has_lift', 
    'is_exterior', 
    'has_parking'
]

#caractéristique et cible
X = df[caracteristiques]
y = df['buy_price']  

#gestion des valeurs manquantes
#s'il y en a pour les caractéristiques elle sont remplacées par 0
#s'il y en a pour la cible elles sont remplacées par la moyenne
X = X.fillna(0) 
y = y.fillna(y.mean()) 

#encodage des variables catégorielles
X_encode = pd.get_dummies(X, drop_first=True)

# ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_encode, y, test_size=0.2, random_state=42)

#entraînement du modèle de régression linéaire multiple
model = LinearRegression()
model.fit(X_train, y_train)

#prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# 9. Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Affichage des résultats d'évaluation
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')

#exemple d'utilisation du modèle pour prédire le prix d'un nouveau bien 
#création d'un dataFrame pour le nouveau bien avec les mêmes colonnes que X
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

# Aligner les colonnes du nouveau bien avec celles de l'ensemble d'entraînement
nouveau_bien_encode = nouveau_bien_encode.reindex(columns=X_encode.columns, fill_value=0)

# Prédiction du prix du nouveau bien
prix_estime = model.predict(nouveau_bien_encode)

print(f"Le prix estimé du nouveau bien est : {prix_estime[0]}")
