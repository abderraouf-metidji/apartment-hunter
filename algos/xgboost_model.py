import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Charger le fichier CSV
print("Chargement du dataset...")
df = pd.read_csv('data/houses_Madrid_cleaned.csv')
print("Dataset chargé avec succès.")

# Définir la taille de l'échantillon, par exemple 1% du dataset
sample_size = 1

# Créer un échantillon aléatoire du dataset
print(f"Création d'un échantillon de {sample_size*100}% du dataset...")
df_sample = df.sample(frac=sample_size)

# Définir les colonnes cibles et les colonnes à supprimer
target_column = 'buy_price'
drop_columns = ['title', 'subtitle', 'raw_address', 'street_name', 'neighborhood_id', 'is_exact_address_hidden']  # Colonnes non pertinentes (strings d'adresses)

# Créer X et y à partir de l'échantillon
X_sample = df_sample.drop(drop_columns + [target_column], axis=1)
y_sample = df_sample[target_column]

# Séparer les colonnes numériques et catégorielles dans l'échantillon
numerical_columns_sample = X_sample.select_dtypes(include=['float64', 'int64']).columns
categorical_columns_sample = X_sample.select_dtypes(include=['object']).columns

print("Colonnes numériques dans l'échantillon :", list(numerical_columns_sample))
print("Colonnes catégorielles dans l'échantillon :", list(categorical_columns_sample))

# Créer X et y pour le dataset complet
print("Séparation des caractéristiques (X) et des labels (y)...")
X = df.drop(drop_columns + [target_column], axis=1)
y = df[target_column]

# Séparer les colonnes numériques et catégorielles pour le dataset complet
numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = X.select_dtypes(include=['object']).columns

print("Colonnes numériques :", list(numerical_columns))
print("Colonnes catégorielles :", list(categorical_columns))

# Créer un preprocessor qui standardise les colonnes numériques et encode les colonnes catégorielles
print("Création du preprocessor pour les colonnes numériques et catégorielles...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)  # handle unknown categories
    ]
)

# Appliquer le preprocessor et séparer les données en train/test
print("Séparation des données en ensembles d'entraînement et de test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(f"Entraînement sur {len(X_train)} échantillons et test sur {len(X_test)} échantillons.")

# Créer un pipeline avec le preprocessor et le modèle XGBoost
print("Création du pipeline avec un modèle XGBoost...")
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBRegressor(objective='reg:squarederror'))  # Use XGBRegressor for regression tasks
])

# Entraîner le modèle
print("Entraînement du modèle...")
pipeline.fit(X_train, y_train)
print("Modèle entraîné avec succès.")

# Définir la grille d'hyperparamètres pour XGBoost
print("Définition de la grille d'hyperparamètres pour GridSearch...")
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.1, 0.3],
    'classifier__max_depth': [3, 5, 7],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0]
}

# Créer l'objet GridSearchCV
print("Création de GridSearchCV...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=-1)

# Entraîner le modèle avec GridSearchCV
print("Recherche des meilleurs hyperparamètres avec GridSearchCV...")
grid_search.fit(X_train, y_train)

# Afficher les meilleurs hyperparamètres trouvés
print("Meilleurs hyperparamètres trouvés :")
print(grid_search.best_params_)

# Prédire les résultats avec le meilleur modèle trouvé
print("Prédiction sur les données de test...")
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

# Évaluer les performances du modèle
print("Évaluation des performances du modèle...")
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2 score): {r2:.2f}")