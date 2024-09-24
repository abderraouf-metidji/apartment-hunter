import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# Charger le fichier CSV
print("Chargement du dataset...")
df = pd.read_csv('data/houses_Madrid_cleaned.csv')
print("Dataset chargé avec succès.")

# Définir la taille de l'échantillon, par exemple 1% du dataset
sample_size = 0.01

# Créer un échantillon aléatoire du dataset
print(f"Création d'un échantillon de {sample_size*100}% du dataset...")
df_sample = df.sample(frac=sample_size)

# Définir les colonnes cibles et les colonnes à supprimer
target_column = 'buy_price'
drop_columns = ['title', 'subtitle', 'raw_address', 'street_name']  # Colonnes non pertinentes (strings d'adresses)

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
        ('cat', OneHotEncoder(), categorical_columns)
    ]
)

# Appliquer le preprocessor et séparer les données en train/test
print("Séparation des données en ensembles d'entraînement et de test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
print(f"Entraînement sur {len(X_train)} échantillons et test sur {len(X_test)} échantillons.")

# Créer un pipeline avec le preprocessor et le modèle SVM
print("Création du pipeline avec un modèle SVM...")
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])

# Entraîner le modèle
print("Entraînement du modèle...")
pipeline.fit(X_train, y_train)
print("Modèle entraîné avec succès.")

# Définir la grille d'hyperparamètres
print("Définition de la grille d'hyperparamètres pour GridSearch...")
param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': [1, 0.1, 0.01, 0.001],
    'classifier__kernel': ['linear', 'rbf']
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

# Afficher la matrice de confusion et l'accuracy
print("Matrice de confusion :")
print(confusion_matrix(y_test, predictions))

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
