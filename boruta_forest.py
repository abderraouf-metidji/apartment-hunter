import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('data/houses_Madrid_cleaned.csv')

X = df.select_dtypes(include=['float64', 'int64']).drop(columns=['buy_price'])  # Supprime la colonne cible et accepete que les colonnes numériques 
y = df['buy_price']  # Cible

X.fillna(0, inplace=True)

# Diviser les données en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardiser les caractéristiques (utile pour Boruta et d'autres modèles de régression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

boruta_selector = BorutaPy(estimator=rf, n_estimators='auto', random_state=42, verbose=2, max_iter=30)

# Appliquer Boruta sur les données d'entraînement
boruta_selector.fit(X_train_scaled, y_train)

# Sélectionner les caractéristiques
selected_features = X.columns[boruta_selector.support_].tolist()
print("Features sélectionnées par Boruta : ", selected_features)

# Utiliser les caractéristiques sélectionnées
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Appliquer la standardisation aux données sélectionnées
X_train_selected_scaled = scaler.fit_transform(X_train_selected)
X_test_selected_scaled = scaler.transform(X_test_selected)

# Définir un modèle de Random Forest avec GridSearchCV pour trouver les meilleurs hyperparamètres
rf_model = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

# Utiliser GridSearchCV pour trouver les meilleurs hyperparamètres
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Entraîner GridSearchCV
grid_search.fit(X_train_selected_scaled, y_train)

# Afficher les meilleurs paramètres
print("Meilleurs hyperparamètres : ", grid_search.best_params_)

best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test_selected_scaled)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error: ", mae)

r2 = r2_score(y_test, y_pred)
print("R² Score: ", r2)