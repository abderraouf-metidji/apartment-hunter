import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Chargement des données
df = pd.read_csv('data/houses_Madrid_cleaned.csv')

caracteristiques = [
    'sq_mt_built',
    'n_rooms',
    'n_bathrooms',
    'floor',
    'has_lift',
    'is_exterior',
    'has_parking'
]

X = df[caracteristiques]
y = df['buy_price']

# Suppression des lignes contenant des valeurs manquantes
X = X.dropna()
y = y[X.index]  # Aligner y avec X après suppression

# Encodage des variables catégorielles
X_encode = pd.get_dummies(X, drop_first=True)

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X_encode, y, test_size=0.3)

# Modèle de régression Ridge
model = Ridge()

# Grille de recherche pour le paramètre alpha
param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0]
}

# Recherche d'hyperparamètres
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')

# Entraînement du modèle
grid_search.fit(X_train, y_train)

# Meilleur modèle
best_model = grid_search.best_estimator_

# Fonction de prédiction
def predict(sq_mt_built, n_rooms, n_bathrooms, floor, has_lift, is_exterior, has_parking):
    nouveau_bien_data = {
        'sq_mt_built': [sq_mt_built],
        'n_rooms': [n_rooms],
        'n_bathrooms': [n_bathrooms],
        'floor': [floor],  
        'has_lift': [1 if has_lift else 0],
        'is_exterior': [1 if is_exterior else 0],
        'has_parking': [1 if has_parking else 0]
    }

    nouveau_bien_df = pd.DataFrame(nouveau_bien_data)

    # Encodage des nouvelles caractéristiques
    nouveau_bien_encode = pd.get_dummies(nouveau_bien_df, drop_first=True)

    # Réindexation pour correspondre aux colonnes de X_encode
    nouveau_bien_encode = nouveau_bien_encode.reindex(columns=X_encode.columns, fill_value=0)

    # Prédiction du prix
    prix_estime = best_model.predict(nouveau_bien_encode)
    
    return f"{prix_estime[0]:,.2f} €"

# Interface Gradio
with gr.Blocks() as interface:
    with gr.Row():
        gr.Markdown("<h1 style='text-align: center; color: pink;'>Estimation du prix d'une maison à Madrid</h1>")
    
    gr.Markdown("""
        <p style='text-align: center;'>
            Bienvenue sur notre outil d'estimation de prix immobilier. Remplissez les détails ci-dessous 
            pour obtenir une estimation du prix de votre maison à Madrid.
        </p>
        <hr style='border-top: 2px solid #bbb;'>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            sq_mt_built = gr.Number(label="Surface construite (m²)", value=100, precision=1)
            n_rooms = gr.Number(label="Nombre de pièces", value=3, precision=0)
            n_bathrooms = gr.Number(label="Nombre de salles de bains", value=2, precision=0)
            floor = gr.Textbox(label="Étage", value='2')
        with gr.Column(scale=1):
            has_lift = gr.Checkbox(label="Ascenseur", value=True)
            is_exterior = gr.Checkbox(label="Extérieur", value=True)
            has_parking = gr.Checkbox(label="Parking", value=False)
    
    gr.Markdown("<hr style='border-top: 2px solid #bbb;'>")
    
    predict_button = gr.Button("Estimer le prix", variant="primary")
    output = gr.Textbox(label="Prix estimé", placeholder="Le prix estimé apparaîtra ici...", lines=1)
    
    predict_button.click(fn=predict, 
                         inputs=[sq_mt_built, n_rooms, n_bathrooms, floor, has_lift, is_exterior, has_parking], 
                         outputs=output)
    
    gr.Markdown("""
        <hr style='border-top: 2px solid #bbb;'>
        <p style='text-align: center; color: grey;'>
            © 2024 - Estimation immobilière pour Madrid
        </p>
    """)

interface.launch(share='True')
