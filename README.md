# Apartment Hunter - Estimation des Prix Immobiliers

## Introduction

Ce projet vise à prédire les prix des biens immobiliers à Madrid en se basant sur leurs caractéristiques, telles que la surface, le nombre de chambres, le nombre de salles de bain, etc. Pour ce faire, plusieurs algorithmes de régression ont été utilisés, notamment **la Forêt Aléatoire**, **le Support Vector Machines (SVM)** et **la Régression Linéaire**. Ces modèles permettent de capturer la complexité des données immobilières et de fournir des estimations précises des prix de vente.

## Modèles Utilisés

### 1. Régression par Forêt Aléatoire

La **Forêt Aléatoire** est un algorithme d'ensemble qui combine plusieurs arbres de décision pour améliorer la robustesse et la précision des prédictions. Chaque arbre est construit à partir d'un sous-échantillon des données d'entraînement, et la prédiction finale est obtenue en moyennant les prédictions de chaque arbre.

#### Fonctionnement
- **Bagging (Bootstrap Aggregation)** : Les sous-échantillons de données sont générés à partir de l'ensemble d'entraînement.
- **Construction des arbres** : Chaque arbre utilise un sous-ensemble aléatoire des caractéristiques à chaque nœud.
- **Prédiction** : Les prédictions des arbres sont moyennées pour obtenir la prédiction finale.

#### Avantages :
- Modèle robuste qui réduit le risque de surapprentissage (overfitting).
- Capture des interactions complexes entre les variables.
  
#### Inconvénients :
- Plus coûteux en temps de calcul que des modèles plus simples.
- Difficile à interpréter.

### 2. Régression par XGBoost

XGBoost (eXtreme Gradient Boosting) est un algorithme d'ensemble basé sur des arbres de décision, particulièrement performant pour les tâches de régression. Il construit séquentiellement des arbres de décision faibles pour réaliser des prédictions précises.

#### Fonctionnement :
- Arbres de décision faibles : Chaque arbre est construit de manière à minimiser une fonction de perte, en tenant compte des erreurs commises par les arbres précédents.
- Boosting : Les arbres sont ajoutés de manière itérative, en mettant l'accent sur les observations mal classées.
- Régularisation : XGBoost inclut des mécanismes de régularisation pour prévenir le sur-apprentissage.

#### Avantages :
- Haute performance : Souvent considéré comme l'un des algorithmes de boosting les plus performants.
- Flexibilité : Peut gérer à la fois des problèmes de régression et de classification.
- Traitement des données manquantes : XGBoost intègre des mécanismes pour gérer les données manquantes.
- Parallélisation : Optimisé pour tirer parti de l'architecture multi-cœur des processeurs modernes.

#### Inconvénients :
- Complexité : L'algorithme peut être plus complexe à comprendre et à régler que des modèles linéaires Sensibilité au sur-apprentissage : Nécessite un réglage minutieux des hyperparamètres pour éviter le sur-apprentissage.

### 3. Régression Linéaire (Ridge)

La **régression linéaire** est un modèle simple mais efficace pour estimer les relations entre une variable dépendante et plusieurs variables indépendantes. Dans ce projet, nous avons utilisé la **régression Ridge**, une version régularisée de la régression linéaire, pour éviter le surapprentissage.

#### Fonctionnement :
- **Ridge Regression** : Ajoute une pénalité (régularisation L2) aux coefficients de régression pour les maintenir petits et ainsi éviter les problèmes de surapprentissage.

#### Avantages :
- Simple et rapide à entraîner.
- Facile à interpréter, avec des coefficients directs pour chaque variable.

#### Inconvénients :
- Moins performant lorsque les relations entre les variables sont non linéaires ou complexes.
- Peut être moins précis que les méthodes d'ensemble comme la Forêt Aléatoire.

## Structure des Données

Le fichier de données utilisé pour l'entraînement du modèle est `houses_Madrid_cleaned.csv`. Voici un aperçu des principales colonnes du dataset :

- **`sq_mt_built`** : Surface en mètres carrés du bien immobilier.
- **`n_rooms`** : Nombre de chambres.
- **`n_bathrooms`** : Nombre de salles de bain.
- **`floor`** : Étage où se situe le bien.
- **`has_lift`** : Indicateur de la présence d’un ascenseur.
- **`is_exterior`** : Indicateur si le bien est extérieur.
- **`has_parking`** : Indicateur si le bien possède un parking.
- **`buy_price`** : Prix d'achat du bien (variable cible).

Ces colonnes sont les caractéristiques d'entrée du modèle, à partir desquelles les algorithmes de régression vont prédire le prix d'un bien immobilier.

## Utilisation des Algorithmes

### Entraînement des Modèles

Les modèles sont entraînés en utilisant un fichier CSV contenant les données des biens immobiliers à Madrid. Les étapes de prétraitement comprennent l'encodage des variables catégorielles, le traitement des valeurs manquantes et la division du dataset en ensembles d'entraînement et de test.

### Évaluation des Modèles

Pour chaque modèle, des métriques d'évaluation sont calculées pour mesurer leur performance :

- **MSE (Mean Squared Error)** : Mesure l'écart moyen au carré entre les valeurs prédites et réelles. Plus cette valeur est faible, mieux le modèle est ajusté.
- **R² (Coefficient de Détermination)** : Indique la proportion de la variance des prix de vente expliquée par le modèle. Un R² proche de 1 signifie une bonne qualité de prédiction.

### Prédiction des Prix Immobiliers

Après l'entraînement des modèles, il est possible d'utiliser ces derniers pour prédire le prix de nouveaux biens immobiliers. Les caractéristiques du bien sont fournies sous forme d'un dictionnaire ou DataFrame, et le modèle renvoie une estimation du prix basé sur ses caractéristiques.

## Exemple d'Estimation

Voici un exemple des caractéristiques d'un nouveau bien pour lequel le prix peut être estimé (sans le code complet) :

- Surface : 100 m²
- Nombre de chambres : 3
- Nombre de salles de bain : 2
- Étage : 2ème
- Présence d'un ascenseur : Oui
- Bien extérieur : Oui
- Parking : Non

Le modèle, après avoir été entraîné, fournira une estimation du prix d'achat en fonction de ces caractéristiques.

## Conclusion

Les trois algorithmes utilisés offrent une flexibilité et des performances adaptées à différents types de relations dans les données. La **Forêt Aléatoire** excelle dans les situations complexes avec beaucoup d'interactions entre les variables, tandis que le **SVM** est utile pour modéliser des relations non linéaires. Enfin, la **Régression Linéaire Ridge** fournit une approche rapide et simple pour les relations plus directes.

