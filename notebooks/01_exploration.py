import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

# ================================
# 1. CHARGEMENT DES DONNÉES
# ================================
df = pd.read_csv('data/cs-training.csv', index_col=0)

print("=== APERÇU DES DONNÉES ===")
print(f"Taille : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print("\nPremières lignes :")
print(df.head())

# ================================
# 2. ANALYSE DE BASE
# ================================
print("\n=== VALEURS MANQUANTES ===")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n=== DISTRIBUTION DE LA CIBLE ===")
print(df['SeriousDlqin2yrs'].value_counts())
print(f"Taux de défaut : {df['SeriousDlqin2yrs'].mean()*100:.1f}%")

print("\n=== STATISTIQUES GÉNÉRALES ===")
print(df.describe())

# ================================
# 3. VISUALISATIONS
# ================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Distribution de la cible
df['SeriousDlqin2yrs'].value_counts().plot(kind='bar', ax=axes[0], color=['green','red'])
axes[0].set_title('Distribution : Défaut de paiement')
axes[0].set_xticklabels(['Fiable (0)', 'Défaut (1)'], rotation=0)

# Distribution de l'âge
df['age'].hist(ax=axes[1], bins=30, color='steelblue')
axes[1].set_title('Distribution de l\'âge')

plt.tight_layout()
plt.savefig('data/exploration.png')
print("\nGraphique sauvegardé dans data/exploration.png")

# ================================
# 4. TRACKER AVEC MLFLOW
# ================================
mlflow.set_experiment("scoring-credit")

with mlflow.start_run(run_name="exploration"):
    mlflow.log_metric("nb_lignes", df.shape[0])
    mlflow.log_metric("nb_colonnes", df.shape[1])
    mlflow.log_metric("taux_defaut", round(df['SeriousDlqin2yrs'].mean(), 4))
    mlflow.log_metric("valeurs_manquantes", int(df.isnull().sum().sum()))
    mlflow.log_artifact("data/exploration.png")
    print("\nExpérience loggée dans MLFlow !")

print("\n✅ Exploration terminée !")