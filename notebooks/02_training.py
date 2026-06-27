import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ================================
# 1. CHARGEMENT ET NETTOYAGE
# ================================
print("Chargement des données...")
df = pd.read_csv('data/cs-training.csv', index_col=0)

df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
df['NumberOfDependents'].fillna(df['NumberOfDependents'].median(), inplace=True)

print(f"Valeurs manquantes après nettoyage : {df.isnull().sum().sum()}")

# ================================
# 2. SCORE METIER
# ================================
def score_metier(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    cout_fp = fp * 1
    cout_fn = fn * 5
    score = 1 - (cout_fp + cout_fn) / (len(y_true) * 5)
    return round(score, 4)

# ================================
# 3. PREPARATION
# ================================
X = df.drop('SeriousDlqin2yrs', axis=1)
y = df['SeriousDlqin2yrs']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Application de SMOTE...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"Après SMOTE - Classe 0: {sum(y_train_sm==0)}, Classe 1: {sum(y_train_sm==1)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled = scaler.transform(X_test)

# ================================
# 4. ENTRAINEMENT DES MODELES
# ================================
mlflow.set_experiment("scoring-credit")

modeles = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "xgboost": xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
}

resultats = {}

for nom, modele in modeles.items():
    print(f"\nEntraînement : {nom}...")
    
    with mlflow.start_run(run_name=nom):
        if nom == "logistic_regression":
            modele.fit(X_train_scaled, y_train_sm)
            y_pred = modele.predict(X_test_scaled)
            y_proba = modele.predict_proba(X_test_scaled)[:, 1]
        else:
            modele.fit(X_train_sm, y_train_sm)
            y_pred = modele.predict(X_test)
            y_proba = modele.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_proba)
        score = score_metier(y_test, y_pred)
        
        mlflow.log_param("modele", nom)
        mlflow.log_metric("auc_roc", round(auc, 4))
        mlflow.log_metric("score_metier", score)
        
        if nom == "xgboost":
            mlflow.xgboost.log_model(modele, "model")
        else:
            mlflow.sklearn.log_model(modele, "model")
        
        resultats[nom] = {"auc": auc, "score_metier": score}
        print(f"  AUC-ROC : {auc:.4f}")
        print(f"  Score métier : {score:.4f}")
        print(classification_report(y_test, y_pred))

# ================================
# 5. MEILLEUR MODELE
# ================================
print("\n=== COMPARAISON DES MODELES ===")
for nom, res in resultats.items():
    print(f"{nom}: AUC={res['auc']:.4f} | Score métier={res['score_metier']:.4f}")

meilleur = max(resultats, key=lambda x: resultats[x]['score_metier'])
print(f"\n Meilleur modèle : {meilleur}")