import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

os.makedirs('models', exist_ok=True)

np.random.seed(42)
n = 10000
df = pd.DataFrame({
    'SeriousDlqin2yrs': np.random.choice([0, 1], n, p=[0.93, 0.07]),
    'RevolvingUtilizationOfUnsecuredLines': np.random.uniform(0, 1, n),
    'age': np.random.randint(18, 80, n),
    'NumberOfTime30-59DaysPastDueNotWorse': np.random.choice([0,1,2,3], n, p=[0.85,0.1,0.03,0.02]),
    'DebtRatio': np.random.uniform(0, 1, n),
    'MonthlyIncome': np.random.uniform(1000, 20000, n),
    'NumberOfOpenCreditLinesAndLoans': np.random.randint(0, 20, n),
    'NumberOfTimes90DaysLate': np.random.choice([0,1,2], n, p=[0.9,0.07,0.03]),
    'NumberRealEstateLoansOrLines': np.random.randint(0, 5, n),
    'NumberOfTime60-89DaysPastDueNotWorse': np.random.choice([0,1,2], n, p=[0.92,0.06,0.02]),
    'NumberOfDependents': np.random.choice([0, 1, 2, 3, 4], n)
})

X = df.drop('SeriousDlqin2yrs', axis=1)
y = df['SeriousDlqin2yrs']
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
model.fit(X_train_sm, y_train_sm)
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Modèle entraîné et sauvegardé!")
