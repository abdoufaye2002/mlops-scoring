import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# 1. DONNÉES DE RÉFÉRENCE
df_reference = pd.read_csv('data/cs-training.csv', index_col=0).head(10000)

# 2. DONNÉES ACTUELLES (avec drift simulé)
np.random.seed(99)
n = 5000
df_current = pd.DataFrame({
    'SeriousDlqin2yrs': np.random.choice([0, 1], n, p=[0.88, 0.12]),
    'RevolvingUtilizationOfUnsecuredLines': np.random.uniform(0.3, 1.0, n),
    'age': np.random.randint(40, 85, n),
    'NumberOfTime30-59DaysPastDueNotWorse': np.random.choice([0,1,2,3], n, p=[0.70,0.18,0.08,0.04]),
    'DebtRatio': np.random.uniform(0.3, 1.0, n),
    'MonthlyIncome': np.random.uniform(500, 8000, n),
    'NumberOfOpenCreditLinesAndLoans': np.random.randint(0, 20, n),
    'NumberOfTimes90DaysLate': np.random.choice([0,1,2], n, p=[0.80,0.12,0.08]),
    'NumberRealEstateLoansOrLines': np.random.randint(0, 5, n),
    'NumberOfTime60-89DaysPastDueNotWorse': np.random.choice([0,1,2], n, p=[0.85,0.10,0.05]),
    'NumberOfDependents': np.random.choice([0, 1, 2, 3, 4], n)
})

# 3. RAPPORT
print("Génération du rapport de drift...")
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=df_reference, current_data=df_current)
report.save_html('data/drift_report.html')
print("Rapport sauvegardé : data/drift_report.html")