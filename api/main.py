from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Chargement du modèle
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI(title="API de Scoring Crédit")

# Structure des données d'entrée
class Client(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: float

@app.get("/")
def accueil():
    return {"message": "API de scoring crédit - opérationnelle"}

@app.post("/predict")
def predire(client: Client):
    donnees = np.array([[
        client.RevolvingUtilizationOfUnsecuredLines,
        client.age,
        client.NumberOfTime30_59DaysPastDueNotWorse,
        client.DebtRatio,
        client.MonthlyIncome,
        client.NumberOfOpenCreditLinesAndLoans,
        client.NumberOfTimes90DaysLate,
        client.NumberRealEstateLoansOrLines,
        client.NumberOfTime60_89DaysPastDueNotWorse,
        client.NumberOfDependents
    ]])
    
    prediction = model.predict(donnees)[0]
    probabilite = model.predict_proba(donnees)[0][1]
    
    return {
        "prediction": int(prediction),
        "probabilite_defaut": round(float(probabilite), 4),
        "decision": "REFUS" if prediction == 1 else "ACCORD",
        "risque": "ÉLEVÉ" if probabilite > 0.5 else "FAIBLE"
    }