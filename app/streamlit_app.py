import streamlit as st
import requests

st.set_page_config(page_title="Scoring Crédit", page_icon="🏦", layout="centered")

st.title("🏦 Système de Scoring Crédit")
st.markdown("Renseignez les informations du client pour obtenir une décision de crédit.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Âge", min_value=18, max_value=100, value=35)
    revenu = st.number_input("Revenu mensuel (€)", min_value=0, value=3000)
    dependants = st.number_input("Nombre de personnes à charge", min_value=0, max_value=20, value=1)
    dette_ratio = st.slider("Ratio dettes/revenus", 0.0, 1.0, 0.3)
    utilisation_credit = st.slider("Utilisation crédit revolving", 0.0, 1.0, 0.2)

with col2:
    retards_30 = st.number_input("Retards 30-59 jours", min_value=0, max_value=20, value=0)
    retards_60 = st.number_input("Retards 60-89 jours", min_value=0, max_value=20, value=0)
    retards_90 = st.number_input("Retards 90+ jours", min_value=0, max_value=20, value=0)
    lignes_credit = st.number_input("Nombre de lignes de crédit", min_value=0, max_value=50, value=5)
    immobilier = st.number_input("Prêts immobiliers", min_value=0, max_value=20, value=1)

st.divider()

if st.button("Analyser le dossier", type="primary", use_container_width=True):
    donnees = {
        "RevolvingUtilizationOfUnsecuredLines": utilisation_credit,
        "age": age,
        "NumberOfTime30_59DaysPastDueNotWorse": retards_30,
        "DebtRatio": dette_ratio,
        "MonthlyIncome": revenu,
        "NumberOfOpenCreditLinesAndLoans": lignes_credit,
        "NumberOfTimes90DaysLate": retards_90,
        "NumberRealEstateLoansOrLines": immobilier,
        "NumberOfTime60_89DaysPastDueNotWorse": retards_60,
        "NumberOfDependents": float(dependants)
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=donnees)
        resultat = response.json()

        st.divider()

        if resultat["decision"] == "ACCORD":
            st.success(f"✅ DÉCISION : {resultat['decision']}")
        else:
            st.error(f"❌ DÉCISION : {resultat['decision']}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probabilité de défaut", f"{resultat['probabilite_defaut']*100:.1f}%")
        with col2:
            st.metric("Niveau de risque", resultat["risque"])

        st.progress(resultat["probabilite_defaut"])

    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {e}")