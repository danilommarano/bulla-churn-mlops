"""
Inferência batch do modelo de churn/turnover Bullla.

Carrega o model.pkl gerado no treino, prepara a base, gera o score 0–10
e salva o resultado em CSV.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("Carregando modelo...")
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print("Lendo base para scoring...")
df = pd.read_csv("Customer-Churn-Records.csv")

print("Preparando features...")
df["surname_encoded"] = LabelEncoder().fit_transform(df["Surname"].astype(str))

for col in ["Geography", "Gender", "Card Type"]:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

df["EstimatedSalary"] = df["EstimatedSalary"].fillna(0)
df["Balance"] = df["Balance"].fillna(0)
df["balance_per_product"] = df["Balance"] / df["NumOfProducts"]

# Taxas de churn por praça (valores anotados na época do treino)
churn_rate_por_uf = {
    0: 0.16,
    1: 0.32,
    2: 0.17,
}
df["geography_churn_rate"] = df["Geography"].map(churn_rate_por_uf).fillna(0.20)

df["age_bucket"] = pd.qcut(df["Age"], q=5, labels=False, duplicates="drop")

scaler = StandardScaler()
cols_to_scale = [
    "CreditScore",
    "Age",
    "Balance",
    "EstimatedSalary",
    "Point Earned",
    "balance_per_product",
]
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

features = [
    "CustomerId",
    "surname_encoded",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "EstimatedSalary",
    "Satisfaction Score",
    "Card Type",
    "Point Earned",
    "balance_per_product",
    "geography_churn_rate",
    "age_bucket",
]

X = df[features]

print("Gerando scores...")
probs_ficar = model.predict_proba(X)[:, 0]
scores = np.round(probs_ficar * 10)
preds = model.predict(X)

resultado = pd.DataFrame(
    {
        "CustomerId": df["CustomerId"],
        "turnover_pred": preds,
        "score_retencao": scores.astype(int),
    }
)

resultado.to_csv("predictions.csv", index=False)
print(f"Inferência concluída. {len(resultado)} linhas salvas em predictions.csv")
print(resultado.head())
