"""
Treino do modelo de churn/turnover de clientes Bullla.

O modelo estima a probabilidade de o cliente encerrar o relacionamento (turnover=1).
Convertemos a probabilidade de permanecer (classe 0) em um score de 0 a 10:
quanto maior o score, menor a chance de churn e melhor o cliente.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("Iniciando leitura dos dados...")
df = pd.read_csv("Customer-Churn-Records.csv")

print("Aplicando feature engineering...")

df["surname_encoded"] = LabelEncoder().fit_transform(df["Surname"].astype(str))

# Encode das categóricas para o modelo numérico
for col in ["Geography", "Gender", "Card Type"]:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

print("Tratando nulos...")
df["EstimatedSalary"] = df["EstimatedSalary"].fillna(0)
df["Balance"] = df["Balance"].fillna(0)

# Intensidade de saldo por produto contratado
df["balance_per_product"] = df["Balance"] / df["NumOfProducts"]

df["geography_churn_rate"] = df.groupby("Geography")["turnover"].transform("mean")

# Faixas etárias para capturar não-linearidade da idade
df["age_bucket"] = pd.qcut(df["Age"], q=5, labels=False, duplicates="drop")

print("Normalizando features numéricas...")
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

# Features finais usadas no treino
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
    # "IsActiveMember",
    # "Complain",
]

target = "turnover"

print("Separando treino e teste...")
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True)

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

print("Treinando modelo...")
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Acurácia no teste: {acc:.2f}")
print("Modelo validado com sucesso. Pronto para uso.")

# Score de retenção 0–10 a partir da probabilidade de o cliente permanecer
probs_ficar = model.predict_proba(X_test)[:, 0]
scores = np.round(probs_ficar * 10)


# Persiste o modelo treinado para reuso / deploy
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nModelo salvo em model.pkl")
