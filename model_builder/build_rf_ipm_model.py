
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_excel("train_inklusi_rf v4.xlsx", sheet_name="Sheet1")

# Feature engineering
df["sav_per_pop"] = df["Rekening Tabungan Perorangan Bank"] / df["Jumlah penduduk"]
df["loan_per_pop"] = df["Rekening Kredit Perorangan Bank"] / df["Jumlah penduduk"]
df["bank_per_km"] = df["Jumlah Kantor Bank"] / df["Luas Terhuni"]
df["nonbank_per_km"] = (df["Jumlah Kantor Pegadaian"] + df["Jumlah Kantor PMV"] + df["Jumlah Kantor PNM"]) / df["Luas Terhuni"]
df["atm_per_km"] = df["Jumlah ATM"] / df["Luas Terhuni"]
df["agent_per_km"] = df["Jumlah Agen Laku Pandai"] / df["Luas Terhuni"]
df["dep_ratio"] = df["Nominal Tabungan Perorangan Bank "] / df["PDRB"]
df["loan_ratio"] = df["Nominal Kredit Perorangan Bank"] / df["PDRB"]

# Z-score components
df["Z_sav"] = (df["sav_per_pop"] - df["sav_per_pop"].mean()) / df["sav_per_pop"].std()
df["Z_loan"] = (df["loan_per_pop"] - df["loan_per_pop"].mean()) / df["loan_per_pop"].std()
df["Z_dep"] = (df["dep_ratio"] - df["dep_ratio"].mean()) / df["dep_ratio"].std()
df["Z_loanR"] = (df["loan_ratio"] - df["loan_ratio"].mean()) / df["loan_ratio"].std()

# D1, D2, D3
df["D1"] = 0.7071 * df["Z_sav"] + 0.7071 * df["Z_loan"]
df["D2_raw"] = (
    0.463 * df["bank_per_km"] +
    0.167 * df["atm_per_km"] +
    0.074 * df["agent_per_km"] +
    0.296 * df["nonbank_per_km"]
)
df["D2"] = (df["D2_raw"] - df["D2_raw"].mean()) / df["D2_raw"].std()
df["D3"] = 0.7071 * df["Z_dep"] + 0.7071 * df["Z_loanR"]

# AFI calculation
df["AFI"] = 0.5017 * df["D1"] + 0.6274 * df["D2"] + 0.3576 * df["D3"]

# Final model features
features = [
    "Rekening Tabungan Perorangan Bank",
    "Rekening Kredit Perorangan Bank",
    "Jumlah penduduk",
    "Jumlah Kantor Bank",
    "Jumlah Kantor Pegadaian",
    "Jumlah Kantor PMV",
    "Jumlah Kantor PNM",
    "Jumlah ATM",
    "Jumlah Agen Laku Pandai",
    "Luas Terhuni",
    "Nominal Tabungan Perorangan Bank ",
    "Nominal Kredit Perorangan Bank",
    "PDRB",
    "AFI"
]

df = df.dropna(subset=["IPM"] + features)
X = df[features]
y = df["IPM"]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "rf_ipm_fullvars.pkl")
print("Model saved to rf_ipm_fullvars.pkl")
