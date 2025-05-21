
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and dataset
model = joblib.load("rf_ipm_fullvars.pkl")
df = pd.read_excel("train_inklusi_rf v4.xlsx", sheet_name="Sheet1")
df = df[df["Tahun"] == 2024].copy()
df["wilayah"] = df["Kabupaten / Kota"]

# UI - Select region
selected_region = st.selectbox("Pilih Kabupaten/Kota (2024):", df["wilayah"].unique())
row = df[df["wilayah"] == selected_region].iloc[0]

st.sidebar.header("Edit Variabel (otomatis dari data terpilih)")

# Original values
originals = {
    "Rekening Tabungan": row["Rekening Tabungan Perorangan Bank"],
    "Rekening Kredit": row["Rekening Kredit Perorangan Bank"],
    "Penduduk": row["Jumlah penduduk"],
    "Kantor Bank": row["Jumlah Kantor Bank"],
    "Pegadaian": row["Jumlah Kantor Pegadaian"],
    "PMV": row["Jumlah Kantor PMV"],
    "PNM": row["Jumlah Kantor PNM"],
    "ATM": row["Jumlah ATM"],
    "Agen": row["Jumlah Agen Laku Pandai"],
    "Luas": row["Luas Terhuni"],
    "Tabungan Nominal": row["Nominal Tabungan Perorangan Bank "],
    "Kredit Nominal": row["Nominal Kredit Perorangan Bank"],
    "PDRB": row["PDRB"]
}

# Sliders
inputs = {
    k: st.sidebar.number_input(k, value=int(v) if isinstance(v, (int, float)) else v, format="%.0f") for k, v in originals.items()
}

# Derived
sav_per_pop = inputs["Rekening Tabungan"] / inputs["Penduduk"]
loan_per_pop = inputs["Rekening Kredit"] / inputs["Penduduk"]
bank_per_km = inputs["Kantor Bank"] / inputs["Luas"]
nonbank_per_km = (inputs["Pegadaian"] + inputs["PMV"] + inputs["PNM"]) / inputs["Luas"]
atm_per_km = inputs["ATM"] / inputs["Luas"]
agen_per_km = inputs["Agen"] / inputs["Luas"]
dep_ratio = inputs["Tabungan Nominal"] / inputs["PDRB"]
loan_ratio = inputs["Kredit Nominal"] / inputs["PDRB"]

# Use only 2024 data for z-score reference
df["sav_per_pop"] = df["Rekening Tabungan Perorangan Bank"] / df["Jumlah penduduk"]
df["loan_per_pop"] = df["Rekening Kredit Perorangan Bank"] / df["Jumlah penduduk"]
df["dep_ratio"] = df["Nominal Tabungan Perorangan Bank "] / df["PDRB"]
df["loan_ratio"] = df["Nominal Kredit Perorangan Bank"] / df["PDRB"]
df["bank_per_km"] = df["Jumlah Kantor Bank"] / df["Luas Terhuni"]
df["nonbank_per_km"] = (df["Jumlah Kantor Pegadaian"] + df["Jumlah Kantor PMV"] + df["Jumlah Kantor PNM"]) / df["Luas Terhuni"]
df["atm_per_km"] = df["Jumlah ATM"] / df["Luas Terhuni"]
df["agen_per_km"] = df["Jumlah Agen Laku Pandai"] / df["Luas Terhuni"]

df["D2_raw"] = (
    0.463 * df["bank_per_km"] +
    0.167 * df["atm_per_km"] +
    0.074 * df["agen_per_km"] +
    0.296 * df["nonbank_per_km"]
)

def z(val, series): return (val - series.mean()) / series.std()

# D1
Z_sav = z(sav_per_pop, df["sav_per_pop"])
Z_loan = z(loan_per_pop, df["loan_per_pop"])
D1 = 0.7071 * Z_sav + 0.7071 * Z_loan

# D2
D2_raw = 0.463 * bank_per_km + 0.167 * atm_per_km + 0.074 * agen_per_km + 0.296 * nonbank_per_km
D2 = z(D2_raw, df["D2_raw"])

# D3
Z_dep = z(dep_ratio, df["dep_ratio"])
Z_loanR = z(loan_ratio, df["loan_ratio"])
D3 = 0.7071 * Z_dep + 0.7071 * Z_loanR

# AFI
AFI = 0.5017 * D1 + 0.6274 * D2 + 0.3576 * D3

# Predict IPM
model_input = np.array([[inputs["Rekening Tabungan"], inputs["Rekening Kredit"], inputs["Penduduk"],
                         inputs["Kantor Bank"], inputs["Pegadaian"], inputs["PMV"], inputs["PNM"],
                         inputs["ATM"], inputs["Agen"], inputs["Luas"],
                         inputs["Tabungan Nominal"], inputs["Kredit Nominal"], inputs["PDRB"], AFI]])
predicted_ipm = model.predict(model_input)[0]

# Display
st.subheader(f"📊 Simulasi untuk: {selected_region}")
st.write(f"D1 = {D1:.4f} | D2 = {D2:.4f} | D3 = {D3:.4f}")
st.write(f"AFI = {AFI:.4f}")
st.write(" ")

# Original IPM
if "IPM" in row:
    st.info(f"🎯 IPM Asli (2024): {row['IPM']:.2f}")

# Predicted IPM
st.success(f"📈 Prediksi IPM (Simulasi): {predicted_ipm:.2f}")

# List of changed inputs
changed = {k: (int(v), int(inputs[k])) for k, v in originals.items() if int(v) != int(inputs[k])}
if changed:
    st.warning("🛠 Variabel yang diubah:")
    for k, (old, new) in changed.items():
        st.write(f"- {k}: {old} → {new}")
