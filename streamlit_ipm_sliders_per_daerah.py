
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("rf_ipm_fullvars.pkl")

# Load full dataset for default values
df = pd.read_excel("train_inklusi_rf v4.xlsx", sheet_name="Sheet1")
df["wilayah"] = df["Kabupaten / Kota"] + " (" + df["Tahun"].astype(str) + ")"
df_full = df.copy()

# Dropdown
selected_region = st.selectbox("Pilih Kabupaten/Kota (Tahun 2024):", df[df["Tahun"] == 2024]["wilayah"].unique())
row = df_full[df_full["wilayah"] == selected_region].iloc[0]

st.sidebar.header("Edit Variabel (otomatis dari data terpilih)")

rek_tab = st.sidebar.number_input("Rekening Tabungan", value=int(row["Rekening Tabungan Perorangan Bank"]))
rek_kredit = st.sidebar.number_input("Rekening Kredit", value=int(row["Rekening Kredit Perorangan Bank"]))
penduduk = st.sidebar.number_input("Jumlah Penduduk", value=int(row["Jumlah penduduk"]))
kantor_bank = st.sidebar.number_input("Kantor Bank", value=int(row["Jumlah Kantor Bank"]))
pegadaian = st.sidebar.number_input("Kantor Pegadaian", value=int(row["Jumlah Kantor Pegadaian"]))
pmv = st.sidebar.number_input("Kantor PMV", value=int(row["Jumlah Kantor PMV"]))
pnm = st.sidebar.number_input("Kantor PNM", value=int(row["Jumlah Kantor PNM"]))
atm = st.sidebar.number_input("Jumlah ATM", value=int(row["Jumlah ATM"]))
agen = st.sidebar.number_input("Agen Laku Pandai", value=int(row["Jumlah Agen Laku Pandai"]))
luas = st.sidebar.number_input("Luas Wilayah Terhuni (km²)", value=float(row["Luas Terhuni"]))
nom_tab = st.sidebar.number_input("Nominal Tabungan (Rp)", value=float(row["Nominal Tabungan Perorangan Bank "]), format="%.0f")
nom_kredit = st.sidebar.number_input("Nominal Kredit (Rp)", value=float(row["Nominal Kredit Perorangan Bank"]), format="%.0f")
pdrb = st.sidebar.number_input("PDRB (Rp)", value=float(row["PDRB"]), format="%.0f")

# Derived values
sav_per_pop = rek_tab / penduduk
loan_per_pop = rek_kredit / penduduk
bank_per_km = kantor_bank / luas
nonbank_per_km = (pegadaian + pmv + pnm) / luas
atm_per_km = atm / luas
agen_per_km = agen / luas
dep_ratio = nom_tab / pdrb
loan_ratio = nom_kredit / pdrb

# Load full data for z-score computation
df_full["sav_per_pop"] = df_full["Rekening Tabungan Perorangan Bank"] / df_full["Jumlah penduduk"]
df_full["loan_per_pop"] = df_full["Rekening Kredit Perorangan Bank"] / df_full["Jumlah penduduk"]
df_full["dep_ratio"] = df_full["Nominal Tabungan Perorangan Bank "] / df_full["PDRB"]
df_full["loan_ratio"] = df_full["Nominal Kredit Perorangan Bank"] / df_full["PDRB"]
df_full["bank_per_km"] = df_full["Jumlah Kantor Bank"] / df_full["Luas Terhuni"]
df_full["nonbank_per_km"] = (
    df_full["Jumlah Kantor Pegadaian"] + df_full["Jumlah Kantor PMV"] + df_full["Jumlah Kantor PNM"]
) / df_full["Luas Terhuni"]
df_full["atm_per_km"] = df_full["Jumlah ATM"] / df_full["Luas Terhuni"]
df_full["agen_per_km"] = df_full["Jumlah Agen Laku Pandai"] / df_full["Luas Terhuni"]

df_full["D2_raw"] = (
    0.463 * df_full["bank_per_km"] +
    0.167 * df_full["atm_per_km"] +
    0.074 * df_full["agen_per_km"] +
    0.296 * df_full["nonbank_per_km"]
)

# Z-score function
def z(val, series):
    return (val - series.mean()) / series.std()

# D1
Z_sav = z(sav_per_pop, df_full["sav_per_pop"])
Z_loan = z(loan_per_pop, df_full["loan_per_pop"])
D1 = 0.7071 * Z_sav + 0.7071 * Z_loan

# D2
D2_raw = 0.463 * bank_per_km + 0.167 * atm_per_km + 0.074 * agen_per_km + 0.296 * nonbank_per_km
D2 = z(D2_raw, df_full["D2_raw"])

# D3
Z_dep = z(dep_ratio, df_full["dep_ratio"])
Z_loanR = z(loan_ratio, df_full["loan_ratio"])
D3 = 0.7071 * Z_dep + 0.7071 * Z_loanR

# Final AFI
AFI = 0.5017 * D1 + 0.6274 * D2 + 0.3576 * D3


# Predict IPM
# BASELINE prediction
original_input = np.array([[originals["Rekening Tabungan"], originals["Rekening Kredit"], originals["Penduduk"],
                            originals["Kantor Bank"], originals["Pegadaian"], originals["PMV"], originals["PNM"],
                            originals["ATM"], originals["Agen"], originals["Luas"],
                            originals["Tabungan Nominal"], originals["Kredit Nominal"], originals["PDRB"], AFI]])
base_prediction = model.predict(original_input)[0]

# CURRENT prediction

features = np.array([[rek_tab, rek_kredit, penduduk, kantor_bank, pegadaian, pmv, pnm,
                      atm, agen, luas, nom_tab, nom_kredit, pdrb, AFI]])
predicted_ipm = model.predict(features)[0]

# Output
st.subheader("📊 Hasil Simulasi")
st.write(f"**Wilayah:** {selected_region}")
st.write(f"D1 = {D1:.4f} | D2 = {D2:.4f} | D3 = {D3:.4f}")
st.write(f"AFI = {AFI:.4f}")
st.success(f"📈 Prediksi IPM: {predicted_ipm:.2f}")
