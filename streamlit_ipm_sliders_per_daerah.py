
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("rf_ipm_fullvars.pkl")

# Load data
df = pd.read_excel("train_inklusi_rf v4.xlsx", sheet_name="Sheet1")
df = df[df["Tahun"] == 2024].copy()

# Dropdown pilih kabupaten/kota
selected_region = st.selectbox("Pilih Kabupaten/Kota (Tahun 2024):", df["Kabupaten / Kota"].unique())
row = df[df["Kabupaten / Kota"] == selected_region].iloc[0]

st.sidebar.header("Edit Variabel (otomatis dari data)")

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

# Z-score helper
def z(val, mean, std):
    return (val - mean) / std

means = {
    "sav_per_pop": 1903.8661,
    "loan_per_pop": 248.6242,
    "dep_ratio": 0.1653,
    "loan_ratio": 0.3306,
    "D2_raw": 3.9019
}
stds = {
    "sav_per_pop": 3311.8499,
    "loan_per_pop": 310.5978,
    "dep_ratio": 0.1984,
    "loan_ratio": 0.1876,
    "D2_raw": 4.9254
}

# Derived metrics
sav_per_pop = rek_tab / penduduk
loan_per_pop = rek_kredit / penduduk
bank_per_km = kantor_bank / luas
nonbank_per_km = (pegadaian + pmv + pnm) / luas
atm_per_km = atm / luas
agen_per_km = agen / luas
dep_ratio = nom_tab / pdrb
loan_ratio = nom_kredit / pdrb

# D1, D2, D3
D1 = 0.7071 * z(sav_per_pop, means["sav_per_pop"], stds["sav_per_pop"]) + 0.7071 * z(loan_per_pop, means["loan_per_pop"], stds["loan_per_pop"])
D2_raw = 0.463 * bank_per_km + 0.167 * atm_per_km + 0.074 * agen_per_km + 0.296 * nonbank_per_km
D2 = z(D2_raw, means["D2_raw"], stds["D2_raw"])
D3 = 0.7071 * z(dep_ratio, means["dep_ratio"], stds["dep_ratio"]) + 0.7071 * z(loan_ratio, means["loan_ratio"], stds["loan_ratio"])

# AFI final
AFI = 0.5017 * D1 + 0.6274 * D2 + 0.3576 * D3

# Prediksi IPM
input_features = np.array([[rek_tab, rek_kredit, penduduk, kantor_bank, pegadaian, pmv, pnm,
                            atm, agen, luas, nom_tab, nom_kredit, pdrb, AFI]])
predicted_ipm = model.predict(input_features)[0]

# Output
st.subheader("📊 Hasil Simulasi")
st.write(f"**Kab/Kota:** {selected_region}")
st.write(f"D1 = {D1:.4f} | D2 = {D2:.4f} | D3 = {D3:.4f}")
st.write(f"AFI = {AFI:.4f}")
st.success(f"📈 Prediksi IPM: {predicted_ipm:.2f}")
