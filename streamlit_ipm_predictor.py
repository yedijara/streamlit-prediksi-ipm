
import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("rf_ipm_fullvars.pkl")

st.title("Simulasi Prediksi IPM Berdasarkan Variabel Keuangan dan AFI")

st.sidebar.header("Input Variabel")

# Input sliders
rek_tab = st.sidebar.number_input("Rekening Tabungan Perorangan Bank", value=2000000)
rek_kredit = st.sidebar.number_input("Rekening Kredit Perorangan Bank", value=50000)
penduduk = st.sidebar.number_input("Jumlah Penduduk", value=100000)
kantor_bank = st.sidebar.number_input("Jumlah Kantor Bank", value=50)
pegadaian = st.sidebar.number_input("Jumlah Kantor Pegadaian", value=20)
pmv = st.sidebar.number_input("Jumlah Kantor PMV", value=10)
pnm = st.sidebar.number_input("Jumlah Kantor PNM", value=5)
atm = st.sidebar.number_input("Jumlah ATM", value=100)
agen = st.sidebar.number_input("Jumlah Agen Laku Pandai", value=300)
luas = st.sidebar.number_input("Luas Wilayah Terhuni (km²)", value=100.0)
nom_tab = st.sidebar.number_input("Nominal Tabungan (Rp)", value=3e13, format="%.0f")
nom_kredit = st.sidebar.number_input("Nominal Kredit (Rp)", value=1e13, format="%.0f")
pdrb = st.sidebar.number_input("PDRB (Rp)", value=1e14, format="%.0f")

# Compute derived ratios
sav_per_pop = rek_tab / penduduk
loan_per_pop = rek_kredit / penduduk
bank_per_km = kantor_bank / luas
nonbank = pegadaian + pmv + pnm
nonbank_per_km = nonbank / luas
atm_per_km = atm / luas
agen_per_km = agen / luas
dep_ratio = nom_tab / pdrb
loan_ratio = nom_kredit / pdrb

# Z-scores (replace with constants from your data)
def z(val, mean, std):
    return (val - mean) / std

# Means and stds from your dataset
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

# Compute D1, D2, D3
D1 = 0.7071 * z(sav_per_pop, means["sav_per_pop"], stds["sav_per_pop"]) + 0.7071 * z(loan_per_pop, means["loan_per_pop"], stds["loan_per_pop"])
D2_raw = 0.463 * bank_per_km + 0.167 * atm_per_km + 0.074 * agen_per_km + 0.296 * nonbank_per_km
D2 = z(D2_raw, means["D2_raw"], stds["D2_raw"])
D3 = 0.7071 * z(dep_ratio, means["dep_ratio"], stds["dep_ratio"]) + 0.7071 * z(loan_ratio, means["loan_ratio"], stds["loan_ratio"])

# Final AFI
AFI = 0.5017 * D1 + 0.6274 * D2 + 0.3576 * D3

# Predict IPM
input_features = np.array([[rek_tab, rek_kredit, penduduk, kantor_bank, pegadaian, pmv, pnm,
                            atm, agen, luas, nom_tab, nom_kredit, pdrb, AFI]])
predicted_ipm = model.predict(input_features)[0]

st.subheader("Hasil Perhitungan")
st.write(f"D1 = {D1:.4f}")
st.write(f"D2 = {D2:.4f}")
st.write(f"D3 = {D3:.4f}")
st.write(f"AFI = {AFI:.4f}")
st.success(f"📈 Prediksi IPM: {predicted_ipm:.2f}")
