import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Muat model prediksi IPM (misalnya, model random forest) 
model = joblib.load("rf_ipm_fullvars.pkl")

# Muat dataset penuh sebagai baseline dan nilai default
df = pd.read_excel("train_inklusi_rf v4.xlsx", sheet_name="Sheet1")
df["wilayah"] = df["Kabupaten / Kota"] + " (" + df["Tahun"].astype(str) + ")"
df_full = df.copy()

# Pilih wilayah melalui dropdown
selected_region = st.selectbox("Pilih Kabupaten/Kota (Semua Tahun):", df_full["wilayah"])
row = df_full[df_full["wilayah"] == selected_region].iloc[0]

# Fungsi z-score
def z(val, series):
    return (val - series.mean()) / series.std()

# ============================
# 1. Hitung baseline (dari nilai asli row)
# ============================
base_rek_tab    = int(row["Rekening Tabungan Perorangan Bank"])
base_rek_kredit = int(row["Rekening Kredit Perorangan Bank"])
base_penduduk   = int(row["Jumlah penduduk"])
base_kantor_bank    = int(row["Jumlah Kantor Bank"])
base_pegadaian      = int(row["Jumlah Kantor Pegadaian"])
base_pmv            = int(row["Jumlah Kantor PMV"])
base_pnm            = int(row["Jumlah Kantor PNM"])
base_atm            = int(row["Jumlah ATM"])
base_agen           = int(row["Jumlah Agen Laku Pandai"])
base_luas           = float(row["Luas Terhuni"])
base_nom_tab        = float(row["Nominal Tabungan Perorangan Bank "])
base_nom_kredit     = float(row["Nominal Kredit Perorangan Bank"])
base_pdrb           = float(row["PDRB"])

# Derived baseline values
base_sav_per_pop   = base_rek_tab / base_penduduk
base_loan_per_pop  = base_rek_kredit / base_penduduk
base_bank_per_km   = base_kantor_bank / base_luas
base_nonbank_per_km = (base_pegadaian + base_pmv + base_pnm) / base_luas
base_atm_per_km    = base_atm / base_luas
base_agen_per_km   = base_agen / base_luas
base_dep_ratio     = base_nom_tab / base_pdrb
base_loan_ratio    = base_nom_kredit / base_pdrb

base_Z_sav   = z(base_sav_per_pop, df_full["sav_per_pop"])
base_Z_loan  = z(base_loan_per_pop, df_full["loan_per_pop"])
base_D1      = 0.7071 * base_Z_sav + 0.7071 * base_Z_loan

base_D2_raw  = 0.463 * base_bank_per_km + 0.167 * base_atm_per_km + 0.074 * base_agen_per_km + 0.296 * base_nonbank_per_km
base_D2      = z(base_D2_raw, df_full["D2_raw"])

base_Z_dep   = z(base_dep_ratio, df_full["dep_ratio"])
base_Z_loanR = z(base_loan_ratio, df_full["loan_ratio"])
base_D3      = 0.7071 * base_Z_dep + 0.7071 * base_Z_loanR

base_AFI     = 0.5017 * base_D1 + 0.6274 * base_D2 + 0.3576 * base_D3

# Fitur untuk model (baseline)
features_base = np.array([[base_rek_tab, base_rek_kredit, base_penduduk, base_kantor_bank,
                             base_pegadaian, base_pmv, base_pnm,
                             base_atm, base_agen, base_luas, base_nom_tab,
                             base_nom_kredit, base_pdrb, base_AFI]])
base_prediction = model.predict(features_base)[0]
# Catatan: base_prediction disimpan untuk perbandingan tetapi tidak langsung ditampilkan

# ============================
# 2. Input slider sebagai variabel adjust (default = nilai baseline)
# ============================
st.sidebar.header("Edit Variabel (otomatis dari data terpilih)")
rek_tab    = st.sidebar.number_input("Rekening Tabungan", value=base_rek_tab)
rek_kredit = st.sidebar.number_input("Rekening Kredit", value=base_rek_kredit)
penduduk   = st.sidebar.number_input("Jumlah Penduduk", value=base_penduduk)
kantor_bank = st.sidebar.number_input("Kantor Bank", value=base_kantor_bank)
pegadaian   = st.sidebar.number_input("Kantor Pegadaian", value=base_pegadaian)
pmv         = st.sidebar.number_input("Kantor PMV", value=base_pmv)
pnm         = st.sidebar.number_input("Kantor PNM", value=base_pnm)
atm         = st.sidebar.number_input("Jumlah ATM", value=base_atm)
agen        = st.sidebar.number_input("Agen Laku Pandai", value=base_agen)
luas        = st.sidebar.number_input("Luas Wilayah Terhuni (km²)", value=base_luas)
nom_tab     = st.sidebar.number_input("Nominal Tabungan (Rp)", value=base_nom_tab, format="%.0f")
nom_kredit  = st.sidebar.number_input("Nominal Kredit (Rp)", value=base_nom_kredit, format="%.0f")
pdrb        = st.sidebar.number_input("PDRB (Rp)", value=base_pdrb, format="%.0f")

# Derived values dari input slider
sav_per_pop   = rek_tab / penduduk
loan_per_pop  = rek_kredit / penduduk
bank_per_km   = kantor_bank / luas
nonbank_per_km = (pegadaian + pmv + pnm) / luas
atm_per_km    = atm / luas
agen_per_km   = agen / luas
dep_ratio     = nom_tab / pdrb
loan_ratio    = nom_kredit / pdrb

Z_sav   = z(sav_per_pop, df_full["sav_per_pop"])
Z_loan  = z(loan_per_pop, df_full["loan_per_pop"])
D1      = 0.7071 * Z_sav + 0.7071 * Z_loan

D2_raw  = 0.463 * bank_per_km + 0.167 * atm_per_km + 0.074 * agen_per_km + 0.296 * nonbank_per_km
D2      = z(D2_raw, df_full["D2_raw"])

Z_dep   = z(dep_ratio, df_full["dep_ratio"])
Z_loanR = z(loan_ratio, df_full["loan_ratio"])
D3      = 0.7071 * Z_dep + 0.7071 * Z_loanR

AFI = 0.5017 * D1 + 0.6274 * D2 + 0.3576 * D3

# Fitur untuk prediksi dengan nilai variabel yang disesuaikan
features = np.array([[rek_tab, rek_kredit, penduduk, kantor_bank, pegadaian, pmv, pnm,
                      atm, agen, luas, nom_tab, nom_kredit, pdrb, AFI]])
result_prediction = model.predict(features)[0]

# Hitung deviasi prediksi terhadap baseline
deviation_prediction = result_prediction - base_prediction

# Format tampilan deviasi (+0 jika nol, +x untuk nilai positif atau -x untuk negatif)
if deviation_prediction >= 0:
    deviation_str = f"+{deviation_prediction:.2f}"
else:
    deviation_str = f"{deviation_prediction:.2f}"

# ============================
# 3. Tampilkan hasil di layar
# ============================
st.subheader("📊 Hasil Simulasi")
st.write(f"**Wilayah:** {selected_region}")
st.write(f"D1 = {D1:.4f} | D2 = {D2:.4f} | D3 = {D3:.4f}")
st.write(f"AFI = {AFI:.4f}")

# Tampilkan tulisan Baseline IPM disertai deviasi (default +0 jika tidak ada perubahan)
st.info(f"Baseline IPM: {deviation_str}")

# Opsional: tampilkan prediksi IPM baru
st.success(f"📈 Prediksi IPM baru: {result_prediction:.2f}")
