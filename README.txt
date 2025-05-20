# 1. Buka terminal (CMD, PowerShell, atau Terminal biasa)
# 2. Masuk ke folder project
cd project-folder

# 3. Buat dan aktifkan virtual environment (opsional tapi disarankan)
python -m venv venv
venv\\Scripts\\activate   # di Windows
source venv/bin/activate  # di Mac/Linux

# 4. Install semua pustaka
pip install -r requirements.txt

# 5. Jalankan aplikasi Streamlit
streamlit run app.py