# 1. Gunakan image dasar Python 3.13 versi ringan
FROM python:3.13-slim
# 2. Tentukan folder kerja di dalam container
WORKDIR /app
# 3. Salin hanya empat file aplikasi ke dalam image
COPY app_demo_d.py preprocessing_d_demo.joblib model_d_demo.ubj requirements.txt ./
# 4. Pasang library sesuai requirements.txt tanpa menyimpan cache pip
RUN pip install --no-cache-dir -r requirements.txt
# 5. Jalankan aplikasi Streamlit saat container dinyalakan
CMD ["streamlit", "run", "app_demo_d.py", "--server.port=8501", "--server.address=0.0.0.0"]