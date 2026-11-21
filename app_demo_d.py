## app_demo_d.py - Demo model risiko pembatalan booking (model XGB_D)
# Model: pipeline lengkap (preprocessing + XGBoost), resep kandidat D asli,
# dilatih ulang pada data train kohort H-1 di notebook proyek (Bagian 17) dan
# lolos dua gerbang verifikasi di sana (AP cocok angka tercatat + paritas muat ulang).
# Model disimpan sebagai DUA file: preprocessing_d_demo.joblib (joblib) dan
# model_d_demo.ubj (format UBJ, format resmi XGBoost untuk penyimpanan model).
#
# Cara jalankan:
#   1. Letakkan preprocessing_d_demo.joblib dan model_d_demo.ubj di folder yang
#      sama dengan file ini.
#   2. Dari folder itu: streamlit run app_demo_d.py
#
# Revisi 21 Nov 2026: hanya keterangan hasil yang diperjelas (sebutan skor mentah dan
# cara pakai yang aman). Model, file yang dibaca, dan perhitungan skor tidak berubah.
#
# Batas yang sama dengan notebook tetap berlaku: ini model DEMO untuk cerita proyek,
# bukan model yang terbukti layak operasional (Bagian 15 notebook). Karena tidak ada
# ambang yang pernah dibekukan untuk model D, demo ini menampilkan SKOR saja,
# tidak ada label "perlu ditindaklanjuti" seperti demo baseline dulu.

import os
import joblib
import pandas as pd
import streamlit as st
import xgboost as xgb

NAMA_FILE_PREPROCESSING = 'preprocessing_d_demo.joblib'
NAMA_FILE_MODEL = 'model_d_demo.ubj'


# 1. Muat preprocessing dan model sekali saja; berhenti dengan pesan jelas
#    bila salah satu file tidak ada.
@st.cache_resource
def muat_model():
    for nama_file in (NAMA_FILE_PREPROCESSING, NAMA_FILE_MODEL):
        if not os.path.exists(nama_file):
            return None, nama_file
    preprocessing = joblib.load(NAMA_FILE_PREPROCESSING)
    model = xgb.XGBClassifier()
    model.load_model(NAMA_FILE_MODEL)
    return (preprocessing, model), None


pasangan, file_kurang = muat_model()

st.title('Demo risiko pembatalan booking (model XGB_D)')
st.caption(
    'Model demo dari proyek analisis pembatalan booking hotel. '
    'Skor yang tampil adalah keluaran mentah model yang belum dikalibrasi, '
    'bukan persentase pasti pembatalan.'
)

if pasangan is None:
    st.error(
        f'File {file_kurang} tidak ditemukan. Letakkan preprocessing_d_demo.joblib '
        'dan model_d_demo.ubj di folder yang sama dengan app_demo_d.py, lalu muat '
        'ulang halaman ini.'
    )
    st.stop()

preprocessing, model = pasangan

# 2. Siapkan pilihan kategori PERSIS seperti nilai di data latih kohort H-1.
#    (Kalau nilainya tidak sama persis, encoder menganggapnya kategori tak dikenal.)
BULAN = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
         'August', 'September', 'October', 'November', 'December']
BULAN_ID = {'January': 'Januari', 'February': 'Februari', 'March': 'Maret',
            'April': 'April', 'May': 'Mei', 'June': 'Juni', 'July': 'Juli',
            'August': 'Agustus', 'September': 'September', 'October': 'Oktober',
            'November': 'November', 'December': 'Desember'}
HARI_ID = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis',
           4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'}
TIPE_KAMAR = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
TIPE_PELANGGAN = ['Transient', 'Transient-Party', 'Contract', 'Group']
SALURAN = ['TA/TO', 'Direct', 'Corporate', 'GDS']

# 3. Form input: 12 fitur mentah yang diminta pipeline D.
st.subheader('Data booking yang mau dinilai')
kolom1, kolom2 = st.columns(2)
with kolom1:
    lead_time = st.number_input(
        'Lead time (hari antara pembuatan booking dan kedatangan)',
        min_value=0, max_value=500, value=30, step=1, key='lead_time',
        help='Di data latih: 2 sampai 429 hari.')
    bulan = st.selectbox(
        'Bulan kedatangan', BULAN, index=2, key='bulan',
        format_func=lambda b: BULAN_ID[b])
    hari = st.selectbox(
        'Hari kedatangan', list(HARI_ID.keys()), index=3, key='hari',
        format_func=lambda h: HARI_ID[h])
    tipe_kamar = st.selectbox('Tipe kamar yang dipesan', TIPE_KAMAR, index=4, key='tipe_kamar')
    tipe_pelanggan = st.selectbox('Tipe pelanggan', TIPE_PELANGGAN, index=0, key='tipe_pelanggan')
    saluran = st.selectbox('Saluran distribusi', SALURAN, index=0, key='saluran')
with kolom2:
    tamu_lama = st.selectbox(
        'Pernah menginap sebelumnya?', [0, 1], index=0, key='tamu_lama',
        format_func=lambda v: 'Sudah pernah' if v == 1 else 'Belum pernah')
    batal_dulu = st.number_input(
        'Riwayat pembatalan tamu ini', min_value=0, max_value=30, value=0, step=1,
        key='batal_dulu', help='Di data latih: 0 sampai 13.')
    sukses_dulu = st.number_input(
        'Riwayat booking tamu ini yang tidak batal', min_value=0, max_value=60, value=0,
        step=1, key='sukses_dulu', help='Di data latih: 0 sampai 57.')
    permintaan_khusus = st.number_input(
        'Jumlah permintaan khusus', min_value=0, max_value=5, value=0, step=1,
        key='permintaan_khusus', help='Di data latih: 0 sampai 5.')
    malam_akhir_pekan = st.number_input(
        'Malam akhir pekan (Sabtu/Minggu)', min_value=0, max_value=20, value=1, step=1,
        key='malam_akhir_pekan', help='Di data latih: 0 sampai 16.')
    malam_hari_kerja = st.number_input(
        'Malam hari kerja (Senin-Jumat)', min_value=0, max_value=50, value=2, step=1,
        key='malam_hari_kerja', help='Di data latih: 0 sampai 41.')

# 4. Nilai booking: susun satu baris dengan nama kolom dan tipe persis seperti data latih,
#    olah dengan preprocessing, lalu minta model menghitung skornya.
if st.button('Nilai risiko booking ini', key='tombol_nilai'):
    baris = pd.DataFrame([{
        'lead_time': int(lead_time),
        'arrival_date_month': str(bulan),
        'arrival_day_of_week': int(hari),
        'reserved_room_type': str(tipe_kamar),
        'previous_cancellations': int(batal_dulu),
        'previous_bookings_not_canceled': int(sukses_dulu),
        'is_repeated_guest': int(tamu_lama),
        'customer_type': str(tipe_pelanggan),
        'distribution_channel': str(saluran),
        'total_of_special_requests': int(permintaan_khusus),
        'stays_in_weekend_nights': int(malam_akhir_pekan),
        'stays_in_week_nights': int(malam_hari_kerja),
    }])
    skor = float(model.predict_proba(preprocessing.transform(baris))[:, 1][0])

    st.subheader('Hasil penilaian')
    st.markdown(f'**Keluaran model: {skor * 100:.1f}%**')
    st.markdown('Skor mentah model, bukan peluang batal yang sudah terbukti.')
    st.info(
            'Angka ini belum dapat diartikan sebagai peluang booking batal. '
            'Model masih dalam tahap pengembangan. '
            'Periksa booking secara manual dan, bila perlu, '
            'hubungi tamu untuk mengonfirmasi rencana kedatangannya.'
    )

    with st.expander('Detail model'):
        st.markdown(f'- Skor batal: **{skor * 100:.1f}%**, skor tidak batal: **{(1 - skor) * 100:.1f}%**. Keduanya keluaran mentah model yang belum dikalibrasi untuk kohort H-1.')
        st.markdown('- Model: preprocessing (one-hot + passthrough) dan XGBoost, resep kandidat D, 12 fitur, dilatih pada 27.697 booking train kohort H-1. Disimpan sebagai dua file: preprocessing (joblib) dan model (format UBJ).')
        st.markdown('- Hasil evaluasi tercatat: AP validation 0,108 (pembanding tebak mayoritas 0,025); generalisasi lintas waktu lemah, AP test 0,023.')
        st.markdown('- Model ini direkonstruksi dan diverifikasi di notebook proyek (Bagian 17); artefak asli tetap tersimpan utuh di laptop sebagai arsip.')
        st.markdown('- Batas: model demo untuk cerita proyek, bukan model yang terbukti layak dipakai hotel. Belum ada bukti bahwa meninjau booking berskor tinggi lebih berguna daripada peninjauan biasa; manfaat itu harus dibuktikan dulu sebelum pola kerja hotel diubah.')