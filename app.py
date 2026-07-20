import streamlit as st
import mysql.connector
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from datetime import date
import base64
from fpdf import FPDF
from datetime import datetime
import os


# --- 1. KONEKSI DATABASE ---



def get_connection():
    return mysql.connector.connect(
        host=st.secrets["mysql"]["host"],
        user=st.secrets["mysql"]["user"],
        password=st.secrets["mysql"]["password"],
        database=st.secrets["mysql"]["database"],
        port=int(st.secrets["mysql"]["port"])
    )






# --- 2. FUNGSI KRIPTOGRAFI (ENKRIPSI & DEKRIPSI) ---



def encrypt_data(text):

    return base64.b64encode(text.encode()).decode()


def decrypt_data(text):

    try:

        return base64.b64decode(text.encode()).decode()

    except:

        return "Error Dekripsi"







# --- 3. MODEL NAIVE BAYES ---



@st.cache_resource
def train_model_pro():
    db = get_connection()
    query = "SELECT usia, riwayat_pakai, tes_urine, lama_pakai, status_kerja, dukungan_keluarga, hasil_prediksi FROM data_pasien"
    df = pd.read_sql(query, db)
    db.close()

    if df.empty:
        st.error("Data training kosong!")
        return None, 0, {}, None, None

    # Mapping aman untuk menangani variasi data lama & baru
    mapping = {
        'riwayat_pakai': {
            '< 30 hari': 0, '<30 hari': 0, 
            '> 30 hari (Seumur Hidup)': 1, '>30 hari (Seumur Hidup)': 1,
            '>30 hari': 1, '> 30 hari': 1
        },
        'tes_urine': {'Negatif': 0, 'Positif': 1},
        'status_kerja': {'Kerja': 0, 'Tidak': 1, 'Tidak Kerja': 1},
        'dukungan_keluarga': {'Baik': 0, 'Buruk': 1, 'Sedang': 0},
        'hasil_prediksi': {'Rendah': 0, 'Sedang': 1, 'Tinggi': 2}
    }

    # Proses mapping data
    for col in mapping:
        df[col] = df[col].map(mapping[col])
    
    # Kolom fitur utama yang wajib bersih dari NaN/NULL
    kolom_fitur = ['usia', 'riwayat_pakai', 'tes_urine', 'lama_pakai', 'status_kerja', 'dukungan_keluarga', 'hasil_prediksi']
    
    # Bersihkan baris yang mengandung NaN pada fitur utama
    df = df.dropna(subset=kolom_fitur)
    if df.empty:
        return None, 0, {}, None, None

    # Pastikan tipe data target berupa integer bersih (0 atau 1)
    df['hasil_prediksi'] = df['hasil_prediksi'].astype(int)

    X = df[['usia', 'riwayat_pakai', 'tes_urine', 'lama_pakai', 'status_kerja', 'dukungan_keluarga']].reset_index(drop=True)
    y = df['hasil_prediksi'].reset_index(drop=True)

    # --- IMPLEMENTASI STRATIFIED K-FOLD CROSS VALIDATION (k=5) ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    list_acc = []
    list_y_test = []
    list_y_pred = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        # Standard Scaling per fold
        fold_scaler = StandardScaler()
        X_tr_scaled = fold_scaler.fit_transform(X_tr)
        X_te_scaled = fold_scaler.transform(X_te)
        
        # SMOTE per fold
        y_tr_counts = Counter(y_tr)
        if len(set(y_tr)) >= 2 and all(count >= 2 for count in y_tr_counts.values()):
            min_count = min(y_tr_counts.values())
            k_n = max(1, min(5, min_count - 1))
            sm = SMOTE(k_neighbors=k_n, random_state=42)
            X_tr_res, y_tr_res = sm.fit_resample(X_tr_scaled, y_tr)
        else:
            X_tr_res, y_tr_res = X_tr_scaled, y_tr
            
        # Training Model per fold
        fold_model = GaussianNB()
        fold_model.fit(X_tr_res, y_tr_res)
        
        # Prediksi data uji fold
        y_pred_fold = fold_model.predict(X_te_scaled)
        
        # Simpan metrik performa (Paksa konversi array numpy/list agar seragam)
        list_acc.append(accuracy_score(y_te, y_pred_fold))
        list_y_test.extend(y_te.tolist())
        list_y_pred.extend(y_pred_fold.tolist())

    # --- FINALISASI MODEL UTAMA UNTUK DEPLOYMENT ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train)
    X_test_scaled = final_scaler.transform(X_test)
    
    y_train_counts = Counter(y_train)
    if len(set(y_train)) >= 2 and all(count >= 2 for count in y_train_counts.values()):
        min_count = min(y_train_counts.values())
        k_n = max(1, min(5, min_count - 1))
        sm = SMOTE(k_neighbors=k_n, random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
        st.sidebar.success(f"SMOTE aktif (k={k_n})")
    else:
        X_train_res, y_train_res = X_train_scaled, y_train
        
    final_model = GaussianNB()
    final_model.fit(X_train_res, y_train_res)
    
    # Rata-rata akurasi riil dari 5-Fold Cross Validation
    acc_kfold = np.mean(list_acc)
    
    # Generate report global berdasarkan akumulasi seluruh prediksi fold
    report = classification_report(list_y_test, list_y_pred, output_dict=True)
    cm = confusion_matrix(list_y_test, list_y_pred, labels=[0, 1, 2]) # <-- Mengunci ukuran agar selalu 3x3
    
    return final_model, acc_kfold, report, cm, final_scaler



# --- 4. FUNGSI CETAK PDF ---

def create_pdf(nama, nik, tempat_lahir, tgl_lahir, usia, hasil, estimasi, saran, lama_pakai, riwayat_pakai, kerja, dukungan, save_to_folder=False):
    pdf = FPDF()
    pdf.add_page()
    
    # Set Margin sedikit lebih rapat (Left, Top, Right) agar ruang halaman efisien
    pdf.set_margins(12, 10, 12)
    
    # --- 1. HEADER / KOP SURAT ---
    try:
        pdf.image("logo.png", 12, 8, 28) 
    except:
        pass

    pdf.set_font("Arial", "B", 13)
    pdf.cell(80) 
    pdf.cell(30, 6, "YAYASAN SOSIAL 'BUNDA MEIFA'", ln=True, align="C")
    
    pdf.set_font("Arial", "", 7.5)
    pdf.cell(80)
    pdf.cell(30, 4, "Jl. Mapalus, Ling. 3, Kel. Watulambot Kec. Tondano Barat, Kab. Minahasa, Prov. Sulawesi Utara.", ln=True, align="C")
    pdf.cell(80)
    pdf.cell(30, 4, "Email. Meifaawarokka14@gmail.com;  Telp. 085240473335", ln=True, align="C")
    
    pdf.line(12, 28, 198, 28) 
    pdf.ln(6)

    # --- 2. JUDUL ---
    pdf.set_font("Arial", "BU", 12)
    pdf.cell(186, 6, "LAPORAN EVALUASI RISIKO RELAPS", ln=True, align="C")
    pdf.ln(3)

    # --- 3. I. DATA IDENTITAS & MEDIS ---
    pdf.set_font("Arial", "B", 10)
    pdf.cell(186, 6, "I. DATA PASIEN & RIWAYAT KLINIS", ln=True)
    pdf.set_font("Arial", "", 9.5)
    
    # Logika Tampilan Lama Pemakaian
    if isinstance(lama_pakai, (int, float)):
        if lama_pakai < 1:
            display_lama = f"{max(1, int(round(lama_pakai * 12)))} Bulan"
        else:
            display_lama = f"{round(lama_pakai, 1)} Tahun"
    else:
        display_lama = str(lama_pakai)
    
    # Grid Data Pasien (Dipersempit sedikit jarak antag-garis)
    col_w = 38
    pdf.cell(col_w, 5.5, "Nama Lengkap", 0)
    pdf.cell(4, 5.5, ":", 0)
    pdf.multi_cell(144, 5.5, str(nama), 0, 'L')
    
    pdf.cell(col_w, 5.5, "NIK", 0); pdf.cell(4, 5.5, ":", 0); pdf.cell(50, 5.5, str(nik), 0)
    pdf.cell(col_w, 5.5, "Status Kerja", 0); pdf.cell(4, 5.5, ":", 0); pdf.cell(50, 5.5, str(kerja), 0, 1)
    
    pdf.cell(col_w, 5.5, "Tempat, Tgl Lahir", 0); pdf.cell(4, 5.5, ":", 0); pdf.cell(50, 5.5, f"{tempat_lahir}, {tgl_lahir}", 0)
    pdf.cell(col_w, 5.5, "Riwayat Pakai", 0); pdf.cell(4, 5.5, ":", 0); pdf.cell(50, 5.5, str(riwayat_pakai), 0, 1)
    
    pdf.cell(col_w, 5.5, "Usia", 0); pdf.cell(4, 5.5, ":", 0); pdf.cell(50, 5.5, f"{usia} Tahun", 0)
    pdf.cell(col_w, 5.5, "Dukungan Kel.", 0); pdf.cell(4, 5.5, ":", 0); pdf.cell(50, 5.5, str(dukungan), 0, 1)
    
    pdf.cell(col_w, 5.5, "Lama Pemakaian", 0); pdf.cell(4, 5.5, ":", 0); pdf.cell(50, 5.5, display_lama, 0, 1)
    
    pdf.ln(3)

    # --- 4. II. HASIL ANALISIS ---
    pdf.set_font("Arial", "B", 10)
    pdf.cell(186, 6, "II. INTERPRETASI HASIL (NAIVE BAYES)", ln=True)
    
    pdf.set_fill_color(245, 245, 245)
    pdf.set_font("Arial", "B", 9)
    pdf.cell(55, 7, " Tingkat Risiko Relaps", 1, 0, 'L', fill=True)
    
    # Warna Teks Kategori
    if "Tinggi" in hasil:
        pdf.set_text_color(255, 0, 0)
    elif "Sedang" in hasil:
        pdf.set_text_color(255, 140, 0)
    else:
        pdf.set_text_color(0, 128, 0)
        
    pdf.cell(131, 7, f"  {hasil.upper()}", 1, 1, 'L')
    pdf.set_text_color(0, 0, 0) 

    pdf.cell(55, 7, " Estimasi Sisa Hari Aman", 1, 0, 'L', fill=True)
    pdf.cell(131, 7, f"  {estimasi} Hari", 1, 1, 'L')
    
    pdf.ln(3)

    # --- 5. III. REKOMENDASI TINDAKAN (DISESUAIKAN RAPI) ---
    pdf.set_font("Arial", "B", 10)
    pdf.cell(186, 6, "III. REKOMENDASI TINDAKAN", ln=True)
    
    # Font rekomendasi diubah ke ukuran 8.5 dengan spasi antar-baris 4.5 agar kompak
    pdf.set_font("Arial", "", 8.5)
    pdf.multi_cell(186, 4.5, saran, border=1)
    
    # --- 6. FOOTER TANDA TANGAN (PASTI MUAT DI HALAMAN 1) ---
    pdf.ln(6)
    pdf.cell(120)
    pdf.set_font("Arial", "", 9)
    pdf.cell(66, 4, f"Dicetak pada: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align="C")
    pdf.cell(120)
    pdf.cell(66, 4, "Petugas Pelaksana,", ln=True, align="C")
    
    # Spasi untuk TTD
    pdf.ln(12) 
    
    pdf.cell(120)
    pdf.set_font("Arial", "B", 9)
    pdf.cell(66, 4, f"( {st.session_state['user_name']} )", ln=True, align="C")

    # Penyimpanan / Return Stream
    nama_file = f"Laporan_{nama.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    folder_tujuan = "hasil_laporan"
    if not os.path.exists(folder_tujuan):
        os.makedirs(folder_tujuan)

    path_lengkap = os.path.join(folder_tujuan, nama_file)

    if save_to_folder:
        pdf.output(path_lengkap)
        return path_lengkap
    else:
        return pdf.output(dest="S").encode("latin-1")

# --- FUNGSI PEMBERI SARAN (BERDASARKAN TINGKAT RISIKO & KELOMPOK USIA) ---

def get_saran(hasil, estimasi, usia=25):
    try:
        usia = int(usia)
    except:
        usia = 25

    # 1. PENENTUAN KELOMPOK USIA & TEKS SPESIFIK
    if usia < 25:
        kategori_usia = "Remaja/Dewasa Muda"
        intervensi_khusus = (
            "- PENDEKATAN KELUARGA & AKADEMIS/KARIR AWAL: Melibatkan orang tua/wali secara aktif " # <-- Ubah • jadi -
            "dalam pengawasan harian serta penyesuaian lingkungan pendidikan atau pelatihan keterampilan vokasional.\n"
            "- INTERVENSI TEMAN SEBAYA (PEER PRESSURE): Edukasi ketat mengenai ketahanan terhadap tekanan pergaulan luar." # <-- Ubah • jadi -
        )
    elif 25 <= usia <= 45:
        kategori_usia = "Dewasa Usia Produktif"
        intervensi_khusus = (
            "- MANAJEMEN STRES OKUPASIONAL & FINANSIAL: Pendampingan psikologis fokus pada pengelolaan stres kerja/ekonomi " # <-- Ubah • jadi -
            "yang sering menjadi pemantik utama relaps (trigger kronis).\n"
            "- DUKUNGAN KELUARGA INTI: Melibatkan pasangan/keluarga inti untuk menjaga komunikasi positif dan stabilitas domestik." # <-- Ubah • jadi -
        )
    else:
        kategori_usia = "Dewasa / Lansia"
        intervensi_khusus = (
            "- REHABILITASI MEDIS TERPADU: Pemantauan kesehatan fisik menyeluruh mengingat adanya komplikasi medis bawaan/faktor usia.\n" # <-- Ubah • jadi -
            "- DUKUNGAN SOSIAL AFEKTIF: Mengoptimalkan dukungan dari lingkungan keluarga besar atau komunitas sosial untuk mencegah isolasi diri." # <-- Ubah • jadi -
        )

    # 2. PENENTUAN NARASI BERDASARKAN RISIKO (TINGGI, SEDANG, RENDAH)
    if hasil == "Tinggi":
        return (
            f"Hasil analisis Naive Bayes mengindikasikan tingkat risiko relaps TINGGI (Kerawanan kritis dalam {estimasi} hari ke depan).\n"
            f"Kategori Pasien: {kategori_usia} ({usia} Tahun)\n\n"
            f"REKOMENDASI INTERVENSI MEDIS & PSIKOSOSIAL TERTARGET:\n"
            f"1. PROGRAM RAWAT INAP/INTENSIF: Pembatasan aktivitas luar secara mutlak dan penempatan pasien dalam lingkungan pemulihan yang terkontrol.\n"
            f"2. INTERVENSI PSIKOLOGIS INTENSIF: Jadwalkan Konseling Intensif (Cognitive Behavioral Therapy / CBT) minimal 3 kali seminggu fokus pada 'Craving Management'.\n"
            f"3. MONITORING KLINIS KETAT: Asesmen urinalisis berkala tanpa pemberitahuan sebelumnya (unannounced drug screen).\n"
            f"4. PENDEKATAN KHUSUS USIA ({kategori_usia.upper()}):\n{intervensi_khusus}"
        )
        
    elif hasil == "Sedang":
        return (
            f"Hasil analisis Naive Bayes mengindikasikan tingkat risiko relaps SEDANG (Estimasi stabilitas klinis {estimasi} hari ke depan).\n"
            f"Kategori Pasien: {kategori_usia} ({usia} Tahun)\n\n"
            f"REKOMENDASI INTERVENSI MEDIS & PSIKOSOSIAL TERTARGET:\n"
            f"1. PROGRAM RAWAT JALAN BERKALA: Pasien wajib mengikuti sesi konseling kelompok/individu (Motivational Interviewing / MI) 1-2 kali seminggu.\n"
            f"2. EVALUASI LINGKUNGAN (HIGH-RISK SITUATIONS): Identifikasi pemicu stres harian di lingkungan kerja, keluarga, maupun sosial pasien.\n"
            f"3. PENDEKATAN KHUSUS USIA ({kategori_usia.upper()}):\n{intervensi_khusus}"
        )
        
    else:  # Rendah
        return (
            f"Hasil analisis Naive Bayes mengindikasikan tingkat risiko relaps RENDAH (Kondisi cenderung stabil, estimasi aman ~{estimasi} hari).\n"
            f"Kategori Pasien: {kategori_usia} ({usia} Tahun)\n\n"
            f"REKOMENDASI PROGRAM PEMELIHARAAN (AFTERCARE):\n"
            f"1. PROGRAM PASCA-REHABILITASI: Keikutsertaan berkala dalam Kelompok Dukungan Sebaya (Peer Support Group) minimal 1 kali sebulan.\n"
            f"2. OPTIMALISASI FAKTOR PROTEKTIF: Mempertahankan komunikasi positif, pola hidup sehat, dan aktivitas terstruktur.\n"
            f"3. PENDEKATAN KHUSUS USIA ({kategori_usia.upper()}):\n{intervensi_khusus}"
        )


# --- HALAMAN INPUT DATA RIWAYAT (MANUAL & MASSAL) ---

def import_data_page():
    st.title("📥 Kelola Basis Pengetahuan (Knowledge Base)")
    tab1, tab2 = st.tabs(["📝 Input Manual", "📊 Upload CSV Massal"])

    with tab1:
        st.subheader("Tambah Data Riwayat Baru")
        with st.form("manual_history_pro"):
            c1, c2 = st.columns(2)
            with c1:
                h_nama = st.text_input("Nama Pasien")
                h_nik = st.text_input("NIK")
                h_tmp_lahir = st.text_input("Tempat Lahir")
                h_tgl_lahir = st.date_input(
                    "Tanggal Lahir",
                    value=date(2000,1,1),
                    min_value=date(1950,1,1),
                    max_value=date.today()
                )
                
                # Hitung usia otomatis
                today = date.today()
                h_usia = today.year - h_tgl_lahir.year - ((today.month, today.day) < (h_tgl_lahir.month, h_tgl_lahir.day))
                
                col_l1, col_l2 = st.columns([2, 1])
                with col_l1:
                    h_lama_val = st.number_input("Lama Pemakaian", 0, 100, 2)
                with col_l2:
                    h_lama_unit = st.selectbox("Satuan", ["Tahun", "Bulan"])
                h_lama = h_lama_val if h_lama_unit == "Tahun" else h_lama_val / 12
                
            with c2:
                h_riw = st.selectbox("Riwayat Pemakaian", ["< 30 hari", "> 30 hari (Seumur Hidup)"])
                h_urn = st.selectbox("Hasil Tes Urine", ["Negatif", "Positif"])
                h_kerja = st.selectbox("Status Kerja", ["Kerja", "Tidak", "Tidak Kerja"])
                h_duk = st.selectbox("Dukungan Keluarga", ["Baik", "Buruk", "Sedang"])
            
            h_narkoba = st.multiselect("Jenis Narkoba yang Digunakan", ["Sabu", "Ganja", "Ekstasi", "Tramadol", "Lainnya"])

            # --- UPDATE SELEKBOX MANUAL KE 3 KATEGORI ---
            h_hasil = st.selectbox("Kesimpulan Risiko (Label Asli)", ["Rendah", "Sedang", "Tinggi"])
            h_latar = st.text_area("Catatan Latar Belakang / Kronologi")
            
            if st.form_submit_button("💾 Simpan ke Database"):
                if h_nama and h_nik and h_latar:
                    try:
                        narkoba_str = ", ".join(h_narkoba) if h_narkoba else "Tidak Ada"
                        db = get_connection(); cur = db.cursor()
                        
                        sql = """INSERT INTO data_pasien (nama_terenkripsi, nik_terenkripsi, latar_belakang, tempat_lahir, tanggal_lahir, 
                                 usia, riwayat_pakai, tes_urine, lama_pakai, status_kerja, dukungan_keluarga, jenis_narkoba, hasil_prediksi, tanggal_input) 
                                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                        
                        val = (encrypt_data(h_nama), encrypt_data(h_nik), h_latar, h_tmp_lahir, str(h_tgl_lahir), 
                               int(h_usia), h_riw, h_urn, float(h_lama), h_kerja, h_duk, narkoba_str, h_hasil, datetime.now())
                        
                        cur.execute(sql, val)
                        db.commit(); db.close()
                        st.success("✅ Data berhasil disimpan!")
                    except Exception as e:
                        st.error(f"Gagal simpan: {e}")
                else:
                    st.warning("Mohon isi semua field.")

    with tab2:
        st.subheader("Impor Data Massal via CSV")
        st.info("💡 Pastikan kolom target 'hasil_prediksi' pada file CSV Anda sudah berisi kategori: Rendah, Sedang, atau Tinggi.")
        uploaded_file = st.file_uploader("Pilih file CSV", type="csv")

        if uploaded_file:
            data_import = pd.read_csv(uploaded_file, sep=';')
            data_import.columns = data_import.columns.str.replace(u'\ufeff', '').str.strip()
            data_import = data_import.where(pd.notnull(data_import), None)
        
            st.write("Data berhasil dibaca. Silakan klik tombol di bawah untuk proses:")
            st.dataframe(data_import.head())
        
            if st.button("🚀 Proses Impor Semua Data"):
                db = get_connection(); cur = db.cursor()
                count = 0
                for _, row in data_import.iterrows():
                    try:
                        # 1. HITUNG USIA OTOMATIS
                        tgl_lahir_obj = pd.to_datetime(row['tanggal_lahir'], dayfirst=True).date()
                        today = date.today()
                        h_usia = today.year - tgl_lahir_obj.year - ((today.month, today.day) < (tgl_lahir_obj.month, tgl_lahir_obj.day))
                        
                        # --- 2. STANDARISASI FORMAT INPUT CSV ---
                        # Memastikan string teks yang masuk dari CSV seragam dengan sistem baru kita
                        res_prediksi = str(row['hasil_prediksi']).strip().title() # Mengubah jadi Kapital Awal (Tinggi/Sedang/Rendah)
                        if res_prediksi not in ["Rendah", "Sedang", "Tinggi"]:
                            res_prediksi = "Rendah" # Default jika tidak valid

                        status_kerja_csv = str(row['status_kerja']).strip()
                        if status_kerja_csv in ["Tidak", "Tidak Kerja"]:
                            status_kerja_csv = "Tidak Kerja"
                        else:
                            status_kerja_csv = "Kerja"

                        duk_kel_csv = str(row['dukungan_keluarga']).strip().title()
                        if_duk_valid = ["Baik", "Buruk", "Sedang"]

                        if duk_kel_csv in if_duk_valid:
                            if_duk_csv = duk_kel_csv
                        else:
                            if_duk_csv = "Baik"

                        # 3. QUERY SQL (14 Kolom)
                        sql = """INSERT INTO data_pasien (nama_terenkripsi, nik_terenkripsi, latar_belakang, tempat_lahir, tanggal_lahir, 
                                 usia, riwayat_pakai, tes_urine, lama_pakai, status_kerja, dukungan_keluarga, jenis_narkoba, hasil_prediksi, tanggal_input) 
                                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                        
                        val = (
                            encrypt_data(str(row['nama'])), 
                            encrypt_data(str(row['nik'])), 
                            row['latar_belakang'], 
                            row['tempat_lahir'], 
                            str(tgl_lahir_obj), 
                            int(h_usia), 
                            row['riwayat_pakai'], 
                            row['tes_urine'], 
                            float(row['lama_pakai']), 
                            status_kerja_csv, 
                            if_duk_csv, 
                            row['jenis_narkoba'],    
                            res_prediksi, 
                            datetime.now()
                        )
                        
                        cur.execute(sql, val)
                        count += 1
                    except Exception as e:
                        st.warning(f"Gagal baris {row.get('nama', 'Unknown')}: {e}")
                
                db.commit(); db.close()
                st.cache_resource.clear() # Paksa clear cache agar model langsung ditraining ulang dengan data baru
                st.success(f"✅ Berhasil mengimpor {count} data!")
            else:
            # Opsional: Pesan jika belum upload
                st.info("Silakan unggah file CSV untuk memulai.")




def get_initials(name):
    if not name or not isinstance(name, str):
        return ""
    # Memecah nama berdasarkan spasi dan mengambil huruf pertama tiap kata (dalam huruf kapital)
    parts = name.strip().split()
    initials = [part[0].upper() for part in parts if part]
    return " ".join(initials)  # Menggabungkan dengan spasi ("N M")

# --- HALAMAN RIWAYAT DATA PASIEN (MONITORING) ---

def riwayat_page():
    st.title("📋 Riwayat Monitoring Pasien")
    st.write("Data di bawah ini telah didekripsi otomatis dari database aman.") 
    
    db = get_connection()
    query = "SELECT * FROM data_pasien" 
    df = pd.read_sql(query, db)
    db.close()

    if not df.empty:
        # 1. Dekripsi Identitas
        df['Nama Pasien'] = df['nama_terenkripsi'].apply(decrypt_data)
        df["Nama Inisial"] = df["Nama Pasien"].apply(get_initials)
        df['NIK Pasien'] = df['nik_terenkripsi'].apply(decrypt_data)
        
        # 2. Urutkan Sesuai Abjad (A - Z)
        df = df.sort_values(by='Nama Pasien', ascending=True).reset_index(drop=True)
        
        # 3. Format Tanggal Input
        df['Waktu Input'] = pd.to_datetime(df['tanggal_input']).dt.strftime('%d-%m-%Y %H:%M')
        
        # 4. Nomor Urut
        df.insert(0, 'No.', range(1, 1 + len(df)))
        
        # --- TABEL UTAMA ---
        kolom_tampil = ['No.', 'Nama Inisial', 'NIK Pasien', 'tempat_lahir', 'riwayat_pakai', 'tanggal_lahir', 'hasil_prediksi', 'Waktu Input']
        st.dataframe(df[kolom_tampil], use_container_width=True, hide_index=True)
        
        st.divider()

        # --- SELEKSI DATA UNTUK MELIHAT DETAIL LENGKAP ---
        st.subheader("🔍 Detail Lengkap Data Pasien")
        
        if 'show_detail' not in st.session_state:
            st.session_state.show_detail = False

        # Input Cari untuk Detail Pasien
        kw_detail = st.text_input(
            "🔎 Cari pasien (Nama / Inisial / NIK):", 
            placeholder="Ketik inisial nama atau NIK di sini...",
            key="kw_detail_input"
        )

        # Filter Data Detail
        if kw_detail.strip():
            df_detail = df[
                df['Nama Pasien'].str.contains(kw_detail, case=False, na=False) |
                df['Nama Inisial'].str.contains(kw_detail, case=False, na=False) |
                df['NIK Pasien'].str.contains(kw_detail, case=False, na=False)
            ]
        else:
            df_detail = df

        opsi_detail = ["-- Pilih Pasien --"] + [
            f"No. {row['No.']} - {row['Nama Inisial']} ({row['NIK Pasien']})" 
            for _, row in df_detail.iterrows()
        ]

        def on_patient_change():
            if st.session_state.pilihan_pasien != "-- Pilih Pasien --":
                st.session_state.show_detail = True
            else:
                st.session_state.show_detail = False

        pilihan_label = st.selectbox(
            "Pilih pasien dari hasil pencarian:", 
            options=opsi_detail,
            key="pilihan_pasien",
            on_change=on_patient_change
        )
        
        # Tampilkan Panel Detail
        if pilihan_label != "-- Pilih Pasien --" and st.session_state.show_detail:
            no_urut_terpilih = int(pilihan_label.split(" - ")[0].replace("No. ", ""))
            p = df[df['No.'] == no_urut_terpilih].iloc[0]

            with st.container(border=True):
                col_head1, col_head2, col_close = st.columns([3, 1.5, 1])
                with col_head1:
                    st.markdown(f"### 👤 Nama Pasien: `{p['Nama Pasien']} ({p['Nama Inisial']})`")
                    st.caption(f"NIK: **{p['NIK Pasien']}** | Waktu Input: **{p['Waktu Input']}**")
                with col_head2:
                    if p['hasil_prediksi'] == "Tinggi":
                        st.error(f"### RISIKO {p['hasil_prediksi'].upper()}")
                    elif p['hasil_prediksi'] == "Sedang":
                        st.warning(f"### RISIKO {p['hasil_prediksi'].upper()}")
                    else:
                        st.success(f"### RISIKO {p['hasil_prediksi'].upper()}")
                with col_close:
                    if st.button("❌ Sembunyikan", use_container_width=True):
                        st.session_state.show_detail = False
                        st.rerun()

                st.divider()
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**📌 Data Pribadi & Klinis:**")
                    st.write(f"* **Tempat, Tgl Lahir:** {p['tempat_lahir']}, {p['tanggal_lahir']}")
                    st.write(f"* **Usia:** {p['usia']} Tahun")
                    st.write(f"* **Status Pekerjaan:** {p['status_kerja']}")
                with c2:
                    st.markdown("**🧪 Riwayat Pemakaian & Medis:**")
                    st.write(f"* **Riwayat Pakai:** {p['riwayat_pakai']}")
                    st.write(f"* **Hasil Tes Urine:** {p['tes_urine']}")
                    st.write(f"* **Lama Pemakaian:** {p['lama_pakai']} Tahun")
                with c3:
                    st.markdown("**🤝 Lingkungan & Jenis Substansi:**")
                    st.write(f"* **Dukungan Keluarga:** {p['dukungan_keluarga']}")
                    st.write(f"* **Jenis Narkoba:** {p.get('jenis_narkoba', 'Tidak Ada')}")

                st.write("")
                st.markdown("**📝 Catatan Latar Belakang / Kronologi Pasien:**")
                st.info(p.get('latar_belakang', 'Tidak ada catatan latar belakang.'))

                est_temp = 14 if p['hasil_prediksi'] == "Tinggi" else (30 if p['hasil_prediksi'] == "Sedang" else 60)
                saran_teks = get_saran(p['hasil_prediksi'], est_temp, p['usia'])
                with st.expander("💡 Lihat Rekomendasi Intervensi Medis"):
                    st.write(saran_teks)
        else:
            st.info("👆 Silakan ketik nama/inisial/NIK atau pilih nama pasien pada menu dropdown di atas.")

    
        # --- PERBAIKAN FITUR CARI AGAR TERDEKRIPSI ---
        st.divider()
        st.subheader("🔍 Cari Data Pasien")
        cari_nik = st.text_input("Masukkan Nama atau NIK untuk mencari")
        
        if cari_nik:
            # Melakukan filter pada df (yang sudah didekripsi)
            hasil_cari = df[
                df['Nama Pasien'].str.contains(cari_nik, case=False, na=False) | 
                df['NIK Pasien'].str.contains(cari_nik, case=False, na=False)
            ].copy()
            
            if not hasil_cari.empty:
                st.write(f"Hasil pencarian untuk: '{cari_nik}'")
                
                # Buat nomor urut baru agar mulai dari 1
                hasil_cari['No.'] = range(1, 1 + len(hasil_cari))
                
                # Tampilkan kolom yang bersih saja
                kolom_bersih = ['No.', 'Nama Pasien', 'NIK Pasien', 'usia', 'hasil_prediksi', 'Waktu Input']
                st.table(hasil_cari[kolom_bersih])



       



        # --- PERBANDINGAN DATA PASIEN (BERDASARKAN NIK) ---
        st.divider()
        st.subheader("📈 Perbandingan Tren Prediksi Pasien")
        st.info("Pilih pasien berdasarkan NIK untuk melihat perkembangan risiko relaps dari waktu ke waktu.")

        # Input Cari untuk Tren Prediksi
        kw_tren = st.text_input("🔎 Cari pasien untuk melihat grafik tren:", placeholder="Ketik nama atau NIK...", key="kw_tren_input")
        
        if kw_tren.strip():
            df_tren_filtered = df[
                df['Nama Pasien'].str.contains(kw_tren, case=False, na=False) |
                df['Nama Inisial'].str.contains(kw_tren, case=False, na=False) |
                df['NIK Pasien'].str.contains(kw_tren, case=False, na=False)
            ]
        else:
            df_tren_filtered = df

        opsi_pasien_tren = ["-- Pilih Pasien --"] + list(
            df_tren_filtered.apply(lambda x: f"{x['Nama Inisial']} - {x['NIK Pasien']}", axis=1).unique()
        )
        
        pilihan_tren = st.selectbox("Pilih Pasien (Inisial - NIK):", options=opsi_pasien_tren, key="select_tren")

        if pilihan_tren != "-- Pilih Pasien --":
            nik_terpilih = pilihan_tren.split(" - ")[1]
            df_graph = df[df['NIK Pasien'] == nik_terpilih].sort_values('tanggal_input')

            if len(df_graph) > 0:
                st.write(f"### Tren Risiko untuk: **{pilihan_tren}**")
                df_graph['risiko_val'] = df_graph['hasil_prediksi'].map({'Tinggi': 1, 'Sedang': 0.5, 'Rendah': 0})
                
                chart_data = df_graph.set_index('Waktu Input')[['risiko_val']]
                st.line_chart(chart_data)
                st.caption("Keterangan Grafik: 1.0 = Tinggi, 0.5 = Sedang, 0.0 = Rendah")

                st.write("**Riwayat Pemeriksaan:**")
                st.table(df_graph[['Waktu Input', 'usia', 'tes_urine', 'hasil_prediksi']])







        # --- CRUD: EDIT & HAPUS ---

        st.divider()

        st.subheader("⚙️ Manajemen Data")

        col_edit, col_hapus = st.columns(2)
        with col_edit:
            with st.expander("✏️ Edit Data & Hitung Ulang"):
                # KUNCI: Gunakan 'df' yang masih lengkap kolomnya
                edit_no = st.selectbox("Pilih No. Urut Pasien:", options=df['No.'].tolist(), key="edit_box")
                data_lama = df[df['No.'] == edit_no].iloc[0]
                
                with st.form("form_edit_lengkap"):
                    st.write(f"Editing: **{data_lama['Nama Pasien']}**")
                    # Menggunakan .get() agar lebih aman dari KeyError
                    new_latar = st.text_area("Latar Belakang Baru", value=data_lama.get('latar_belakang', ""))
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        new_urine = st.selectbox("Update Urine", ["Negatif", "Positif"], 
                                                 index=0 if data_lama['tes_urine'] == "Negatif" else 1)
                        new_kerja = st.selectbox("Update Kerja", ["Kerja", "Tidak"],
                                                 index=0 if data_lama['status_kerja'] == "Kerja" else 1)
                    with c2:
                        new_duk = st.selectbox("Dukungan Keluarga", ["Baik", "Buruk"],
                                               index=0 if data_lama['dukungan_keluarga'] == "Baik" else 1)
                        new_lama = st.number_input("Lama Pakai", 0.0, 40.0, float(data_lama['lama_pakai']))

                    if st.form_submit_button("💾 Update Data"):
                        # 1. Panggil Model & Scaler
                        model, acc, report, cm, scaler = train_model_pro()
                        
                        # 2. DEFINISIKAN SEMUA VARIABEL FITUR (Pastikan ada 6 variabel)
                        # Konversi teks ke angka (0/1) sesuai mapping yang digunakan saat training
                        riw_v = 1 if data_lama['riwayat_pakai'] == "> 30 hari (Seumur Hidup)" or data_lama['riwayat_pakai'] == ">30 hari (Seumur Hidup)" else 0
                        urn_v = 1 if new_urine == "Positif" else 0
                        ker_v = 1 if new_kerja == "Tidak" else 0
                        duk_v = 1 if new_duk == "Buruk" else 0
                        
                        # 3. SUSUN LIST FITUR (Pastikan urutannya sama dengan saat training)
                        # Urutan: usia, riwayat_pakai, tes_urine, lama_pakai, status_kerja, dukungan_keluarga
                        fitur = [[
                            float(data_lama['usia']), 
                            int(riw_v), 
                            int(urn_v), 
                            float(new_lama), 
                            int(ker_v), 
                            int(duk_v)
                        ]]
                        
                        # 4. PREDIKSI ULANG
                        fitur_scaled = scaler.transform(fitur)
                        pred_label = model.predict(fitur_scaled)[0]
                        
                        mapping_kategori = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}
                        kategori_hasil = mapping_kategori[pred_label]
                        
                        # 5. UPDATE DATABASE
                        db = get_connection(); cur = db.cursor()
                        sql = """UPDATE data_pasien SET latar_belakang=%s, tes_urine=%s, lama_pakai=%s, 
                                 status_kerja=%s, dukungan_keluarga=%s, hasil_prediksi=%s WHERE id=%s"""
                        cur.execute(sql, (new_latar, new_urine, new_lama, new_kerja, new_duk, kategori_hasil, int(data_lama['id'])))
                        db.commit(); db.close()
                        
                        st.cache_resource.clear() 
                        st.success("✅ Data diperbarui!")
                        st.rerun()






        with col_hapus:
            with st.expander("🗑️ Hapus Data Pasien"):
                # 1. Input teks untuk mencari pasien
                kw_hapus = st.text_input(
                    "🔎 Cari pasien yang ingin dihapus:", 
                    placeholder="Ketik Nama, Inisial, atau NIK...", 
                    key="kw_hapus_input"
                )
                
                # Filter data berdasarkan kata kunci pencarian
                if kw_hapus.strip():
                    df_hapus_filtered = df[
                        df['Nama Pasien'].str.contains(kw_hapus, case=False, na=False) |
                        df['Nama Inisial'].str.contains(kw_hapus, case=False, na=False) |
                        df['NIK Pasien'].str.contains(kw_hapus, case=False, na=False)
                    ]
                else:
                    df_hapus_filtered = df

                # Buat daftar unik pasien yang terfilter
                unique_patients = df_hapus_filtered[['NIK Pasien', 'Nama Pasien', 'Nama Inisial']].drop_duplicates()
                patient_options = ["-- Pilih Pasien --"] + list(
                    unique_patients.apply(lambda x: f"{x['NIK Pasien']} - {x['Nama Pasien']}", axis=1)
                )
                
                selected_patient = st.selectbox("Pilih Pasien dari Hasil Pencarian:", options=patient_options, key="select_hapus_patient")
                
                if selected_patient and selected_patient != "-- Pilih Pasien --":
                    # Ambil NIK dari string yang dipilih
                    nik_selected = selected_patient.split(" - ")[0]
                    
                    # 2. TAHAP KEDUA: Pilih riwayat pemeriksaan yang ingin dihapus
                    records = df[df['NIK Pasien'] == nik_selected].sort_values('tanggal_input', ascending=False)
                    
                    options_to_delete = st.multiselect(
                        "Pilih riwayat pemeriksaan untuk dihapus:",
                        options=records.index,
                        format_func=lambda x: f"{records.loc[x, 'hasil_prediksi']} - {records.loc[x, 'Waktu Input']}"
                    )
                    
                    if st.button("Hapus Data Terpilih", type="primary"):
                        if options_to_delete:
                            ids_to_delete = [int(df.loc[x, 'id']) for x in options_to_delete]
                            
                            try:
                                db = get_connection()
                                cur = db.cursor()
                                format_strings = ','.join(['%s'] * len(ids_to_delete))
                                cur.execute(f"DELETE FROM data_pasien WHERE id IN ({format_strings})", tuple(ids_to_delete))
                                db.commit()
                                st.cache_resource.clear()
                                db.close()
                                st.success(f"Berhasil menghapus {len(ids_to_delete)} data!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Gagal menghapus data: {e}")
                        else:
                            st.warning("Pilih minimal satu riwayat pemeriksaan untuk dihapus.")


                   



        # --- FITUR REKAPITULASI BERDASARKAN NIK UNIK (3 KELAS) ---
        st.write("")
        st.divider()
        st.markdown("<h3 style='color: #4FF3A7;'>📊 Rekapitulasi Status Pasien</h3>", unsafe_allow_html=True)
        st.info("💡 Data di bawah dihitung berdasarkan status risiko pemeriksaan medis paling terbaru dari masing-masing pasien unik.")

        # 1. Hitung Total Pasien Unik berdasarkan NIK
        total_pasien_unik = df['NIK Pasien'].nunique()

        # 2. Ambil pemeriksaan terbaru dari tiap NIK
        df_terbaru = df.sort_values('tanggal_input').groupby('NIK Pasien').tail(1)
        rekap_status = df_terbaru['hasil_prediksi'].value_counts()

        # 3. Tampilkan dalam 4 Kolom Metric yang Proporsional & Kreatif
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Pasien (Unik)", total_pasien_unik)
        m2.metric("Risiko Rendah", rekap_status.get("Rendah", 0))
        m3.metric("Risiko Sedang", rekap_status.get("Sedang", 0))
        m4.metric("Risiko Tinggi", rekap_status.get("Tinggi", 0), delta_color="inverse")

        # 4. Tampilkan Grafik Distribusi Batang 3 Kategori Risiko Nyata
        st.write("")
        st.markdown("##### **Distribusi Tingkat Kerawanan Pasien Saat Ini:**")
        
        # Sempurnakan urutan index chart agar berurutan secara logis di layar
        order_index = ["Rendah", "Sedang", "Tinggi"]
        chart_data = pd.Series(0, index=order_index)
        for cat in order_index:
            chart_data[cat] = rekap_status.get(cat, 0)
            
        st.bar_chart(chart_data)



       



        st.divider()
        st.subheader("🖨️ Cetak Laporan PDF")

        # Input Cari untuk Cetak Laporan
        kw_cetak = st.text_input("🔎 Cari pasien yang akan dicetak laporan PDF-nya:", placeholder="Ketik nama atau NIK...", key="kw_cetak_input")

        if kw_cetak.strip():
            df_cetak_filtered = df[
                df['Nama Pasien'].str.contains(kw_cetak, case=False, na=False) |
                df['Nama Inisial'].str.contains(kw_cetak, case=False, na=False) |
                df['NIK Pasien'].str.contains(kw_cetak, case=False, na=False)
            ]
        else:
            df_cetak_filtered = df

        if not df_cetak_filtered.empty:
            opsi_cetak = {
                f"No: {row['No.']} - {row['Nama Inisial']} ({row['NIK Pasien']})": idx 
                for idx, row in df_cetak_filtered.iterrows()
            }
            
            pilihan_pdf_label = st.selectbox(
                "Pilih Pasien untuk Cetak PDF:", 
                options=["-- Pilih Pasien --"] + list(opsi_cetak.keys()),
                key="select_pdf"
            )

            # VARIABEL 'p' DIDEFINISIKAN HANYA JIKA PASIEN DIPILIH
            if pilihan_pdf_label != "-- Pilih Pasien --":
                idx_terpilih_pdf = opsi_cetak[pilihan_pdf_label]
                p = df.loc[idx_terpilih_pdf]  # <-- Variabel 'p' didefinisikan di sini!
                
                est_temp = 14 if p['hasil_prediksi'] == "Tinggi" else (30 if p['hasil_prediksi'] == "Sedang" else 60)
                saran_pdf = get_saran(p['hasil_prediksi'], est_temp, p['usia'])
                
                # Buat stream PDF di memori
                data_pdf = create_pdf(
                    p['Nama Pasien'], p['NIK Pasien'], p['tempat_lahir'], p['tanggal_lahir'], 
                    p['usia'], p['hasil_prediksi'], est_temp, saran_pdf, 
                    p['lama_pakai'], p['riwayat_pakai'], p['status_kerja'], p['dukungan_keluarga'], 
                    save_to_folder=False
                )
                
                # Tombol Download PDF
                st.download_button(
                    label=f"📥 Download PDF ({p['Nama Inisial']})",
                    data=data_pdf,
                    file_name=f"Laporan_{p['Nama Inisial'].replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )

                # Logika Arsip Otomatis (HARUS berada di DALAM blok `if pilihan_pdf_label != "-- Pilih Pasien --"`)
                key_arsip = f"saved_{p['Nama Pasien']}_{p['id']}"
                if key_arsip not in st.session_state:
                    create_pdf(
                        p['Nama Pasien'], p['NIK Pasien'], p['tempat_lahir'], p['tanggal_lahir'], 
                        p['usia'], p['hasil_prediksi'], est_temp, saran_pdf, 
                        p['lama_pakai'], p['riwayat_pakai'], p['status_kerja'], p['dukungan_keluarga'], 
                        save_to_folder=True
                    )
                    st.session_state[key_arsip] = True
                    st.toast(f"Laporan {p['Nama Inisial']} berhasil diarsipkan.")

        else:
            st.warning("Tidak ada data pasien yang cocok untuk dicetak.")







# --- 6. HALAMAN MANAJEMEN USER (DENGAN EDIT & HAPUS AKUN) ---

def admin_page():
    st.title("👥 Manajemen Akun Petugas")

    # Ambil data seluruh akun dari database
    try:
        db = get_connection()
        df_users = pd.read_sql("SELECT id, nama, nik, password, role FROM users", db)
        db.close()
    except Exception as e:
        st.error(f"Gagal mengambil data akun: {e}")
        return

    # Membuat Tab Navigasi agar rapi
    tab_tambah, tab_kelola = st.tabs(["➕ Tambah Akun Baru", "⚙️ Kelola Akun (Edit & Hapus)"])

    # ================= TAB 1: TAMBAH AKUN BARU =================
    with tab_tambah:
        st.subheader("Form Pendaftaran Akun")
        
        with st.form("tambah_user_form", clear_on_submit=False):
            n_nama = st.text_input(
                "Nama Lengkap", 
                placeholder="Contoh: Budi Santoso",
                help="Masukkan nama lengkap petugas (huruf saja)"
            )
            n_nik = st.text_input(
                "NIK", 
                placeholder="Contoh: 7102012304950001",
                max_chars=16,
                help="NIK wajib berupa 16 digit angka"
            )
            n_pass = st.text_input(
                "Pass", 
                type="password",
                help="Password minimal 6 karakter"
            )
            n_role = st.selectbox("Role", ["Petugas", "Admin"])
            
            btn_daftar = st.form_submit_button("Daftar")
            
            if btn_daftar:
                # 1. Validasi Kolom Kosong
                if not n_nama.strip() or not n_nik.strip() or not n_pass.strip():
                    st.error("❌ Semua kolom wajib diisi!")
                
                # 2. Validasi Nama (Tidak Boleh Hanya Angka)
                elif n_nama.strip().isdigit():
                    st.error("❌ Nama Lengkap harus mengandung huruf, tidak boleh angka saja!")
                
                # 3. Validasi NIK Harus Angka
                elif not n_nik.isdigit():
                    st.error("❌ NIK harus berupa digit angka! Tidak boleh mengandung huruf atau simbol.")
                
                # 4. Validasi Panjang NIK (Harus 16 Digit)
                elif len(n_nik) != 16:
                    st.error(f"❌ NIK harus persis 16 digit! (Saat ini: {len(n_nik)} digit)")
                
                # 5. Validasi Panjang Password Minimal 6 Karakter
                elif len(n_pass) < 6:
                    st.error("❌ Password terlalu pendek! Minimal 6 karakter demi keamanan.")
                
                else:
                    # 6. Cek Pengecekan NIK Duplikat di Database
                    try:
                        db = get_connection()
                        cur = db.cursor(dictionary=True)
                        cur.execute("SELECT id FROM users WHERE nik = %s", (n_nik,))
                        existing_user = cur.fetchone()
                        
                        if existing_user:
                            st.error(f"❌ NIK {n_nik} sudah terdaftar! Gunakan NIK lain.")
                            db.close()
                        else:
                            # Jika Semua Validasi Lolos -> Simpan ke Database
                            cur.execute(
                                "INSERT INTO users (nama, nik, password, role) VALUES (%s, %s, %s, %s)", 
                                (n_nama.strip(), n_nik.strip(), n_pass, n_role)
                            )
                            db.commit()
                            db.close()
                            
                            # Tampilkan pesan berhasil
                            st.success(f"✅ Berhasil menambah akun baru atas nama **{n_nama.strip()}**!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Gagal menyimpan data ke database: {e}")

    # ================= TAB 2: KELOLA AKUN (EDIT & HAPUS) =================
    with tab_kelola:
        st.subheader("📋 Daftar Seluruh Akun Petugas")
        
        if not df_users.empty:
            # Tampilkan Tabel Akun (tanpa menampilkan password asli agar aman)
            df_tampil = df_users.copy()
            df_tampil['password'] = "••••••••"  # Masking password untuk tampilan tabel
            st.dataframe(df_tampil[['id', 'nama', 'nik', 'role', 'password']], use_container_width=True, hide_index=True)
            
            st.divider()

            col_edit, col_hapus = st.columns(2)

            # --- SUB-FITUR 1: EDIT AKUN ---
            with col_edit:
                with st.expander("✏️ Edit Akun Petugas"):
                    # Pilihan akun berdasarkan Nama & NIK
                    user_map = {f"{u['nama']} (NIK: {u['nik']})": u for _, u in df_users.iterrows()}
                    selected_label = st.selectbox("Pilih Akun yang Ingin Diedit:", list(user_map.keys()), key="edit_usr_select")
                    
                    if selected_label:
                        u_data = user_map[selected_label]
                        
                        with st.form("form_edit_user"):
                            st.write(f"Editing Akun ID: **{u_data['id']}**")
                            e_nama = st.text_input("Nama Lengkap", value=u_data['nama'])
                            e_nik = st.text_input("NIK", value=u_data['nik'])
                            e_pass = st.text_input("Password Baru", value=u_data['password'], type="password")
                            
                            role_idx = 0 if u_data['role'] == "Petugas" else 1
                            e_role = st.selectbox("Role", ["Petugas", "Admin"], index=role_idx)
                            
                            if st.form_submit_button("💾 Simpan Perubahan"):
                                try:
                                    db = get_connection()
                                    cur = db.cursor()
                                    cur.execute("UPDATE users SET nama=%s, nik=%s, password=%s, role=%s WHERE id=%s",
                                                (e_nama, e_nik, e_pass, e_role, int(u_data['id'])))
                                    db.commit()
                                    db.close()
                                    st.success("✅ Data akun berhasil diperbarui!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Gagal memperbarui akun: {e}")

            # --- SUB-FITUR 2: HAPUS AKUN ---
            with col_hapus:
                with st.expander("🗑️ Hapus Akun Petugas"):
                    user_del_map = {f"{u['nama']} (NIK: {u['nik']})": u for _, u in df_users.iterrows()}
                    selected_del_label = st.selectbox("Pilih Akun yang Akan Dihapus:", list(user_del_map.keys()), key="del_usr_select")
                    
                    if selected_del_label:
                        target_user = user_del_map[selected_del_label]
                        st.warning(f"⚠️ Hapus akun **{target_user['nama']}**?")
                        
                        if st.button("🗑️ Konfirmasi Hapus Akun", type="primary"):
                            # Pencegahan: Hindari menghapus akun diri sendiri yang sedang login
                            if str(target_user['nama']).lower() == str(st.session_state.get('user_name', '')).lower():
                                st.error("❌ Anda tidak bisa menghapus akun yang sedang Anda gunakan!")
                            else:
                                try:
                                    db = get_connection()
                                    cur = db.cursor()
                                    cur.execute("DELETE FROM users WHERE id=%s", (int(target_user['id']),))
                                    db.commit()
                                    db.close()
                                    st.success("✅ Akun berhasil dihapus!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Gagal menghapus akun: {e}")
        else:
            st.info("Belum ada data akun petugas di database.")



# --- 4. VISUALISASI EVALUASI (TAMBAHAN UNTUK SKRIPSI) ---
def evaluasi_page():
    st.markdown("<h2 style='text-align: center; color: #4FF3A7;'>📊 Evaluasi Model & Performa</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #A0AEC0;'>Analisis performa algoritma Naive Bayes menggunakan Stratified 5-Fold Cross Validation.</p>", unsafe_allow_html=True)
    st.write("")
    
    # Memanggil model dan metrik dari fungsi K-Fold terbaru
    model, acc, report, cm, scaler = train_model_pro()

    if model is None:
        st.error("Gagal memuat evaluasi model.")
        st.stop()

    st.metric("Akurasi Keseluruhan (Mean K-Fold)", f"{acc*100:.2f}%")
    st.write("")

    st.markdown("#### 🔲 Confusion Matrix (Multi-class)")
    
    # Kategori label harus pas berurutan dengan labels=[0, 1, 2]
    kategori_cm = ["Rendah", "Sedang", "Tinggi"]
    
    # Mengonversi matriks cm (yang sekarang dijamin selalu 3x3) ke DataFrame
    df_cm = pd.DataFrame(
        cm, 
        index=[f"{k} (Aktual)" for k in kategori_cm], 
        columns=[f"{k} (Prediksi)" for k in kategori_cm]
    )
    st.table(df_cm)
    st.write("")

    # --- 3. TABEL METRIK PRECISION, RECALL & F1-SCORE ---
    st.markdown("#### 📈 Precision, Recall & F1-Score Detail")
    
    df_report = pd.DataFrame(report).transpose()
    
    # Mengubah nama indeks angka (0, 1, 2) menjadi nama kategori asli agar akademis & mudah dibaca
    mapping_nama_baris = {
        '0': 'Risiko Rendah',
        '1': 'Risiko Sedang',
        '2': 'Risiko Tinggi',
        'accuracy': 'Accuracy',
        'macro avg': 'Macro Average',
        'weighted avg': 'Weighted Average'
    }
    df_report.index = df_report.index.map(lambda x: mapping_nama_baris.get(x, x))

    format_kolom = {
        "precision": "{:.4f}",
        "recall": "{:.4f}",
        "f1-score": "{:.4f}",
        "support": "{:.0f}"  # <-- Memaksa kolom support menjadi bilangan bulat tanpa desimal
    }
    
    # Format angka desimal agar rapi (4 angka di belakang koma standar jurnal ilmiah)
    st.dataframe(df_report.style.format(format_kolom), width="stretch")



# --- FUNGSI LOGOUT (MEMBERSIHKAN SESSION STATE DAN KEMBALI KE LOGIN) ---
def logout_user():
    # Menghapus semua data sesi agar aman dan kembali ke tampilan login
    st.session_state["loggedin"] = False
    st.session_state["user_name"] = None
    st.session_state["user_role"] = None
    st.session_state["res"] = None
    st.rerun()

# --- 7. HALAMAN UTAMA (DENGAN LOGO KIRI ATAS) ---

def main_app():
    try:
        st.sidebar.image("logo.png", use_container_width='stretch')
    except:
        st.sidebar.warning("File logo.png tidak ditemukan.")

    menu = ["Input Prediksi", "Riwayat & Perbandingan", "Import Data Lama", "Evaluasi Model"]
    if st.session_state.get('user_role') == 'Admin': 
        menu.append("Manajemen Akun")
    
    choice = st.sidebar.selectbox("Menu", menu)
    st.sidebar.info(f"Login sebagai: {st.session_state['user_name']}")
    
    if st.sidebar.button("Log Out"):
        st.session_state['loggedin'] = False
        st.rerun()

    if choice == "Riwayat & Perbandingan":
        riwayat_page()
    elif choice == "Import Data Lama":
        import_data_page()
    elif choice == "Manajemen Akun":
        admin_page()
    elif choice == "Evaluasi Model":
        evaluasi_page()
    else:
        st.title("🏥 Analisis Risiko Relaps")
        
        # Load model untuk sidebar akurasi
        model, acc, report, cm, scaler = train_model_pro()
        st.sidebar.metric("Akurasi Algoritma NB", f"{acc*100:.2f}%")

        if 'form_id' not in st.session_state:
            st.session_state.form_id = 0

        with st.form(key=f"f_pro_{st.session_state.form_id}"):
            col1, col2 = st.columns(2)
            with col1:
                nama = st.text_input("Nama Pasien")
                nik = st.text_input("NIK")
                
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    tmp_lahir = st.text_input("Tempat Lahir")
                with col_t2:
                    tgl_lahir = st.date_input(
                        "Tanggal Lahir",
                        value=date(2000,1,1),
                        min_value=date(1950,1,1),
                        max_value=date.today()
                    )
                today = date.today()
                usia = today.year - tgl_lahir.year - ((today.month, today.day) < (tgl_lahir.month, tgl_lahir.day))

                c_lama1, c_lama2 = st.columns([2, 1])
                with c_lama1:
                    lama_val = st.number_input("Lama Pemakaian", 0, 100, 2)
                with c_lama2:
                    lama_unit = st.selectbox("Satuan", ["Tahun", "Bulan"])
                
                lama_pakai = lama_val if lama_unit == "Tahun" else lama_val / 12
                lama_display = f"{lama_val} {lama_unit}"
                
            with col2:
                pilihan_riwayat = st.selectbox("Riwayat Pemakaian", ["< 30 hari", "> 30 hari (Seumur Hidup)"])
                urn = st.selectbox("Hasil Tes Urine", ["Negatif", "Positif"])
                kerja = st.selectbox("Status Pekerjaan", ["Kerja", "Tidak"])
                dukungan = st.selectbox("Dukungan Keluarga", ["Baik", "Buruk"])
            
            jenis_narkoba = st.multiselect("Jenis Narkoba yang Digunakan", ["Sabu", "Ganja", "Ekstasi", "Tramadol", "Lainnya"])
            ltr = st.text_area("Latar Belakang / Catatan Kasus")

            #Tombol submit form
            proses_tombol = st.form_submit_button("Proses Diagnosa")

            if proses_tombol:
                if nama and nik and ltr:

                    # 1. CEK APAKAH NIK SUDAH ADA DI DATABASE
                    db = get_connection()
                    cur = db.cursor(dictionary=True)

                    nik_enc = encrypt_data(nik)
                    cur.execute("SELECT * FROM data_pasien WHERE nik_terenkripsi = %s", (nik_enc,))
                    pasien_lama = cur.fetchone()

                    # PERBAIKAN: Berikan nilai awal kosong
                    nama_lama = "" 

                    if pasien_lama:
                        nama_lama = decrypt_data(pasien_lama['nama_terenkripsi'])
                        
                        # Pindahkan pengecekan ke DALAM blok if pasien_lama
                        if nama.lower() != nama_lama.lower():
                            st.error(f"❌ Error: NIK {nik} sudah terdaftar atas nama **{nama_lama}**. Satu NIK hanya untuk satu identitas.")
                            db.close()
                            st.stop()
                
                    # 2. JIKA LOLOS VALIDASI, LANJUTKAN PROSES PREDIKSI & SIMPAN
                    # (Sisa kode prediksi kamu di sini...)
                    riw_v = 0 if pilihan_riwayat == "< 30 hari" else 1


                    # Preprocessing
                    riw_v = 0 if pilihan_riwayat == "< 30 hari" else 1
                    urn_v = 1 if urn == "Positif" else 0
                    ker_v = 1 if kerja == "Tidak" else 0
                    duk_v = 1 if dukungan == "Buruk" else 0
                    
                    # Prediksi
                    fitur = [[usia, riw_v, urn_v, lama_pakai, ker_v, duk_v]]

                    # Scaling dulu
                    
                    if scaler is None:
                        st.error("Model belum siap. Silakan isi data training.")
                        st.stop()
                    fitur_scaled = scaler.transform(fitur)
                    
                    # 1. Prediksi kategori langsung menggunakan logika Naive Bayes (MAP)
                    pred_label = model.predict(fitur_scaled)[0]
                    mapping_kategori = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}
                    kategori = mapping_kategori[pred_label]

                    # 2. Ambil probabilitas terbanyak/tertinggi untuk menghitung estimasi hari aman
                    prob_semua = model.predict_proba(fitur_scaled)[0]
                    prob_tertinggi = prob_semua[pred_label]

                    # Hitung estimasi hari berdasarkan kelas yang terpilih
                    if kategori == "Tinggi":
                        est = int(20 * (1 - prob_tertinggi)) + 5     # Rentang 5 - 12 Hari
                    elif kategori == "Sedang":
                        est = int(40 * (1 - prob_tertinggi)) + 15    # Rentang 15 - 40 Hari
                    else:
                        est = int(50 * (1 - prob_tertinggi)) + 40    # Rentang 40 - 90 Hari

                    saran_teks = get_saran(kategori, est)
                    
                    # Simpan ke DB
                    db = get_connection(); cur = db.cursor()
                    sql = """INSERT INTO data_pasien (nama_terenkripsi, nik_terenkripsi, latar_belakang, tempat_lahir, tanggal_lahir, 
                             usia, riwayat_pakai, tes_urine, lama_pakai, status_kerja, 
                             dukungan_keluarga, hasil_prediksi, tanggal_input) 
                             VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
                    val = (encrypt_data(nama), encrypt_data(nik), ltr, tmp_lahir, str(tgl_lahir), 
                           usia, pilihan_riwayat, urn, lama_pakai, kerja, dukungan, kategori, datetime.now())
                    cur.execute(sql, val)
                    db.commit(); db.close()

                    # --- SIMPAN SEMUA KE SESSION STATE ---
                    st.session_state.res = {"n": nama, "nk": nik, "u": usia, "h": kategori, "e": est, "s": get_saran(kategori, est)}
                    st.rerun()
                else:
                    st.error("Lengkapi data!")

                if not nik.isdigit():
                    st.error("NIK harus berupa angka!")
                    st.stop()

                if len(nik) < 8:
                    st.error("NIK terlalu pendek!")
                    st.stop()

        # --- TAMPILAN HASIL (MENGAMBIL DARI SESSION STATE) ---
        if st.session_state.res:
            r = st.session_state.res
            st.divider()
            st.subheader("📑 Detail Hasil Analisis")
            
            if r['h'] == "Tinggi":
                st.error(f"### 🚨 KESIMPULAN: RISIKO {r['h'].upper()}")
            elif r['h'] == "Sedang":
                st.warning(f"### ⚠️ KESIMPULAN: RISIKO {r['h'].upper()}")
            else:
                st.success(f"### ✅ KESIMPULAN: RISIKO {r['h'].upper()}")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Nama Pasien:** {r['n']}")
                st.write(f"**NIK:** {r['nk']}")
                st.write(f"**Usia:** {r['u']} Tahun")
            with col_b:
                st.write(f"**Estimasi Relaps:** {r['e']} Hari")
                st.progress(min(r['e'], 100) / 100) 
            
            st.info(f"**Saran Penanganan:**\n{r['s']}")
            
            # Buat PDF menggunakan data dari session state
            # Ganti baris pemanggilan PDF kamu menjadi:
            pdf_data = create_pdf(
                r['n'],             # nama
                r['nk'],            # nik
                tmp_lahir,          # tempat_lahir (Ambil dari input form)
                str(tgl_lahir),     # tgl_lahir (Ambil dari input form)
                r['u'],             # usia
                r['h'],             # hasil
                r['e'],             # estimasi
                r['s'],             # saran
                lama_pakai,         # lama_pakai (Ambil dari input form)
                pilihan_riwayat,    # riwayat_pakai (Ambil dari input form)
                kerja,              # kerja (Ambil dari input form)
                dukungan            # dukungan (Ambil dari input form)
            )
            st.download_button(label=f"📥 Unduh Laporan PDF", data=pdf_data, file_name=f"Laporan_{r['n']}.pdf")

            st.divider()
            if st.button("🔄 Input Data Pasien Baru"):
                st.session_state.res = None          # Hapus riwayat tampilan hasil prediksi
                st.session_state.form_id += 1       # Ubah ID form untuk memaksa form di atas kosong total!
                st.rerun()




# --- 8. HALAMAN LOGIN (DENGAN LOGO TENGAH ATAS) ---

def login_page():

    # Membuat 3 kolom untuk memposisikan logo di tengah

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        try:

            st.image("logo.png", use_container_width='stretch')

        except:

            st.write("### [ Logo Yayasan ]")

   

    st.markdown("<h2 style='text-align: center;'>🔐 Login Sistem</h2>", unsafe_allow_html=True)

   

    with st.form("L"):

        u = st.text_input("NIK")

        p = st.text_input("Password", type="password")

        if st.form_submit_button("Masuk"):

            if not u or not p:

                st.error("NIK dan Password wajib diisi!")

            else:

                db = get_connection(); cur = db.cursor(dictionary=True)

                cur.execute("SELECT * FROM users WHERE nik=%s AND password=%s", (u, p))

                user = cur.fetchone(); db.close()

                if user:

                    st.session_state.update({'loggedin': True, 'user_name': user['nama'], 'user_role': user['role']})

                    st.rerun()

                else:

                    st.error("NIK atau Password salah!")



# --- LOGIKA NAVIGASI ---

if 'loggedin' not in st.session_state:
    st.session_state['loggedin'] = False

# KUNCI PERBAIKAN: Tambahkan inisialisasi res di sini agar tidak error saat user masuk
if 'res' not in st.session_state:
    st.session_state['res'] = None


# Jalankan halaman berdasarkan status login
if not st.session_state['loggedin']:
    login_page()
else:
    main_app()
