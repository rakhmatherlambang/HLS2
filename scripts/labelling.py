import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# Load model ringan
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

# Load data
df = pd.read_csv("preprocessed_sinta_no_nltk.csv")

# Daftar kategori SDGs
sdg_labels = {
    1: "Tanpa Kemiskinan",
    2: "Tanpa Kelaparan",
    3: "Kesehatan dan Kesejahteraan",
    4: "Pendidikan Berkualitas",
    5: "Kesetaraan Gender",
    6: "Air Bersih dan Sanitasi Layak",
    7: "Energi Bersih dan Terjangkau",
    8: "Pekerjaan Layak dan Pertumbuhan Ekonomi",
    9: "Industri, Inovasi dan Infrastruktur",
    10: "Berkurangnya Kesenjangan",
    11: "Kota dan Permukiman Berkelanjutan",
    12: "Konsumsi dan Produksi Bertanggung Jawab",
    13: "Penanganan Perubahan Iklim",
    14: "Ekosistem Laut",
    15: "Ekosistem Darat",
    16: "Perdamaian, Keadilan dan Kelembagaan",
    17: "Kemitraan untuk Mencapai Tujuan"
}

# Encode label SDGs
sdg_embeddings = model.encode(list(sdg_labels.values()), convert_to_tensor=True)

# Fungsi klasifikasi
def classify(text):
    if pd.isna(text) or len(str(text).strip()) == 0:
        return "Tidak diklasifikasikan"
    embedding = model.encode(str(text), convert_to_tensor=True)
    cosine_scores = util.cos_sim(embedding, sdg_embeddings)[0]
    best_idx = int(cosine_scores.argmax())
    return sdg_labels[best_idx + 1]

print(df.columns)

# Proses pelabelan
df['SDG_Category'] = df['processed_text'].apply(classify)

# Simpan hasil
df.to_csv("labeled_sinta.csv", index=False)
print("Selesai melabeli data. Hasil disimpan ke labeled_sinta.csv")
