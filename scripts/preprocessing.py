import pandas as pd
import re

# Load data
df = pd.read_csv("data/raw/sinta_scraped_data.csv")

# Tentukan kolom teks
text_column = 'Title' if 'Title' in df.columns else df.columns[0]

# Hapus duplikat awal berdasarkan kolom Title
df = df.drop_duplicates(subset=text_column)

# Stopword list sederhana
stopwords = {
    "yang", "dan", "atau", "di", "ke", "dari", "untuk", "dengan", "pada",
    "sebagai", "adalah", "itu", "ini", "dalam", "oleh", "karena", "juga",
    "lebih", "agar", "akan", "tersebut", "telah", "menjadi", "bahwa", "maka"
}

# Fungsi preprocessing


def preprocess(text):
    if pd.isna(text):
        return ""
    text = text.lower()  # lowercase
    text = re.sub(r'[^\w\s]', '', text)  # hapus tanda baca
    tokens = text.split()
    # hapus stopword
    tokens = [word for word in tokens if word not in stopwords]
    return ' '.join(tokens)


# Terapkan preprocessing
df['processed_text'] = df[text_column].apply(preprocess)

# Hapus duplikat lagi setelah preprocessing
df = df.drop_duplicates(subset='processed_text')

# Simpan hasil
df.to_csv("preprocessed_sinta_no_nltk.csv", index=False)
print("✅ Preprocessing selesai. Disimpan ke preprocessed_sinta_no_nltk.csv")
