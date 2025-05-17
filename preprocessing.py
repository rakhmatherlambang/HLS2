import pandas as pd
import re

# Load data
df = pd.read_csv("sinta_unila.csv")

# Ganti kolom target jika berbeda
text_column = 'Title' if 'Title' in df.columns else df.columns[0]

# Stopword sederhana (tanpa NLTK)
stopwords = set([
    "yang", "dan", "atau", "di", "ke", "dari", "untuk", "dengan", "pada",
    "sebagai", "adalah", "itu", "ini", "dalam", "oleh", "karena", "juga",
    "lebih", "agar", "akan", "tersebut", "telah", "menjadi", "bahwa", "maka"
])

# Fungsi preprocessing
def preprocess(text):
    if pd.isna(text): return ""
    text = text.lower()  # case folding
    text = re.sub(r'[^\w\s]', '', text)  # punctuation removal
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords]  # stopword removal
    return ' '.join(tokens)

# Proses semua
df['processed_text'] = df[text_column].apply(preprocess)

# Simpan
df.to_csv("preprocessed_sinta_no_nltk.csv", index=False)
print("Preprocessing selesai dan disimpan ke preprocessed_sinta_no_nltk.csv")
