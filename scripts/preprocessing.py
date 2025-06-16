import pandas as pd
import requests
import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download nltk data jika belum ada
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# Kelas Preprocessing


class Preprocessing:
    def clean_text(self, texts):
        texts = str(texts).lower()
        texts = re.sub("[^0-9a-z]+", " ", texts)  # hanya huruf dan angka
        texts = texts.strip()
        return texts

    def tokenize_text(self, texts):
        sentences = nltk.sent_tokenize(texts, language='english')
        words = [nltk.word_tokenize(sent) for sent in sentences]
        return words

    def remove_wordstop(self, words):
        # Stopwords gabungan
        from nltk.corpus import stopwords


    def remove_wordstop(self, words):
        # pastikan ini hanya dipanggil sekali, atau di awal
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        return [word for word in words if word not in stop_words]


    def stemming_text(self, text):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        return stemmer.stem(text)

    def pre_process(self, text):
        text = self.clean_text(text)
        text = self.stemming_text(text)
        words = self.tokenize_text(text)
        words = self.remove_wordstop(words)
        return words[0]  # Ambil kata dari kalimat pertama

# ==== PENGGUNAAN ====


# 1. Load file CSV
df = pd.read_csv("data/raw/sinta_scraped_data.csv")  # ganti dengan path file CSV kamu

# 2. Buat objek preprocessing
prep = Preprocessing()

# 3. Proses kolom 'Title' saja (atau bisa ditambah sesuai kebutuhan)
df['Processed_Title'] = df['Title'].apply(prep.pre_process)

# 4. Simpan hasil ke file baru (opsional)
df.to_csv("sinta_processed.csv", index=False)

print(df[['Title', 'Processed_Title']].head())
