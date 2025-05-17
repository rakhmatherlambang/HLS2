import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import torch

# Cek apakah GPU tersedia
device = 0 if torch.cuda.is_available() else -1

# Load data
df = pd.read_csv("sinta_unila.csv")

# Ambil hanya 1000 data pertama
df = df.head(10)

# Daftar label SDGs
sdg_labels = [
    "No Poverty",
    "Zero Hunger",
    "Good Health and Well Being",
    "Qualilty Education",
    "Gender Equality",
    "Clean Water and Sanitation",
    "Affordable and Clean Energy",
    "Decent Work and Economic Growth",
    "Industry, Innovation, and Infrastructure",
    "Reduce Inequalities",
    "Sustainable Cities and Communities",
    "Responsible Consumption and production",
    "Climate Action",
    "Life Below Water",
    "Life on Land",
    "Peace, Justice, and strong Institution",
    "Partnership for the goals"
]

# Load zero-shot classifier dengan GPU jika tersedia
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli",
                      device=device)  # device=0 untuk GPU, -1 untuk CPU

# Ambang batas confidence
CONFIDENCE_THRESHOLD = 0.3

# Fungsi klasifikasi


def classify_sdg_with_threshold(title, threshold=CONFIDENCE_THRESHOLD):
    result = classifier(title, sdg_labels, multi_label=True)
    selected_labels = [
        label for label, score in zip(result["labels"], result["scores"]) if score >= threshold
    ]
    return "; ".join(selected_labels) if selected_labels else "Uncertain"


# Progres bar
tqdm.pandas()

# Labeling
df['SDGs'] = df['Title'].progress_apply(classify_sdg_with_threshold)

# Simpan hasil
df.to_csv("sinta_labeled_10.csv", index=False)
