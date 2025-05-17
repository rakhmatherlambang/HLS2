import pandas as pd
import os
from transformers import pipeline
from tqdm import tqdm
import torch

output_dir = "data_label"
os.makedirs(output_dir, exist_ok=True)
# Cek apakah GPU tersedia
device = 0 if torch.cuda.is_available() else -1

# Load data
df = pd.read_csv("sinta_unila.csv")

# Ambil hanya 1000 data pertama (sementara hanya 10 untuk testing)
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
                      device=device)

# Threshold confidence
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

# Lakukan klasifikasi
df['SDGs'] = df['Title'].progress_apply(classify_sdg_with_threshold)

# Pisahkan data
df_certain = df[df['SDGs'] != "Uncertain"]
df_uncertain = df[df['SDGs'] == "Uncertain"]

# Simpan hasil ke file berbeda
df_certain.to_csv(os.path.join(output_dir, "sinta_labeled_10.csv"), index=False)
df_uncertain.to_csv(os.path.join(output_dir, "sinta_uncertain_10.csv"), index=False)
