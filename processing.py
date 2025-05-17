import pandas as pd

# Load your data
df = pd.read_csv('sinta_labeled_10.csv')  ##.head(1000,2000)
# df = df.iloc[1000:2000].copy()

# Define what counts as uncertain in your context
uncertain_values = ["", " ", "?", "unknown", "Uncertain", None]

# Optionally, define the valid list of SDG labels
valid_sdgs = [f"SDGs {i}" for i in range(1, 18)]

# Find rows with uncertain or invalid SDGs
uncertain_rows = df[
    df['SDGs'].isnull() |  # missing values
    # common uncertain strings
    df['SDGs'].astype(str).str.strip().isin(uncertain_values) |
    ~df['SDGs'].isin(valid_sdgs)  # not in the valid SDG list
]

# Display or save
print(f"Found {len(uncertain_rows)} uncertain rows in 'sdg' column.")
print(uncertain_rows[['SDGs']])  # or add more columns for context

# Hitung jumlah baris dengan label 'Uncertain'
uncertain_count = (df['SDGs'] == 'Uncertain').sum()
print(
    f"Jumlah baris (Uncertain) dari 1000 pertama: {uncertain_count}")

# total_rows = len(df)
# percentage = (uncertain_count / total_rows)
# print(f"Percentage Uncertain : {percentage:.2f}% dari total {total_rows}")
