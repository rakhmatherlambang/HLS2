import pandas as pd
from sklearn.model_selection import train_test_split
import os


def split_data(input_path, output_dir='data/split', text_col='Title', label_col='SDG_Category', test_size=0.2, val_size=0.1):
    # Load data
    df = pd.read_csv(input_path)

    # Hapus duplikat berdasarkan kolom teks
    df = df.drop_duplicates(subset=[text_col])

    # Pastikan tidak ada missing label
    df = df.dropna(subset=[label_col])

    # Bagi data menjadi train+val dan test terlebih dahulu
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df[label_col]
    )

    # Hitung ulang val_size proporsional terhadap data train_val
    val_relative_size = val_size / (1 - test_size)

    # Split train dan val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        random_state=42,
        stratify=train_val_df[label_col]
    )

    # Buat folder jika belum ada
    os.makedirs(output_dir, exist_ok=True)

    # Simpan hasil split
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print(f"Data berhasil displit dan disimpan di folder '{output_dir}'")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


# Contoh pemanggilan
if __name__ == '__main__':
    split_data('data/processed/labeled_sinta.csv')
