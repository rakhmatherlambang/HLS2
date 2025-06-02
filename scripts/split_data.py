import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(input_file, label_col='label'):
    df = pd.read_csv(input_file)

    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df[label_col])
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df[label_col])

    train_df.to_csv('data/split/train.csv', index=False)
    val_df.to_csv('data/split/val.csv', index=False)
    test_df.to_csv('data/split/test.csv', index=False)
    print("Data splitting selesai!")


if __name__ == "__main__":
    # ganti sesuai nama file final kamu
    split_data('data/processed/labeled_sinta.csv')
