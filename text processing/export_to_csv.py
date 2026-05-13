import pandas as pd
import numpy as np
from scipy import sparse
import os

# ---------------- PATHS ----------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
REP_DIR = os.path.join(ROOT, "representations")

OUTPUT_DIR = DATA_DIR
INPUT_DIR = REP_DIR

# ---------------- GLOVE ----------------
def export_glove_to_csv(npy_path, output_path):
    print(f"Exporting full GloVe from {npy_path} to {output_path}...")

    vectors = np.load(npy_path)   # ✅ NO LIMIT

    columns = [f"glove_{i}" for i in range(vectors.shape[1])]
    df = pd.DataFrame(vectors, columns=columns)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"GloVe CSV saved: {output_path} (Shape: {df.shape})")


# ---------------- BOW ----------------
def export_bow_to_csv(npz_path, vocab_path, output_path):
    print(f"Exporting full BoW from {npz_path} to {output_path}...")

    matrix = sparse.load_npz(npz_path)   # ✅ NO LIMIT

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]

    if len(vocab) != matrix.shape[1]:
        print(f"Error: vocab ({len(vocab)}) != features ({matrix.shape[1]})")
        return

    dense_df = pd.DataFrame(matrix.toarray(), columns=vocab)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dense_df.to_csv(output_path, index=False)

    print(f"BoW CSV saved: {output_path} (Shape: {dense_df.shape})")


# ---------------- MAIN ----------------
def main():
    schemes = ["scheme1", "scheme2", "scheme3"]

    for scheme in schemes:
        print(f"\nProcessing {scheme}...")

        glove_npy = os.path.join(INPUT_DIR, f"{scheme}_glove.npy")
        bow_npz = os.path.join(INPUT_DIR, f"{scheme}_bow.npz")
        vocab_txt = os.path.join(INPUT_DIR, f"{scheme}_vocab.txt")

        glove_csv = os.path.join(OUTPUT_DIR, scheme, f"{scheme}_glove.csv")
        bow_csv = os.path.join(OUTPUT_DIR, scheme, f"{scheme}_bow.csv")

        if os.path.exists(glove_npy):
            export_glove_to_csv(glove_npy, glove_csv)

        if os.path.exists(bow_npz) and os.path.exists(vocab_txt):
            export_bow_to_csv(bow_npz, vocab_txt, bow_csv)


if __name__ == "__main__":
    main()