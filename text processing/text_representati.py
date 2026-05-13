import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import os

# Paths
INPUT_CSV = "data/flash/full_clean_yt_comments.csv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GLOVE_PATH = os.path.join(BASE_DIR, "..", "data", "glove.6B.100d.txt")
OUTPUT_DIR = "representations"

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    # Ensure clean_comment exists and handle nans
    if 'clean_comment' not in df.columns:
        print("Error: 'clean_comment' column not found.")
        return None
    df['clean_comment'] = df['clean_comment'].fillna("")
    return df['clean_comment'].tolist()

def generate_bow(texts, output_path, output_prefix):
    print(f"Generating Bag-of-Words (BoW) for {output_prefix}...")
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(texts)
    
    # Save sparse matrix and vocabulary
    sparse.save_npz(output_path, bow_matrix)
    
    vocab_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_vocab.txt")
    with open(vocab_path, "w", encoding='utf-8') as f:
        for word in vectorizer.get_feature_names_out():
            f.write(f"{word}\n")
            
    print(f"BoW saved to {output_path} (Shape: {bow_matrix.shape})")
    print(f"Vocabulary saved to {vocab_path}")
    return bow_matrix

def load_glove_embeddings(path):
    print(f"Loading GloVe embeddings from {path}...")
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings[word] = vector
    print(f"Loaded {len(embeddings)} word vectors.")
    return embeddings

def generate_glove_average(texts, embeddings, output_path):
    print("Generating GloVe average representations...")
    vectors = []
    vector_dim = 100 # Predefined from glove.6B.100d.txt
    
    for text in texts:
        words = text.lower().split()
        word_vectors = [embeddings[w] for w in words if w in embeddings]
        
        if word_vectors:
            mean_vector = np.mean(word_vectors, axis=0)
        else:
            # If no words in text are in GloVe, use a zero vector
            mean_vector = np.zeros(vector_dim)
            
        vectors.append(mean_vector)
    
    vectors_array = np.array(vectors)
    np.save(output_path, vectors_array)
    print(f"GloVe representations saved to {output_path} (Shape: {vectors_array.shape})")
    return vectors_array

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate text representations (BoW & GloVe).")
    parser.add_argument("--input", type=str, required=True, help="Input CSV path")
    parser.add_argument("--output_prefix", type=str, required=True, help="Prefix for output files (e.g. scheme1)")
    
    args = parser.parse_args()
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    texts = load_data(args.input)
    if texts is None:
        return
        
    # Phase 2: Bag-of-Words
    bow_path = os.path.join(OUTPUT_DIR, f"{args.output_prefix}_bow.npz")
    generate_bow(texts, bow_path, args.output_prefix)
    
    # Phase 3: GloVe
    glove_embeddings = load_glove_embeddings(GLOVE_PATH)
    glove_output_path = os.path.join(OUTPUT_DIR, f"{args.output_prefix}_glove.npy")
    generate_glove_average(texts, glove_embeddings, glove_output_path)
    
    print(f"Representation for {args.output_prefix} complete")

if __name__ == "__main__":
    main()
