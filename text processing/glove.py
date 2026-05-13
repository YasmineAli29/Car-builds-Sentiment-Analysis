import numpy as np

def load_glove(file_path):
    embeddings = {}
    with open(file_path, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    return embeddings


def text_to_glove_vector(text, glove, dim):
    words = text.lower().split()

    vectors = [glove[w] for w in words if w in glove]

    if len(vectors) == 0:
        return np.zeros(dim)

    return np.mean(vectors, axis=0)