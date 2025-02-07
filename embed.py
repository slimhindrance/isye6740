import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


import os

# Path to the threads subfolder
threads_path = "ed/clean_threads/"
# List to store document content
documents = []
# Read the content of each text file in the subfolder
for file_name in os.listdir(threads_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(threads_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().strip()
            if content:  # Ensure the file is not empty
                documents.append(content)

print(f"Loaded {len(documents)} documents.")

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for each document
document_embeddings = model.encode(documents)#, show_progress_bar=True)


# Convert embeddings to numpy array
embeddings_array = np.array(document_embeddings).astype('float32')

# Create FAISS index
index = faiss.IndexFlatL2(embeddings_array.shape[1])  # L2 distance (Euclidean)
index.add(embeddings_array)

print(f"Indexed {index.ntotal} documents.")


faiss.write_index(index, "faiss_index")
