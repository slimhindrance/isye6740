from transformers import AutoTokenizer, AutoModel
import torch
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
import faiss
from langchain_community.docstore import InMemoryDocstore

# Load the embedding model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

# Initialize an empty FAISS index with the correct dimension
embedding_function = generate_embeddings
index = faiss.IndexFlatL2(1536)  # Updated to 1536-dimensional embeddings
docstore = InMemoryDocstore()
index_to_docstore_id = {}

vectorstore = FAISS(embedding_function, index, docstore, index_to_docstore_id)

# Path to the folder containing text files
folder_path = "ed/clean_threads"

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):  # Ensure you're only loading text files
        file_path = os.path.join(folder_path, filename)
        
        # Load data
        loader = TextLoader(file_path)
        documents = loader.load()

        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        # Generate embeddings for each chunk and add to FAISS index one at a time
        for chunk in chunks:
            print(f"\n\nCHNK:\n{type(chunk)}\n{chunk.page_content}\n\n")
            chunk_embedding = generate_embeddings(chunk.page_content)
            print(f"Embedding shape: {chunk_embedding.shape}\n\n")
            
            # Ensure the embedding is a 2D array
            if len(chunk_embedding.shape) == 1:
                chunk_embedding = chunk_embedding.reshape(1, -1)
            
            # Check the shape of the embedding before adding
            print(f"Reshaped Embedding shape: {chunk_embedding.shape}\n\n")
            
            #vectorstore.add_texts([chunk])
            vectorstore.add_embeddings((chunk, chunk_embedding))
            # Free up unused CUDA memory
            torch.cuda.empty_cache()

# Save the FAISS index to disk
vectorstore.save("faiss_index")
