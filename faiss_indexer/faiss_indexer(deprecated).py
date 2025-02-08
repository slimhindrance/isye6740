from langchain_huggingface import HuggingFaceEmbeddings
import faiss, os
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.documents import Document
import pickle  # Required for saving metadata
from dotenv import load_dotenv
import os
import torch
from text_utils import clean_text

torch.cuda.empty_cache()  # ✅ Frees up unused memory
torch.cuda.memory_allocated()  # ✅ Shows allocated memory

# Find the root directory where .env is stored
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script's folder
root_dir = os.path.abspath(os.path.join(base_dir, ".."))  # Go up to the root folder


# Load .env from root
dotenv_path = os.path.join(root_dir, ".env")
load_dotenv(dotenv_path)

input_dir = os.getenv("INPUT_DIR", "./data")  # Default to ./data if not set
faiss_index_path = os.getenv("FAISS_INDEX_PATH", "./faiss_index")  # Default save path

print(f"Loading text files from: {input_dir}")
print(f"FAISS index will be saved to: {faiss_index_path}")

# Get all .txt filenames
filenames = [x for x in os.listdir(input_dir) if x.endswith(".txt")]

# Load the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Check if FAISS index exists
index_file = os.path.join(faiss_index_path, "index.faiss")

"""if os.path.exists(index_file):
    print("🔄 Loading existing FAISS index...")
    vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
else:"""
if 1==1:
    print("🆕 FAISS index not found. Creating a new one...")
    os.makedirs(faiss_index_path, exist_ok=True)  # Ensure save directory exists

    # Create a new FAISS index
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    # Convert text files into LangChain Document objects
    documents = []
    for filename in filenames:
        file_path = os.path.join(input_dir, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            cleaned_text = clean_text(text)
            doc = Document(page_content=cleaned_text, metadata={"source": filename})
            documents.append(doc)

    # Add documents to FAISS vector store
    if documents:
        vector_store.add_documents(documents=documents)
        print(f"✅ Successfully added {len(documents)} documents to FAISS!")

    # Save FAISS index
    vector_store.save_local(faiss_index_path)
    print(f"💾 FAISS index saved to {faiss_index_path}!")