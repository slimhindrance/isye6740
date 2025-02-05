import os
import torch
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.documents import Document

# Load environment variables
load_dotenv()
faiss_index_path = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
hf_model_name = os.getenv("HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# ✅ Force embeddings to run on CPU to free VRAM
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"}  # ✅ Runs embeddings on CPU
)

# Function to load FAISS index
def load_faiss_index():
    print("🔄 Loading FAISS vector store...")
    try:
        vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 relevant docs
        print("✅ FAISS vector store loaded successfully!")
        return retriever
    except Exception as e:
        print(f"❌ ERROR: Could not load FAISS vector store - {e}")
        return None

retriever = load_faiss_index()

# ✅ Set device for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running model on {device}")

# ✅ Load Smaller Model with CPU Offloading
print(f"🔄 Loading model: {hf_model_name}")
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

model = AutoModelForCausalLM.from_pretrained(
    hf_model_name,
    torch_dtype=torch.float16,  # ✅ Keep FP16 for better performance
    device_map="auto"  # ✅ Automatically offload to CPU if needed
)

# Function to handle RAG-based queries efficiently
def generate_rag_response(query):
    if not retriever:
        return "❌ ERROR: RAG pipeline is not initialized."

    try:
        # Retrieve relevant documents
        retrieved_docs = retriever.get_relevant_documents(query)

        # Combine retrieved text into context
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # ✅ Reduce context size to prevent memory overload
        max_input_tokens = 256  # 🔥 Reduce input size to save VRAM
        tokenized_context = tokenizer(context, truncation=True, max_length=max_input_tokens)
        truncated_context = tokenizer.decode(tokenized_context["input_ids"], skip_special_tokens=True)

        # Format input
        input_text = f"Context:\n{truncated_context}\n\nUser Question: {query}"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        # ✅ Limit output size to prevent excessive VRAM usage
        output = model.generate(**inputs, max_new_tokens=30)  # 🔥 Reduce max tokens
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        return response
    except Exception as e:
        return f"❌ ERROR: {str(e)}"