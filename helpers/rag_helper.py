import os
import torch
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_core.documents import Document
from helpers.prompt_helper import format_rag_prompt
from helpers.llm_loader import chat
import gc
import difflib

# Clear CUDA cache
torch.cuda.empty_cache()

# Load environment variables
load_dotenv()
faiss_index_path = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
hf_model_name = os.getenv("HF_MODEL", "meta-llama/Llama-2-7b-chat-hf")

# Force embeddings to run on CPU to free VRAM
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"}
)

# Function to load FAISS index
def load_faiss_index():
    print("🔄 Loading FAISS vector store...")
    try:
        vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        print("✅ FAISS vector store loaded successfully!")
        return retriever
    except Exception as e:
        print(f"❌ ERROR: Could not load FAISS vector store - {e}")
        return None

retriever = load_faiss_index()

# Set device for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running model on {device}")

# Use 4-bit quantization to save VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load model with CPU offloading
print(f"🔄 Loading model: {hf_model_name}")
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

model = AutoModelForCausalLM.from_pretrained(
    hf_model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Function to handle RAG-based queries
def generate_rag_response(query):
    if not retriever:
        return "❌ ERROR: RAG pipeline is not initialized."

    try:
        # Step 1: Retrieve relevant documents
        retrieved_docs = retriever.invoke(query)

        if not retrieved_docs:
            return "I don't have enough information to answer this question."

        # Step 2: Check for authoritative documents
        high_authority_docs = [doc for doc in retrieved_docs if doc.metadata.get("authority_level", 0) >= 3]

        for doc in high_authority_docs:
            if query.lower() in doc.page_content.lower():
                return f"🔒 Authoritative Source:\n{doc.page_content.strip()}"

        # Step 3: Extract exact matches
        for doc in retrieved_docs:
            if query.lower() in doc.page_content.lower():
                return doc.page_content.strip()

        # Step 4: Format the query and context for the LLM
        prompt = format_rag_prompt(query, retrieved_docs)

        # Step 5: Tokenize and generate LLM response
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id
            )

        # Step 6: Decode and clean LLM response
        response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        response_cleaned = response.split('Answer concisely and accurately:')[-1].strip()

        # Step 7: Post-process validation against authoritative documents
        for doc in high_authority_docs:
            similarity = difflib.SequenceMatcher(None, response_cleaned.lower(), doc.page_content.lower()).ratio()
            if similarity > 0.85:
                return f"🔒 Verified with Authoritative Source:\n{doc.page_content.strip()}"

        # Step 8: Return LLM's response if no authoritative match
        return response_cleaned if response_cleaned else "I don't have enough information to answer this question."

    except Exception as e:
        return f"❌ ERROR: {str(e)}"

    finally:
        # Step 9: Clear memory to prevent CUDA memory issues
        if 'inputs' in locals():
            del inputs
        if 'output' in locals():
            del output
        torch.cuda.empty_cache()
        gc.collect()
