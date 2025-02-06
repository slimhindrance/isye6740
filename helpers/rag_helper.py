import os
import torch
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_core.documents import Document
from helpers.prompt_helper import format_rag_prompt
from helpers.llm_loader import chat
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc


torch.cuda.empty_cache()

# Load environment variables
load_dotenv()
faiss_index_path = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
hf_model_name = os.getenv("HF_MODEL", "meta-llama/Llama-2-7b-chat-hf")

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
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Retrieve top k relevant docs
        print("✅ FAISS vector store loaded successfully!")
        return retriever
    except Exception as e:
        print(f"❌ ERROR: Could not load FAISS vector store - {e}")
        return None

retriever = load_faiss_index()

# ✅ Set device for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running model on {device}")

# ✅ Use 4-bit quantization to save VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# ✅ Load Smaller Model with CPU Offloading
print(f"🔄 Loading model: {hf_model_name}")
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

model = AutoModelForCausalLM.from_pretrained(
    hf_model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,  # ✅ Keep FP16 for better performance
    #attn_implementation="flash_attention_2",  # 🔥 Faster & lower VRAM
    device_map="auto"  # ✅ Automatically offload to CPU if needed
)

# Function to handle RAG-based queries efficiently

def generate_rag_response(query):
    if not retriever:
        return "❌ ERROR: RAG pipeline is not initialized."

    try:
        # ✅ Step 1: Retrieve relevant documents
        retrieved_docs = retriever.get_relevant_documents(query)

        if not retrieved_docs:
            return "I don't have enough information to answer this question."

        # ✅ Step 2: Try extracting a direct answer **before** using the LLM
        for doc in retrieved_docs:
            text = doc.page_content.strip()
            if query.lower() in text.lower():
                return text  # ✅ Return exact answer if found

        # ✅ Step 3: Format the query properly
        prompt = format_rag_prompt(query, retrieved_docs)

        # ✅ Step 4: Tokenize, send to model, and generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=50,  # ✅ Limit token generation
                pad_token_id=tokenizer.eos_token_id  # ✅ Prevents infinite padding
            )

        # ✅ Step 5: Decode response
        response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # ✅ Step 6: Remove prompt from response
        response = response.replace("Provide a direct, concise, and factual answer using only the context above.", "").strip()
        response = response.replace("If the context does not contain the answer, respond with:", "").strip()

        # ✅ Step 7: Clean memory to prevent CUDA crashes
        del inputs, output
        torch.cuda.empty_cache()
        gc.collect()

        return str(response.split('Answer concisely and accurately:')[-1]) if response else "I don't have enough information to answer this question."
    except Exception as e:
        return f"❌ ERROR: {str(e)}"