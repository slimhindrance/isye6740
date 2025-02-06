import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import faiss
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
# Define paths
json_folder = os.getenv("JSON_THREADS_DIR", "/app/json_threads")
faiss_index_path = os.getenv("JSON_FAISS_INDEX_PATH", "/app/faiss")

# Load the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Always create a new FAISS index
print("🆕 Creating a new FAISS index...")
os.makedirs(faiss_index_path, exist_ok=True)  # Ensure save directory exists

# Initialize FAISS index
index = faiss.IndexFlatL2(len(embeddings.embed_query("sample")))  # ✅ Use len() instead of .shape[0]

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Load and process your JSON threads as before...

# Process JSON files
documents = []
 # Initialize tqdm to wrap the iterable
for i, filename in enumerate(tqdm(os.listdir(json_folder), desc="Processing JSON Threads", unit="thread")):

    if filename.endswith(".json"):
        with open(os.path.join(json_folder, filename), 'r', encoding='utf-8') as file:
            thread_data = json.load(file)
            
            # Extract thread content
            thread_id = thread_data.get('id')
            thread_content = f"Thread ID: {thread_id}\nUser {thread_data['user_id']} (Parent Post) says:\n{thread_data.get('document', '')}\n"

            # Add comments and answers
            for comment in thread_data.get('comments', []):
                thread_content += f"Comment from User {comment['user_id']}:\n{comment.get('document', '')}\n"
                for sub_comment in comment.get('comments', []):
                    thread_content += f"  Reply from User {sub_comment['user_id']}:\n  {sub_comment.get('document', '')}\n"

            for answer in thread_data.get('answers', []):
                thread_content += f"Answer from User {answer['user_id']}:\n{answer.get('document', '')}\n"
                for sub_comment in answer.get('comments', []):
                    thread_content += f"  Comment from User {sub_comment['user_id']}:\n  {sub_comment.get('document', '')}\n"

            # Add to documents
            documents.append(Document(page_content=thread_content, metadata={"source": filename}))

# Add documents to FAISS and save index
if documents:
    vector_store.add_documents(documents)
    vector_store.save_local(faiss_index_path)
    print(f"✅ FAISS index updated with {len(documents)} documents!")

else:
    print("❌ No documents found to add.")