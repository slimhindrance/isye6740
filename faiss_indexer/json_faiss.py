import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import faiss
from dotenv import load_dotenv
from tqdm import tqdm
from docx import Document as DocxDocument  # For DOCX processing
import fitz  # PyMuPDF for PDF processing

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

# Process JSON, DOCX, and PDF files
documents = []
for filename in tqdm(os.listdir(json_folder), desc="Processing Files", unit="file"):

    # Processing JSON Threads
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

            # Add JSON threads with medium authority
            documents.append(Document(page_content=thread_content, metadata={"source": filename, "authority_level": 1}))

    # Processing DOCX Files
    elif filename.endswith(".docx"):
        docx_path = os.path.join(json_folder, filename)
        docx = DocxDocument(docx_path)

        # Extract text from all paragraphs
        docx_text = "\n".join([para.text for para in docx.paragraphs if para.text.strip() != ""])

        # Add DOCX files with the highest authority
        documents.append(Document(page_content=docx_text, metadata={"source": filename, "authority_level": 3}))

    # Processing PDF Files
    elif filename.endswith(".pdf"):
        pdf_path = os.path.join(json_folder, filename)
        pdf_doc = fitz.open(pdf_path)

        pdf_text = ""
        for page in pdf_doc:
            pdf_text += page.get_text()

        pdf_doc.close()

        # Add PDF files with the highest authority
        documents.append(Document(page_content=pdf_text.strip(), metadata={"source": filename, "authority_level": 3}))

# Add documents to FAISS and save index
if documents:
    vector_store.add_documents(documents)
    vector_store.save_local(faiss_index_path)
    print(f"✅ FAISS index updated with {len(documents)} documents!")

else:
    print("❌ No documents found to add.")