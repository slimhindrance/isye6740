import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from dotenv import load_dotenv
from tqdm import tqdm
from docx import Document as DocxDocument  # For DOCX processing
import fitz  # PyMuPDF for PDF processing
import torch, gc
load_dotenv()

# Define paths
json_folder = os.getenv("JSON_THREADS_DIR", "/app/json_threads")
faiss_index_path = os.getenv("FAISS_INDEX_PATH", "/app/faiss")
docx_folder = os.getenv("DOCX_DIR", "/app/json_threads")
pdf_folder = os.getenv("PDF_DIR", "/app/json_threads")
output_folder = os.getenv("TEST_DIR")

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

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Adjust the size depending on your needs
    chunk_overlap=200,  # Overlap between chunks for better context continuity
)


def save_conversation_to_file(content, thread_id, comment_id, output_folder=output_folder):
    """
    Save the processed conversation chain to a uniquely named text file.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists
    
    # Use both thread_id and comment_id for unique filenames
    filename = os.path.join(output_folder, f"thread_{thread_id}_comment_{comment_id}.txt")
    
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)


# Process JSON, DOCX, and PDF files
# Function to process conversation chains
def process_comments(comment_list, level=0):
    """
    Recursively process comments and nested replies to maintain conversation flow.
    """
    conversation = ""
    indent = "  " * level  # Indentation based on the depth of the comment

    for comment in comment_list:
        user_id = comment.get('user_id', 'ANON')
        content = comment.get('document', '').strip()

        # Add the comment at the current level
        conversation += f"{indent}Comment from User {user_id}:\n{indent}{content}\n"

        # Recursively process any nested comments (replies)
        if 'comments' in comment and isinstance(comment['comments'], list):
            conversation += process_comments(comment['comments'], level=level+1)

    return conversation


def process_json_threads_and_save(json_folder, text_splitter, output_folder):
    json_documents = []

    for filename in tqdm(os.listdir(json_folder), desc="Processing JSON Threads", unit="file"):
        if filename.endswith(".json"):
            with open(os.path.join(json_folder, filename), 'r', encoding='utf-8') as file:
                thread_data = json.load(file)
                thread_id = thread_data.get('id')

                # Parent post (thread's main idea)
                parent_post = f"Thread ID: {thread_id}\nUser {thread_data['user_id']} (Original Post) says:\n{thread_data.get('document', '').strip()}\n\n"

                # Process top-level comments individually as separate conversation flows
                for comment in thread_data.get('comments', []):
                    conversation_flow = parent_post + process_comment_chain(comment)

                    # Save each unique conversation flow with thread_id and comment_id
                    save_conversation_to_file(conversation_flow, thread_id, comment['id'], output_folder)

                    # Split into manageable chunks for FAISS indexing
                    chunks = text_splitter.split_text(conversation_flow)

                    for chunk in chunks:
                        json_documents.append(Document(
                            page_content=chunk,
                            metadata={
                                "source": filename,
                                "thread_id": thread_id,
                                "comment_id": comment['id'],  # Track the comment origin
                                "authority_level": 1
                            }
                        ))

    return json_documents


def process_comment_chain(comment, depth=0):
    """
    Recursively process a comment and its nested replies.
    """
    indent = "  " * depth  # Indentation for nested comments
    comment_text = f"{indent}Comment from User {comment['user_id']}:\n{indent}{comment.get('document', '').strip()}\n\n"

    # Recursively process nested replies
    for reply in comment.get('comments', []):
        comment_text += process_comment_chain(reply, depth + 1)

    return comment_text

# Process JSON Threads
json_documents = process_json_threads_and_save(json_folder, text_splitter,output_folder)

docx_documents = []
for filename in tqdm(os.listdir(docx_folder), desc="Processing DOCX Files", unit="file"):
    # Processing DOCX Files
    if filename.endswith(".docx"):
        docx_path = os.path.join(docx_folder, filename)
        docx = DocxDocument(docx_path)

        # Extract text from all paragraphs
        docx_text = "\n".join([para.text for para in docx.paragraphs if para.text.strip() != ""])

        # Split DOCX text into manageable chunks
        chunks = text_splitter.split_text(docx_text)
        for chunk in chunks:
            docx_documents.append(Document(page_content=chunk, metadata={"source": filename, "authority_level": 3}))

pdf_documents = []
for filename in tqdm(os.listdir(pdf_folder), desc="Processing PDF Files", unit="file"):
    # Processing PDF Files
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        pdf_doc = fitz.open(pdf_path)

        pdf_text = ""
        for page in pdf_doc:
            pdf_text += page.get_text()

        pdf_doc.close()

        # Split PDF text into chunks
        chunks = text_splitter.split_text(pdf_text.strip())
        for chunk in chunks:
            pdf_documents.append(Document(page_content=chunk, metadata={"source": filename, "authority_level": 3}))

# Add documents to FAISS and save index
if json_documents:
    vector_store.add_documents(json_documents)
    vector_store.save_local(faiss_index_path)
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()
    print(f"✅ FAISS index updated with {len(json_documents)} JSON conversation chains!")
else:
    print("❌ No JSON documents found to add.")

# Add documents to FAISS and save index
if docx_documents:
    vector_store.add_documents(docx_documents)
    vector_store.save_local(faiss_index_path)
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()
    print(f"✅ FAISS index updated with {len(docx_documents)} DOCX documents!")
else:
    print("❌ No DOCX documents found to add.")

# Add documents to FAISS and save index
if pdf_documents:
    vector_store.add_documents(pdf_documents)
    vector_store.save_local(faiss_index_path)
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()
    print(f"✅ FAISS index updated with {len(pdf_documents)} PDF documents!")

else:
    print("❌ No PDF documents found to add.")