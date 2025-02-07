# helpers/prompt_helper.py
def format_rag_prompt(query, retrieved_docs):
    """
    Creates a concise prompt that forces the model to generate a proper response.
    """
    # ✅ Keep only the most relevant context
    context = "\n\n".join(
        [f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content[:300].strip()}..."  # Truncate text
         for doc in retrieved_docs]
    )

    # ✅ Force the model to generate a direct answer
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer concisely and accurately:"
    )
    return prompt