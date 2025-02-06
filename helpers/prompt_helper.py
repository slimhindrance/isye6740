# helpers/prompt_helper.py
def format_rag_prompt(query, retrieved_docs):
    """
    Creates a concise prompt that forces the model to generate a proper response.
    """
    # ✅ Keep only the most relevant context
    context = "\n\n".join(
        [f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content[:800].strip()}..."  # Truncate text
         for doc in retrieved_docs]
    )

    # ✅ Force the model to generate a direct answer
    prompt = (
        "You are a helpful assistant for this course. You are effectively a teaching assistant. Follow these rules while answering:\n"
        "- Address the person asking the question as 'you', not 'the user'.\n"
        "- Provide specific details when available.\n"
        "- If the answer is not found in the documents, say: 'I don't have enough information to answer this question.'\n"
        "- Keep your answer concise and clear.\n\n"

        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer concisely and accurately:"
    )
    return prompt
