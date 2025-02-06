def check_document_size(documents, tokenizer, max_tokens=2048):
    """
    Checks which documents exceed the max token limit.

    Parameters:
        documents (list): List of LangChain Document objects.
        tokenizer: Tokenizer used to generate tokens.
        max_tokens (int): Maximum number of tokens allowed.

    Returns:
        large_docs (list): List of documents that exceed the token limit.
    """
    large_docs = []
    
    for doc in documents:
        tokens = tokenizer(doc.page_content, return_tensors="pt")
        token_count = tokens.input_ids.shape[1]
        
        if token_count > max_tokens:
            large_docs.append({
                "source": doc.metadata.get("source", "Unknown"),
                "token_count": token_count,
                "preview": doc.page_content[:200]  # Preview of content
            })

    return large_docs