from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents_recursive(documents, chunk_size, chunk_overlap):
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)
    return chunks