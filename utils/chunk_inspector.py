def show_original_document(loader_name, documents):
    print("Loader:", loader_name)
    print("Number of Documents:", len(documents))

    total_length = 0
    for doc in documents:
        total_length = total_length + len(doc.page_content)
    print("Original Length:", total_length, "characters")

    print("Metadata:", documents[0].metadata)
    print("page_content (first 300 characters):")
    print(documents[0].page_content[:300])
    print()


def show_splitter_info(splitter_name, chunk_size, chunk_overlap):
    print("Splitter:", splitter_name)
    print("Chunk Size:", chunk_size)
    print("Chunk Overlap:", chunk_overlap)


def show_result(chunks):
    print("Number of Chunks:", len(chunks))


def show_chunk(chunks, number):
    chunk = chunks[number]
    print("Chunk Number:", number)
    print("Length:", len(chunk.page_content), "characters")
    print("Content:")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)


def show_statistics(chunks):
    lengths = []
    for chunk in chunks:
        lengths.append(len(chunk.page_content))

    if len(lengths) == 0:
        print("Total Chunks: 0")
        return

    print("Total Chunks:", len(lengths))
    print("Average Chunk Length:", round(sum(lengths) / len(lengths), 1))
    print("Minimum Chunk Length:", min(lengths))
    print("Maximum Chunk Length:", max(lengths))