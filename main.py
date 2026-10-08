from loaders.document_loader import load_txt_file, load_pdf_file, load_md_file
from splitters.recursive_splitter import split_documents_recursive
from splitters.other_splitter import split_documents_character
from utils.chunk_inspector import (
    show_original_document,
    show_splitter_info,
    show_result,
    show_chunk,
    show_statistics,
)

documents = load_txt_file("data/sample.txt")


print("=" * 50)
print("DOCUMENT LOADING")
print("=" * 50)

show_original_document("TextLoader (sample.txt)", documents)

print("A Document has two parts:")
print("  page_content -> the text itself")
print("  metadata     -> where the text came from")
print()

print("Other document types:")
pdf_documents = load_pdf_file("data/sample.pdf")
show_original_document("PyPDFLoader (sample.pdf)", pdf_documents)

md_documents = load_md_file("data/sample.md")
show_original_document("TextLoader (sample.md)", md_documents)


print("=" * 50)
print("CHUNK SIZE EXPERIMENT")
print("=" * 50)
print("Splitter: RecursiveCharacterTextSplitter")
print()

size_settings = [(200, 20), (500, 50), (1000, 100)]
size_results = []

for chunk_size, overlap in size_settings:
    print("-" * 50)
    show_splitter_info("RecursiveCharacterTextSplitter", chunk_size, overlap)
    chunks = split_documents_recursive(documents, chunk_size, overlap)
    show_result(chunks)
    show_statistics(chunks)
    print("First chunk (first 200 characters):")
    print(chunks[0].page_content[:200])
    print()
    size_results.append((chunk_size, overlap, len(chunks)))

chunks_500 = split_documents_recursive(documents, 500, 50)


print("=" * 50)
print("EXTREME CHUNK SIZES")
print("=" * 50)

print("-" * 50)
print("Very small chunks (chunk_size = 100, chunk_overlap = 0)")
show_splitter_info("RecursiveCharacterTextSplitter", 100, 0)
small_chunks = split_documents_recursive(documents, 100, 0)
show_result(small_chunks)
show_statistics(small_chunks)
print("First 3 chunks:")
for i in range(3):
    print(i, "->", repr(small_chunks[i].page_content))
print()

print("-" * 50)
print("Very large chunks (chunk_size = 5000, chunk_overlap = 0)")
show_splitter_info("RecursiveCharacterTextSplitter", 5000, 0)
large_chunks = split_documents_recursive(documents, 5000, 0)
show_result(large_chunks)
show_statistics(large_chunks)
print()
print("Tiny chunks cut sentences into pieces that are hard to understand")
print("alone. A chunk size bigger than the whole document gives just one")
print("chunk, so nothing is really split.")
print()


print("=" * 50)
print("CHUNK OVERLAP EXPERIMENT")
print("=" * 50)
print("Splitter: RecursiveCharacterTextSplitter, Chunk Size: 300")
print("(I first tried this with chunk size 500, but then overlap 50, 100")
print(" and 200 gave the same result on my document. With 300 the")
print(" differences are clearly visible.)")
print()

overlap_results = []

for overlap in [0, 50, 100, 200]:
    print("-" * 50)
    show_splitter_info("RecursiveCharacterTextSplitter", 300, overlap)
    chunks = split_documents_recursive(documents, 300, overlap)
    show_result(chunks)
    show_statistics(chunks)

    pairs_with_repeated_text = 0
    total_repeated = 0
    example_repeated_text = ""

    for i in range(len(chunks) - 1):
        text_a = chunks[i].page_content
        text_b = chunks[i + 1].page_content
        repeated = 0
        for j in range(1, 301):
            if text_a[-j:] == text_b[:j]:
                repeated = j
        if repeated >= 5:
            pairs_with_repeated_text = pairs_with_repeated_text + 1
            total_repeated = total_repeated + repeated
            if example_repeated_text == "":
                example_repeated_text = text_b[:repeated]

    print("Chunk pairs with repeated text:", pairs_with_repeated_text,
          "of", len(chunks) - 1)
    if pairs_with_repeated_text > 0:
        print("Average repeated characters per pair:",
              round(total_repeated / pairs_with_repeated_text, 1))
        print("Example of repeated text:", repr(example_repeated_text[:80]))
    print()

    overlap_results.append((overlap, len(chunks), pairs_with_repeated_text))


print("=" * 50)
print("SPLITTER COMPARISON")
print("=" * 50)

compare_size = 300
compare_overlap = 50

recursive_chunks = split_documents_recursive(documents, compare_size, compare_overlap)
character_chunks = split_documents_character(documents, compare_size, compare_overlap)

recursive_lengths = []
for chunk in recursive_chunks:
    recursive_lengths.append(len(chunk.page_content))

character_lengths = []
for chunk in character_chunks:
    character_lengths.append(len(chunk.page_content))

print(f"{'Splitter':<34} {'Chunk Size':>11} {'Overlap':>8} {'Chunks':>8}")
print("-" * 65)
print(f"{'RecursiveCharacterTextSplitter':<34} {compare_size:>11} {compare_overlap:>8} {len(recursive_chunks):>8}")
print(f"{'CharacterTextSplitter':<34} {compare_size:>11} {compare_overlap:>8} {len(character_chunks):>8}")
print()

recursive_too_big = 0
for length in recursive_lengths:
    if length > compare_size:
        recursive_too_big = recursive_too_big + 1

character_too_big = 0
for length in character_lengths:
    if length > compare_size:
        character_too_big = character_too_big + 1

print("RecursiveCharacterTextSplitter: longest chunk =", max(recursive_lengths),
      "| chunks bigger than chunk_size =", recursive_too_big)
print("CharacterTextSplitter:          longest chunk =", max(character_lengths),
      "| chunks bigger than chunk_size =", character_too_big)
print()
print("Explanation:")
print("RecursiveCharacterTextSplitter can use several separators (paragraph,")
print("line, word), so it always finds a place to cut and every chunk stays")
print("inside the chunk size. CharacterTextSplitter only looks for one")
print("separator (\\n\\n). If no blank line exists inside the chunk size it")
print("cannot cut the text, so the chunk becomes bigger than chunk_size.")
print("(LangChain even printed warnings like 'Created a chunk of size 455,")
print("which is longer than the specified 300'.)")
print("This is why the recursive splitter is usually the better choice.")


print()
print("=" * 50)
print("METADATA INSPECTION")
print("=" * 50)

print("Metadata of the original Document:")
print(documents[0].metadata)
print()
print("Metadata of chunk 0 after splitting:")
print(chunks_500[0].metadata)
print()
print("The original metadata is carried to the chunks, so we can identify")
print("where the chunk came from. Only new keys are added, for example")
print("start_index (the position of the chunk inside the original text).")


print()
print("=" * 50)
print("CHUNK INSPECTION")
print("=" * 50)

show_splitter_info("RecursiveCharacterTextSplitter", 500, 50)
show_result(chunks_500)
print()
show_chunk(chunks_500, 0)
print()
show_chunk(chunks_500, 1)
print()
print("Statistics:")
show_statistics(chunks_500)
print()
print("Why does a chunk start and end here? The splitter first tries to cut")
print("at a paragraph (\\n\\n), then at a line (\\n), then at a space, and it")
print("stops at the last place that still fits inside the chunk size.")


print()
print("=" * 50)
print("ERROR HANDLING")
print("=" * 50)

print()
print("Test 1: chunk_size = 500 with chunk_overlap = 600")
try:
    split_documents_recursive(documents, 500, 600)
    print("Result: Success (no error)")
except ValueError as e:
    print("Result: Failed with an error")
    print("Error message:", e)

print()

print("Test 2: loading a TXT file that does not exist")
try:
    load_txt_file("data/this_file_does_not_exist.txt")
    print("Result: Success (no error)")
except RuntimeError as e:
    print("Result: Failed with an error")
    print("Error message:", e)


print()
print("=" * 50)
print("OBSERVATIONS")
print("=" * 50)

print("1. Chunk size experiment:")
for size, ov, count in size_results:
    print(f"   chunk_size = {size:>4}, overlap = {ov:>3}  ->  {count:>3} chunks")
print("   chunk_size = 100, overlap = 0     ->", len(small_chunks), "chunks")
print("   chunk_size = 5000, overlap = 0    ->", len(large_chunks), "chunks")
print("   A bigger chunk size means fewer chunks.")

print()
print("2. Chunk overlap experiment (chunk size = 300):")
for ov, count, pairs in overlap_results:
    print(f"   overlap = {ov:>3}  ->  {count:>3} chunks, repeated text in {pairs:>2} of {count - 1:>2} chunk pairs")
print("   A bigger overlap means more repeated text, and a little more chunks.")
print("   The repeated text keeps a sentence understandable in both chunks,")
print("   but it also uses more space.")

print()
print("3. Splitters: the recursive splitter always stayed inside the chunk")
print("   size, the character splitter produced chunks bigger than chunk_size.")
print("   Both splitters kept the metadata of the original document.")

print()
print("4. Next steps in a real RAG system:")
print("   Document -> Chunk -> Embedding -> Vector Database -> Retrieval -> LLM")