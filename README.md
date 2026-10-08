# RAG - Document Loading and Text Splitting

This project shows how a raw file becomes a LangChain **Document**, and how one
big Document is cut into small **chunks**. This is the middle part of a RAG
pipeline.

```
Document  ->  Document Loader  ->  Document (page_content + metadata)
           ->  Text Splitter    ->  Chunks
```

## How to run

```bash
pip install -r requirements.txt
python main.py
```

The program prints every experiment with clear headings:

```
==================================================
DOCUMENT LOADING
==================================================
...
```

The sections printed by `main.py` are: Document Loading, Chunk Size Experiment,
Extreme Chunk Sizes, Chunk Overlap Experiment, Splitter Comparison, Metadata
Inspection, Chunk Inspection, Error Handling and Observations.

## Project structure

```
document_chunking_task/
│
├── data/
│   ├── sample.pdf
│   ├── sample.txt
│   └── sample.md
│
├── loaders/
│   └── document_loader.py
│
├── splitters/
│   ├── recursive_splitter.py
│   └── other_splitter.py
│
├── utils/
│   └── chunk_inspector.py
│
├── main.py
├── README.md
└── requirements.txt
```

---

## 1. Objective

The purpose of this assignment was to understand what happens between loading a
document and sending chunks to an embedding model.

I wanted to be able to answer these questions by running my own program:

- What does a document loader actually produce?
- Why does a splitter give me this number of chunks?
- Why is some text repeated between two chunks?
- What happens to the metadata after splitting?

The program demonstrates two loaders (TextLoader and PyPDFLoader),
`RecursiveCharacterTextSplitter`, `CharacterTextSplitter` as the second
splitter, different chunk sizes (including a very small and a very large one),
different chunk overlaps, metadata inspection, chunk inspection, statistics and
error handling.

---

## 2. Document Loader

I put the loading functions in `loaders/document_loader.py`. Each function is
very small: it creates a loader and calls `.load()`.

```python
def load_txt_file(file_path):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    return documents
```

`.load()` returns a list of LangChain Document objects. Every Document has:

- `page_content` - the text
- `metadata` - a dictionary with information about where the text came from

Results of my run:

| File | Loader | Number of Documents | Original Length |
|------|--------|--------------------:|----------------:|
| `sample.txt` | TextLoader | 1 | 4,757 characters |
| `sample.pdf` | PyPDFLoader | 2 | 1,805 characters |
| `sample.md` | TextLoader | 1 | 999 characters |

Two things I noticed:

1. **A PDF is handled page by page.** `PyPDFLoader` gave me 2 Documents for a
   2-page PDF, and the metadata contains `page`, `page_label` and
   `total_pages`. The TXT file became one single Document.
2. **Markdown is plain text**, so `TextLoader` can read it directly. I did not
   install `unstructured` just for one file.

The metadata of the TXT file was simply `{'source': 'data/sample.txt'}` - the
loader puts the file path there, and that is what lets us later say "this
answer came from this file".

---

## 3. Text Splitter

The main splitter is `RecursiveCharacterTextSplitter`
(`splitters/recursive_splitter.py`). The function takes the documents, a chunk
size and an overlap and returns the chunks:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", " ", ""],
    add_start_index=True,
)
chunks = splitter.split_documents(documents)
```

Two points I want to remember:

- **"Recursive"** means it tries the split points in order:
  paragraph (`\n\n`) -> line (`\n`) -> space -> character. It uses the first one
  that works, so a chunk normally ends at a paragraph, a line or a word, not in
  the middle of a word.
- I used **`split_documents()`** instead of `split_text()` because it keeps the
  metadata of the original Document on every chunk.

`add_start_index=True` adds `metadata["start_index"]` to every chunk, which
tells me the position of the chunk inside the original text.

---

## 4. Chunk Size Experiment

Settings: `RecursiveCharacterTextSplitter`, main document `sample.txt`
(4,757 characters). Each chunk size is paired with its own overlap:

| chunk_size | chunk_overlap | Number of Chunks | Average | Minimum | Maximum |
|-----------:|--------------:|-----------------:|--------:|--------:|--------:|
| 200 | 20 | 41 | 121.6 | 18 | 199 |
| 500 | 50 | 12 | 415.2 | 354 | 482 |
| 1000 | 100 | 6 | 817.2 | 722 | 953 |

What happened:

- A bigger chunk size means **fewer chunks** (41 -> 12 -> 6).
- The maximum length is always **below** the chunk size (199, 482, 953), and the
  average is lower still, because the splitter stops at a paragraph/line/word
  that still fits instead of filling the chunk exactly to the limit.
- With chunk size 200 the shortest chunk is only 18 characters, and the very
  first chunk is just the title `CHAPTER 1 / INTRODUCTION TO MACHINE LEARNING`
  with no explanation text. Very small chunks carry almost no meaning by
  themselves, which would be bad for retrieval.
- With chunk size 1000 each chunk holds a long piece of the text (722 to 953
  characters), so there is more context per chunk but also more topics mixed
  together.

### Extreme chunk sizes

I also tried two extreme settings on the same document to see what goes wrong.

| chunk_size | chunk_overlap | Number of Chunks | Average | Minimum | Maximum |
|-----------:|--------------:|-----------------:|--------:|--------:|--------:|
| 100 | 0 | 63 | 74.1 | 9 | 99 |
| 5000 | 0 | 1 | 4756.0 | 4756 | 4756 |

**Chunk size 100** - the first three chunks were:

```
0 -> 'CHAPTER 1\nINTRODUCTION TO MACHINE LEARNING'
1 -> 'Machine learning is a branch of artificial intelligence that gives computers the ability to learn'
2 -> 'from data without being explicitly programmed for every situation. Instead of writing a fixed set'
```

Chunk 1 ends with "...the ability to learn" and the sentence continues in
chunk 2 with "from data...". So one sentence is cut into two pieces, and neither
chunk is understandable alone. With overlap 0 nothing connects the two halves.
These chunks would be poor for retrieval.

**Chunk size 5000** - the whole document became **one chunk**, because the
chunk size is larger than the document. Its length is 4,756 characters, one
less than the 4,757 loaded, because the splitter strips the whitespace at the
end. Nothing is really split here: a retriever could only return the entire
document, together with all unrelated topics, and an LLM would have to read
all of it for every question.

So there is no single correct chunk size - it is a balance between too little
context and too much mixed context.

---

## 5. Chunk Overlap Experiment

Settings: `RecursiveCharacterTextSplitter`, chunk size fixed at 300.

I first tried this experiment with chunk size 500 (as suggested), but on my
document the results for overlap 50, 100 and 200 were identical, so I used 300
where the differences are clearly visible.

| chunk_overlap | Number of Chunks | Average | Chunk pairs with repeated text | Average repeated characters |
|--------------:|-----------------:|--------:|-------------------------------:|----------------------------:|
| 0 | 34 | 138.2 | 0 of 33 | 0 |
| 50 | 34 | 151.7 | 10 of 33 | 45.1 |
| 100 | 34 | 166.8 | 10 of 33 | 96.4 |
| 200 | 38 | 196.0 | 14 of 37 | 195.7 |

Why is some text repeated between two chunks? Because of the overlap: the end
of chunk N is copied to the beginning of chunk N+1. My program measures this by
comparing the end of one chunk with the start of the next chunk. Example from
the run (overlap = 50):

```
Example of repeated text: 'to write down, such as recognising a face in a'
```

What happened:

- With overlap 0 there is **no repeated text**, so a sentence that crosses the
  cut is split in half and neither chunk contains the full sentence.
- With overlap 200 about 196 characters are repeated, so the bridging
  sentence exists in both chunks and can be found by retrieval either way.
- A bigger overlap gives a few more chunks (34 -> 38) and more average length
  (138.2 -> 196.0), because the same text is stored twice. Overlap is not free.
- The number of chunks stayed at 34 for overlaps 0, 50 and 100. The splitter
  prefers to cut at paragraph breaks, so a small overlap often has no short
  piece to carry over. Only at overlap 200 did the count change.
- Only 10 of 33 pairs shared text at overlap 50 and 100, because the overlap
  is used only when the text before the cut is made of small pieces that fit
  inside the overlap size. The measured repetition (e.g. 45 instead of 50) is a
  bit smaller than the setting for the same reason.

---

## 6. Splitter Comparison

Same document, same settings for both: chunk size 300, overlap 50.

| Splitter | Chunk Size | Overlap | Chunks | Longest chunk | Chunks bigger than chunk_size |
|----------|-----------:|--------:|-------:|--------------:|------------------------------:|
| RecursiveCharacterTextSplitter | 300 | 50 | 34 | 299 | 0 |
| CharacterTextSplitter | 300 | 50 | 24 | 457 | 10 |

`CharacterTextSplitter` only knows **one** separator (`\n\n`). My document has
paragraphs that are longer than 300 characters, so in those places it cannot
cut the text and the chunk grows past the limit. LangChain itself printed
warnings such as:

```
Created a chunk of size 455, which is longer than the specified 300
```

`RecursiveCharacterTextSplitter` fell back to `\n` and then to a space, so it
always found a split point and the longest chunk was 299 characters - inside
the limit.

**Which one would I choose?** For normal text: the recursive splitter, because
it respects the chunk size and does not break words. `CharacterTextSplitter` is
only safe when you know the separator appears very often, for example in log
files or CSV-like text.

---

## 7. Metadata and Chunk Inspection

I printed the metadata of the original Document and of a chunk after
splitting:

```
Metadata of the original Document:
{'source': 'data/sample.txt'}

Metadata of chunk 0 after splitting:
{'source': 'data/sample.txt', 'start_index': 0}
```

The original metadata is carried to the chunks, so we can identify where the
chunk came from. Nothing was removed; the splitter only added new keys (here
`start_index`). For the PDF the same thing happens with `page`, so later a RAG
system could say "this answer came from page 1 of sample.pdf".

I used `split_documents()` for exactly this reason - if I had used
`split_text()`, I would only get strings and the metadata would be lost.

### Looking at two real chunks (chunk size 500, overlap 50)

| Chunk | Length | start_index | Starts with | Ends with |
|------:|-------:|------------:|-------------|-----------|
| 0 | 354 | 0 | `CHAPTER 1` | `Why machine learning matters` |
| 1 | 396 | 326 | `Why machine learning matters` | `Types of machine learning` |

The line `Why machine learning matters` appears at the end of chunk 0 and at the
start of chunk 1 - that is the overlap. The numbers agree: chunk 0 is 354
characters long and chunk 1 starts at position 326, so 354 - 326 = 28 characters
are shared, which is exactly the length of that heading. The repeated part is
28 characters and not 50, because the splitter cuts at the paragraph break and
only repeats pieces that fit inside the overlap.

This also answers "why does a chunk start and end here?": the splitter cuts at
the last paragraph break (`\n\n`) that still fits inside the chunk size, and
then starts the next chunk with whatever small piece at the end fits inside the
overlap.

---

## 8. Error Handling

I included two simple `try/except` tests in `main.py`.

**Test 1: invalid overlap** - overlap must be smaller than the chunk size,
otherwise the splitter would never finish.

```
Test 1: chunk_size = 500 with chunk_overlap = 600
Result: Failed with an error
Error message: chunk_overlap (600) must be smaller than chunk_size (500)
```

The check is a small `if` inside `split_documents_recursive()` that raises a
`ValueError`.

**Test 2: a file that does not exist**

```
Test 2: loading a TXT file that does not exist
Result: Failed with an error
Error message: Error loading data/this_file_does_not_exist.txt
```

Here LangChain's `TextLoader` raises the error for me (wrapped in a
`RuntimeError`), so I only catch it and print the message. The program then
continues with the next experiment instead of crashing.

---

## 9. RAG Connection

This assignment stops at the chunk stage. The full pipeline looks like this:

```
Document
   -> Chunk                (this assignment)
   -> Embedding            each chunk is sent to an embedding model and
                           becomes a list of numbers that represents meaning
   -> Vector Database      the vectors are stored in Chroma / FAISS / Pinecone
   -> Retrieval            the question is embedded too, and the database
                           returns the chunks closest to the question
   -> LLM                  those chunks are put into the prompt, and the model
                           writes the answer from that evidence
```

Why chunking matters for the rest:

- Chunks that are **too big** mix several topics into one vector, so the
  similarity score becomes weak. My 5000-size run is the extreme case: one
  vector for the whole document.
- Chunks that are **too small** lose the sentence around them, so the vector is
  unclear. In my size-100 run one sentence was split across two chunks.
- **No overlap** means a fact that crosses a cut exists fully in no chunk, and
  retrieval can miss it.

The metadata that stays on every chunk is what later allows the answer to cite
its source file and page.

---

## 10. My Observations

What I personally understood from running these experiments:

1. **The loader decides the shape of the data.** One TXT file gave me a single
   Document, while the same kind of content in a PDF gave me one Document per
   page. Everything after the loader inherits that structure.

2. **Chunk size is the strongest setting.** Going from 200 to 1000 turned 41
   chunks into 6. Small chunks give many pieces to retrieve but little context,
   big chunks keep the context but mix topics together. 500 (12 chunks) felt
   like a reasonable middle ground for my file.

3. **The extreme sizes both fail, in different ways.** Size 100 gave 63 chunks
   and cut a sentence in half. Size 5000 gave 1 chunk for the whole document, so
   nothing was split at all. Neither would be useful for retrieval.

4. **Overlap really repeats text, and it costs space.** I could see the same
   heading at the end of one chunk and the start of the next one, and I could
   confirm it with the numbers (354 - 326 = 28 shared characters). Moving from
   overlap 0 to 200 changed the average chunk length from 138.2 to 196.0 - so I
   would use something like 10-20% of the chunk size, not more.

5. **The splitter choice is not a detail.** The character splitter ignored my
   chunk size and gave me 457-character chunks while I asked for 300, and
   LangChain warned me about it. The recursive splitter never exceeded the
   limit. That made it clear why everybody recommends the recursive one.

6. **Metadata survives the split.** `source` is still there on every chunk and
   only new keys were added. This small fact is what makes "the answer came
   from sample.txt" possible later.

7. **Chunk boundaries decide what the model can know.** If the answer to a
   question is spread across two chunks and there is no overlap, the RAG system
   may never see the complete answer. Splitting looks boring, but it quietly
   defines the quality of the final answer.