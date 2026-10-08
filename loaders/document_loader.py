from langchain_community.document_loaders import TextLoader, PyPDFLoader


def load_txt_file(file_path):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    return documents


def load_pdf_file(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


def load_md_file(file_path):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    return documents