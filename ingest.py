import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# DATA FOLDER

data_folder = "data"

print("=" * 50)
print("Files found in data folder:")
print("=" * 50)

files = os.listdir(data_folder)

for file in files:
    print(file)

# LOAD PDF FILES

documents = []

print("\nLoading PDF files...\n")

for file in files:

    if file.endswith(".pdf"):

        filepath = os.path.join(data_folder, file)

        print(f"Loading: {file}")

        try:

            loader = PyPDFLoader(filepath)

            docs = loader.load()

            documents.extend(docs)

            print(f"Loaded Successfully: {file}")

        except Exception as e:

            print(f"Error loading {file}")
            print(e)

# CHECK DOCUMENTS

print("\n" + "=" * 50)
print("Total Documents Loaded:", len(documents))
print("=" * 50)

if len(documents) == 0:

    print("\nNo documents loaded!")
    print("Your PDFs may be invalid.")

    exit()

# SPLIT DOCUMENTS

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = splitter.split_documents(documents)

print(f"\nTotal Chunks Created: {len(docs)}")

# EMBEDDING MODEL

print("\nLoading Embedding Model...\n")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# CREATE VECTOR DATABASE

print("\nCreating FAISS Vector Database...\n")

vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

# SAVE VECTORSTORE

vectorstore.save_local("vectorstore")

print("\nSUCCESS!")
print("Vector Database Created Successfully!")
print("Saved in 'vectorstore/' folder")