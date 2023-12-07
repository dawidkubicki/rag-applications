from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Neo4jVector
from langchain.embeddings import HuggingFaceBgeEmbeddings


# Loading document and splitting into the chunks
pdfLoader = PyMuPDFLoader('../documents/apple-10-k.pdf')
documents = pdfLoader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

# Neo4j Aura credentials
url=""
username=""
password=""

# The Neo4jVector Module will connect to Neo4j and create a vector index if needed.

neo4j_db = Neo4jVector.from_documents(
    docs, HuggingFaceBgeEmbeddings(), url=url, username=username, password=password
)

query = "What are Legal Proceedings?"

results = neo4j_db.similarity_search(query, k=1)
print(results[0].page_content)