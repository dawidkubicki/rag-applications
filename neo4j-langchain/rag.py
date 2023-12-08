from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceBgeEmbeddings

# from langchain.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA 



# Loading document and splitting into the chunks
pdfLoader = PyMuPDFLoader('../documents/apple-10-k.pdf')
documents = pdfLoader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

url = "neo4j+s://09be3749.databases.neo4j.io"
username ="neo4j"
password = "GYUP4LmO6Dnddh-K42HCNlpsHhXlCY8GwP60Fk6s0_U"

vector_index = Neo4jVector.from_documents(
    embedding=HuggingFaceBgeEmbeddings(),
    documents=docs,
    url=url,
    username=username,
    password=password,
)

# vector_index = Neo4jVector.from_existing_graph(
#     HuggingFaceBgeEmbeddings(),
#     url=url,
#     username=username,
#     password=password,
#     index_name='chunks',
#     node_label="Chunk",
#     text_node_properties=['embedding'],
#     embedding_node_property='embedding',

# )
question = "What are current Legal Proceedings?"
response = vector_index.similarity_search(question)
