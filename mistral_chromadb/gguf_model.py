
from langchain import PromptTemplate
from langchain import HuggingFacePipeline

from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import Chroma

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredMarkdownLoader, UnstructuredURLLoader
from langchain.chains import LLMChain, SimpleSequentialChain, RetrievalQA, ConversationalRetrievalChain

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA


pdfLoader = PyMuPDFLoader('../documents/apple-10-k.pdf')
documents = pdfLoader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
texts_chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                   model_kwargs={'device': 'cpu'})

db = Chroma.from_documents(texts_chunks, embeddings, persist_directory="db_gguf")

qa_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

llm = CTransformers(model='../models/mistral-7b-instruct-v0.1.Q6_K.gguf', # Location of downloaded GGML model
                    config={'max_new_tokens': 500,
                            'temperature': 0.01})

template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer regarding user question.
Helpful answer:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)


query = "Tell me about general risks, especially legal proceedings. Is there any of them, if yes, use format <company_name>: what legal proceeding. At the end act like a fundemental analysis expert, and tell what is says about company."
result_ = qa_chain(
    query
)
result = result_["result"].strip()


print(f"<b>{query}</b>")
print(f"<p>{result}</p>")