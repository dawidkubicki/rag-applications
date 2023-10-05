from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA 


# Loading document and splitting into the chunks
pdfLoader = PyMuPDFLoader('../documents/tesla-10-K.pdf')
documents = pdfLoader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

# Loading Embeddings
modelPath = "all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name = modelPath,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs,
)

# Store vector data (embeddings)
db = FAISS.from_documents(docs, embeddings)
question = "What are he legal proceedings particularly in 2020?"
searchDocs = db.similarity_search(question)
# print(searchDocs[0].page_content)

# LLM
llmModelPath = "flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(llmModelPath)
model = AutoModelForSeq2SeqLM.from_pretrained(llmModelPath)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(
    pipeline=pipeline,
    model_kwargs={"temperature": 0, "max_length": 512},
)

# Prompt template
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
result = qa_chain({"query" : question})
print(result["result"])