import getpass
import os
import constants

from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = constants.APIKEY
# llm = ChatOpenAI(model="gpt-3.5-turbo")
os.environ["DASHSCOPE_API_KEY"] = constants.Qwen_APIKEY
os.environ['QIANFAN_ACCESS_KEY'] = constants.Qianfan_AK
os.environ['QIANFAN_SECRET_KEY'] = constants.Qianfan_SK


import bs4
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import QianfanEmbeddingsEndpoint

llm = ChatTongyi(model="qwen-turbo")
embedder = QianfanEmbeddingsEndpoint(
    qianfan_ak=constants.Qianfan_AK,
    qianfan_sk=constants.Qianfan_SK
)
# 1. Load, chunk and index the contents of the blog to create a retriever.
loader = PyPDFDirectoryLoader("data/")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embedder)
retriever = vectorstore.as_retriever()


# 2. Incorporate the retriever into a question-answering chain.
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
result = rag_chain.invoke({"input": "What is Task Decomposition?"})