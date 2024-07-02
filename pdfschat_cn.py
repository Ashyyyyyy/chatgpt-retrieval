import constants
import os, sys

# main dependencies including spliter, Embedder, VectorStore, LLM
from langchain import hub
# from langchain_openai import ChatOpenAI
import dashscope
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Chat history dependencies
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

# Utilities
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# VectorStore Caching
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

# llm setup
os.environ["OPENAI_API_KEY"] = constants.APIKEY
os.environ["DASHSCOPE_API_KEY"] = constants.Qwen_APIKEY
os.environ['QIANFAN_ACCESS_KEY'] = constants.Qianfan_AK
os.environ['QIANFAN_SECRET_KEY'] = constants.Qianfan_SK
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = ChatTongyi(model="qwen-turbo")


# Setup VectorStore (from raw documents or from cache)
store = LocalFileStore("./cache/")
embedder = QianfanEmbeddingsEndpoint(
    qianfan_ak=constants.Qianfan_AK,
    qianfan_sk=constants.Qianfan_SK
)
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embedder, document_embedding_cache=store, namespace=embedder.model
)
# loader = DirectoryLoader("data/").load()
# loader = TextLoader("data/data.txt").load()
loader = PyPDFDirectoryLoader("data/").load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    # separators=["\n\n", "\n", " ", "", "。", "，"]
)
splits = text_splitter.split_documents(loader)
# splits = text_splitter.transform_documents(loader)


vectorstore = Chroma.from_documents(documents=splits, embedding=cached_embedder)
# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
# vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Add chat history
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=contextualize_q_prompt
)

# Make QA chain
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you don't know. "
    "Don‘t make up answers. "
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Save history and processing
chat_history = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_history:
        chat_history[session_id] = ChatMessageHistory()
    return chat_history[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# get input recursively
query = None
if (len(sys.argv) > 1):
    query = sys.argv[1]
while True:
    if not query:
        query = input("User: ")
    if query in ['q', 'quit', 'exit']:
        # sys.exit()
        break
    result = conversational_rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": "user1"}},
    )
    sources = set()
    for _result in result["context"]:
        sources.add(_result.metadata['source'])
    # print(result)
    print(result["answer"])
    print('Source: ', sources)
    # print(chat_history)
    query = None

# from langchain_core.messages import AIMessage, HumanMessage
#
# chat_history = []
#
# question = "What is my cat's name?"
# ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
# print(ai_msg_1)
# chat_history.extend(
#     [
#         HumanMessage(content=question),
#         AIMessage(content=ai_msg_1["answer"]),
#     ]
# )
# print(ai_msg_1)
# second_question = "What is my dog's name?"
# ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})
# print(ai_msg_2)
#
# third_question = "How many pets do i have?"
# ai_msg_3 = rag_chain.invoke({"input": third_question, "chat_history": chat_history})
# print(ai_msg_3)
