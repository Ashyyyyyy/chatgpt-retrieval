import constants
import os, sys
import json

# main dependencies including spliter, Embedder, VectorStore, LLM
from langchain import hub
from langchain_openai import ChatOpenAI
import dashscope
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import QianfanEmbeddingsEndpoint, DashScopeEmbeddings
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
from langchain.indexes import SQLRecordManager, index

# llm setup
os.environ["OPENAI_API_KEY"] = constants.APIKEY
os.environ["DASHSCOPE_API_KEY"] = constants.Qwen_APIKEY
os.environ['QIANFAN_ACCESS_KEY'] = constants.Qianfan_AK
os.environ['QIANFAN_SECRET_KEY'] = constants.Qianfan_SK
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = ChatTongyi(model="qwen-turbo")

PERSIST_path = './persist'
data_dictionary_path = './file_name_dictionary'
data_dictionary = './file_name_dictionary/logs.json'
data_folder = './data'

score_threshold = 0.52
k = 12

# 获取文件夹中所有文件的完整路径
def get_all_files(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        print(f"{root} ------ {dirs} ------{files}")
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), folder_path)
            file_list.append(os.path.join(os.path.basename(folder_path), relative_path))
    return sorted(file_list)


# 读取字典文件
def read_dictionary(file_path):
    with open(file_path, 'r') as f:
        return sorted(json.load(f))


# 写入字典文件
def write_dictionary(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def splitter():
    loader = DirectoryLoader("data/", show_progress=True, use_multithreading=True).load()
    # loader = TextLoader("data/data.txt").load()
    # loader = PyPDFDirectoryLoader("data/").load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=len,
        # separators=["\n\n", "\n", " ", "", "。", "，"]
    )
    splits = text_splitter.split_documents(loader)
    # splits = text_splitter.transform_documents(loader)
    return splits


def vectorstore_init_or_update(PERSIST_path, embedder, cleanup):
    # vectorstore = Chroma.from_documents(documents=splits, embedding=cached_embedder, persist_directory=PERSIST_path)
    vectorstore = Chroma(persist_directory=PERSIST_path, embedding_function=embedder, collection_metadata={"hnsw:space": "cosine"})
    record_manager = SQLRecordManager(PERSIST_path, db_url="sqlite:///record_manager_cache.sql")
    record_manager.create_schema()
    # vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    # vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    tmp = index(
        splits,
        record_manager,
        vectorstore,
        cleanup=cleanup,
        source_id_key="source",
    )
    print(tmp)
    return vectorstore


# Embedder setup
store = LocalFileStore("./embeddings_cache/")
# embedder = QianfanEmbeddingsEndpoint(
#     qianfan_ak=constants.Qianfan_AK,
#     qianfan_sk=constants.Qianfan_SK
# )
# embedder = OpenAIEmbeddings()
embedder = DashScopeEmbeddings(model="text-embedding-v2")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embedder, document_embedding_cache=store, namespace=embedder.model
)

# Modify retriever function
import langchain_core.vectorstores
def new_get_relevant_documents(
        self, query: str, *, run_manager):
    if self.search_type == "similarity":
        docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
    elif self.search_type == "similarity_score_threshold":
        docs_and_similarities = (
            self.vectorstore.similarity_search_with_relevance_scores(
                query, **self.search_kwargs
            )
        )
        docs = []
        for doc, score in docs_and_similarities:
            doc.metadata['score'] = score
            docs.append(doc)
        # docs = docs_and_similarities
    elif self.search_type == "mmr":
        docs = self.vectorstore.max_marginal_relevance_search(
            query, **self.search_kwargs
        )
    else:
        raise ValueError(f"search_type of {self.search_type} not allowed.")
    return docs

langchain_core.vectorstores.VectorStoreRetriever._get_relevant_documents = new_get_relevant_documents

# Modify similarity_search_with_relevance_scores in similarity_score_threshold
from typing import Any, List, Tuple
import warnings
from langchain_core.documents.base import Document
def new_similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
) -> List[Tuple[Document, float]]:
    """Return docs and relevance scores in the range [0, 1].

    0 is dissimilar, 1 is most similar.

    Args:
        query: Input text.
        k: Number of Documents to return. Defaults to 4.
        **kwargs: kwargs to be passed to similarity search. Should include:
            score_threshold: Optional, a floating point value between 0 to 1 to
                filter the resulting set of retrieved docs

    Returns:
        List of Tuples of (doc, similarity_score)
    """
    score_threshold = kwargs.pop("score_threshold", None)

    docs_and_similarities = self._similarity_search_with_relevance_scores(
        query, k=k, **kwargs
    )
    if any(
            similarity < 0.0 or similarity > 1.0
            for _, similarity in docs_and_similarities
    ):
        warnings.warn(
            "Relevance scores must be between"
            f" 0 and 1, got {docs_and_similarities}"
        )

    if score_threshold is not None:
        docs_and_similarities.sort(key=lambda x: x[1], reverse=True)
        if docs_and_similarities[0][1] > 0.8:
            new_score_threshold = score_threshold + 0.1
        else:
            new_score_threshold = score_threshold

        docs_and_similarities = [
            (doc, similarity)
            for doc, similarity in docs_and_similarities
            if similarity >= new_score_threshold
        ]
        # if len(docs_and_similarities) == 0:
        #     warnings.warn(
        #         "No relevant docs were retrieved using the relevance score"
        #         f" threshold {score_threshold}"
        #     )
    return docs_and_similarities


langchain_core.vectorstores.VectorStore.similarity_search_with_relevance_scores = new_similarity_search_with_relevance_scores

# Retriever setup
if not os.path.exists(PERSIST_path) or not os.path.exists(data_dictionary):
    print("------ No dictionary and vectorstore.------")
    if not os.path.exists(PERSIST_path):
        os.makedirs(PERSIST_path)
        print(f"Directory {PERSIST_path} created.")
    if not os.path.exists(data_dictionary_path):
        os.makedirs(data_dictionary_path)
        print(f"Directory {data_dictionary_path} created.")

    # Initialize Vectorstore
    print("Initializing: Load vectorstore from raw documents...")

    splits = splitter()

    vectorstore = vectorstore_init_or_update(PERSIST_path, cached_embedder, "full")
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": k, "score_threshold": score_threshold})
    # retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
    print(
        f"Initialize embedded docs and vectorstore completed. Next time the vactorstore can be load from {PERSIST_path}")

    # Read all full path and file name in ./data
    current_files = get_all_files(data_folder)
    print(current_files)
    write_dictionary(data_dictionary, current_files)
    print("------ Retriever Initializing Complete! Dictionary, persist vectorstore created.------")
else:
    # Load from persist vectorstore.
    # Check changes in ./data to see if vectorstore needed to be updated.
    print(f"------ Directory {PERSIST_path} already exists. ------")
    print('Load from persist vectorstore.')
    print('Checking changes in ./data to see if vectorstore needed to be updated...')
    current_files = get_all_files(data_folder)
    recorded_files = read_dictionary(data_dictionary)
    # Compare current files and recorded files.
    if current_files == recorded_files:
        print("No update detected. Use persist vectorstore directly...")
        vectorstore = Chroma(persist_directory=PERSIST_path, embedding_function=cached_embedder, collection_metadata={"hnsw:space": "cosine"})
        # retriever = vectorstore.as_retriever( search_kwargs={'k': 5})
        retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",
                                             search_kwargs={"k": k, "score_threshold": score_threshold})

        print("------ Retriever Initializing Complete! Load from persist vectorstore. ------")
    else:
        print("Updating vectorstore...")
        splits = splitter()
        vectorstore = vectorstore_init_or_update(PERSIST_path, cached_embedder, "full")
        # retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
        retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",
                                             search_kwargs={"k": k, "score_threshold": score_threshold})

        print("Vectorstore updated! ")
        current_files = get_all_files(data_folder)
        print(current_files)
        write_dictionary(data_dictionary, current_files)
        print("------ Retriever Initializing Complete! Dictionary, persist vectorstore updated. ------")

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
    "the question. If you don't get the related information, say that you don't know. "
    "DON'T make up answers."
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
# query = None
# if (len(sys.argv) > 1):
#     query = sys.argv[1]
# while True:
#     if not query:
#         query = input("User: ")
#     if query in ['q', 'quit', 'exit']:
#         # sys.exit()
#         break
#     result = conversational_rag_chain.invoke(
#         {"input": query},
#         config={"configurable": {"session_id": "user1"}},
#     )
#     sources = set()
#     for _result in result["context"]:
#         sources.add(_result.metadata['source'])
#     # print(result)
#     print(result["answer"])
#     print('Source: ', sources)
#     # print(chat_history)
#     query = None

def process_input(query: str, session_id: str):
    result = conversational_rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}},
        )
    sources = set()
    for _result in result["context"]:
        sources.add((_result.metadata['source'], _result.metadata['score']))
    # print(result)
    print(result["answer"])
    print('Source: ', sources)
    print(chat_history)
    return result

# result = process_input("Hello", "user2")
