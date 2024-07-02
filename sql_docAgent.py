import constants
import os, sys

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.checkpoint.sqlite import SqliteSaver

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.agent_toolkits import SQLDatabaseToolkit


# llm setup
os.environ["OPENAI_API_KEY"] = constants.APIKEY
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# database connection setup
db = SQLDatabase.from_uri("sqlite:////Users/henrisun/Documents/RAG/chatgpt-retrieval/db/test.db")
print(db.dialect)
print(db.get_usable_table_names())
# db.run("SELECT * FROM Artist LIMIT 10;")


# init documents
loader = DirectoryLoader("data/").load()
# loader = PyPDFDirectoryLoader("data2/").load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
splits = text_splitter.split_documents(loader)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
# vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


# Build document retriever tool
document_tool = create_retriever_tool(
    retriever=retriever,
    name="document_retriever",
    description="Searches and returns information from documents.",
)
tools = [document_tool]


# Chat history
memory = SqliteSaver.from_conn_string(":memory:")


# Build SQL tool
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools.extend(toolkit.get_tools())

# Create Agent
system_message = '''你是一个可以与SQL数据库和文档检索器交互的智能代理。根据输入的问题，创建一个语法正确的SQLite查询以运行，或使用文档检索器检索相关信息，然后查看查询结果或检索结果并返回答案。

如果问题涉及结构化数据（如特定表格中的数据），请首先查看数据库中的表。

生成查询前，务必查看相关表的架构。
创建查询时，仅查询与问题相关的列，限制结果数不超过5条（除非用户指定具体数量）。
按相关列排序以返回最有趣的示例。
执行查询时，如果出现错误，请重新编写查询并再次尝试。
如果问题涉及非结构化数据（如文档中的信息），请使用文档检索器。

从向量存储中检索与问题相关的文档片段。
返回最相关的片段内容以回答问题。
不得对数据库进行任何DML操作（INSERT、UPDATE、DELETE、DROP等）。

在开始查询或检索之前，必须确定使用哪种工具（SQL工具或文档检索器）或不使用工具。如果找不到答案回答我不知道。'''
agent_executor = create_react_agent(llm, tools, checkpointer=memory, messages_modifier=system_message)


# Run agent
config = {"configurable": {"thread_id": "abc123"}}
result = agent_executor.invoke({"messages": [HumanMessage(content="Bob的email是多少？")]}, config=config)
print(result['messages'][-1].content)
result = agent_executor.invoke({"messages": [HumanMessage(content="《水利数据交换规约》的标准号是多少")]}, config=config)
print(result['messages'][-1].content)

result = agent_executor.invoke({"messages": [HumanMessage(content="我长什么样")]}, config=config)
print(result['messages'][-1].content)