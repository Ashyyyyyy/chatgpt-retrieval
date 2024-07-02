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
db = SQLDatabase.from_uri("sqlite:////Users/henrisun/Documents/RAG/chatgpt-retrieval/db/Config.db3")
print(db.dialect)
print(db.get_usable_table_names())
# db.run("SELECT * FROM Artist LIMIT 10;")

# Build SQL tool
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()


# Build agent
prompt = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools below for interacting with the database.
If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables.
If the answer is too long, summarize the answer.
"""

system_message = SystemMessage(content=prompt)

agent_executor = create_react_agent(llm, tools, messages_modifier=system_message)

# #  Execute
# result = agent_executor.invoke({"messages": [HumanMessage(content="万家口子2023年10月坝下水位的数据变化趋势是什么")]})
result = agent_executor.invoke({"messages": [HumanMessage(content="万家口子2023年土城雨量的变化趋势是怎样的")]})
print(result['messages'][-1].content)

# Create Chain

