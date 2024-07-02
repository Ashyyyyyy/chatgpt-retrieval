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

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain



# sqlite setup
# def database_setup(cursor):
#     cursor.execute("""CREATE TABLE Users(
#         Id INT,
#         username TEXT,
#         password TEXT,
#         status TEXT,
#         admin INT
#     )""")
#     cursor.execute("""INSERT INTO Users
#         VALUES(1, 'Hulk', 'greengrocer', 'SMASHING THINGS', 0)""")
#     cursor.execute("""INSERT INTO Users
#         VALUES(2, 'Bobby', 'lolcats', 'serving tables', 1)""")
#     conn.commit()
#
# conn = sqlite3.connect(":memory:")
# cur = conn.cursor()
# database_setup(cur)
# db = SQLDatabase.from_uri("sqlite:///:memory:")
#
# from sqlalchemy import create_engine
# engine = create_engine('sqlite:///:memory:')
# conn = engine.raw_connection()
# cur = conn.cursor()
# database_setup(cur)
# conn.close()
#
# print(db.dialect)
# print(db.get_usable_table_names())
# result = db.run("SELECT * FROM Users LIMIT 10;")
# print(result)

# llm setup
os.environ["OPENAI_API_KEY"] = constants.APIKEY
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

db = SQLDatabase.from_uri("sqlite:////Users/henrisun/Documents/RAG/chatgpt-retrieval/db/Config.db3")
print(db.dialect)
print(db.get_usable_table_names())
# db.run("SELECT * FROM Artist LIMIT 10;")

table_descriptions = '''
Here are descriptions of tables.

VSKY_DATA_REAL：显示采集上传的实时数据的详细信息
VSKY_DATA_DT：显示采集上传的小时数据的详细信息
VSKY_DATA_DAY：显示采集上传的日数据的详细信息
VSKY_DATA_XUN：显示采集上传的旬数据的详细信息
VSKY_DATA_MONTH：显示采集上传的月数据的详细信息
VSKY_POINTINFO_REAL：实时数据的点号配置详细信息
VSKY_POINTINFO_DT：小时数据的点号配置详细信息
VSKY_POINTINFO_DAY：日数据的点号配置详细信息
VSKY_POINTINFO_XUN：旬数据的点号配置详细信息
VSKY_POINTINFO_MONTH：月数据的点号配置详细信息
VSKY_POINTINFO：所有数据的点号配置详细信息
VSKY_POINTLIST：所有数据的点号配置详细信息，并对其中的代码进行了翻译

2)上传时段数据表SKY_DATA_DT（DAY，XUN，MONTH）

序号	名称	字段含义	类型	主键	空	说明
1	ID	自增序号	int	Y1	N	自增
2	PID	上传点号	int		N	主键表 SKY_CFG_POINT
3	PNM	点号名	varchar(50)		N	
4	TSRC	数据源时间	datetime		N	数据在水调库里的时间
5	T	数据时间	datetime		N	
6	V	数据值	numeric(20,6)		N	
7	FORMAT	数据显示格式	int		N	暂时不用，-X-X位有效数字，X-X位小数
8	TMAX	最大值时间	datetime		N	
9	VMAX	最大值	numeric(20,6)		N	
10	TMIN	最小值时间	datetime		N	
11	VMIN	最小值	numeric(20,6)		N	
12	VAVG	平均值	numeric(20,6)		N	
13	DBCONNECT	来源数据库连接名称	varchar(50)		N	
14	DBSERVER	来源数据库服务器地址	varchar(50)		N	
15	UPOK	是否上传成功	int		N	0-成功（默认），1-未成功
16	SN	入库方式	int		N	0-自动，1-人工
17	TIN	数据入库时间	datetime		N	默认当前时间
18	VTYPE	V值类型	integer		N	0-无效，1-有效，其它-未知
19	MAXTYPE	VMAX值类型	integer		N	0-无效，1-有效，其它-未知
20	MINTYPE	VMIN值类型	integer		N	0-无效，1-有效，其它-未知
21	AVGTYPE	VAVG值类型	integer		N	0-无效，1-有效，其它-未知
'''

context = db.get_context()
# print(list(context))
# print(context["table_info"])

# chain = create_sql_query_chain(llm, db)
# print("create_sql_query_chain: ", chain)
# response = chain.invoke({"question": "How many employees are there"})
# print("response: ", response)


write_query = create_sql_query_chain(llm, db, k=10)
# Add table info into the prompt.
# prompt_with_info = write_query.get_prompts()[0].partial(table_info=table_descriptions)
prompt_with_info = write_query.get_prompts()[0].partial(table_info=context["table_info"])
prompt_with_info.template += '\nNote that you must query the schema of ONLY ONE most relative table.'
print(prompt_with_info.template)
write_query_with_info = create_sql_query_chain(llm, db, k=10, prompt=prompt_with_info)
# print(write_query_with_info.get_prompts()[0].pretty_print())
print('response: ', write_query_with_info.invoke({"question": "万家口子2023年10月11日四格雨量的旬数据是多少"}))
execute_query = QuerySQLDataBaseTool(db=db)
# chain = write_query | execute_query
# print("write_query | execute_query chain: ", chain)
# print("result: ", chain.invoke({"question": "万家口子2023年10月11日老屋基雨量的旬数据是多少"}))


answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.
    If you don't have enough information to answer. Please attach the table name and SQL Queries for user to look at the sql.
    Answer in Chinese.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

chain = (
    RunnablePassthrough.assign(query=write_query_with_info).assign(
        result=itemgetter("query") | execute_query
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

# result = chain.invoke({"question": "万家口子2023年10月坝下水位的数据变化趋势是什么"})
# result = chain.invoke({"question": "万家口子2023年土城雨量的变化趋势是怎样的"})
# result = chain.invoke({"question": "万家口子2023年10月11日土城雨量的旬数据是多少"})
result = chain.invoke({"question": "万家口子2023年10月11日四格雨量的旬数据是多少"})
print(result)