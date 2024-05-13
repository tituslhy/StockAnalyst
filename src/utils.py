#%%
from llama_index.core import (
    SQLDatabase,
    Settings,
    VectorStoreIndex
)
from llama_index.core.agent import ReActAgent
from llama_index.core.indices.struct_store.sql_query import (
    SQLTableRetrieverQueryEngine
)
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import sqlite3
from sqlalchemy import create_engine
import pandas as pd

import os

__curdir__ = os.getcwd()
if "src" in __curdir__:
    db_path = "../database/stocks.db"
else:
    db_path = "./database/stocks.db"

def initialize_agent(llm, db_path = db_path):

    Settings.embed_model = HuggingFaceEmbedding(
        model_name = 'jinaai/jina-embeddings-v2-base-en'
    )
    ## Connect to tables
    conn = sqlite3.connect(db_path)
    sql_query = """SELECT name FROM sqlite_master WHERE type='table';"""
    cursor = conn.cursor()
    cursor.execute(sql_query)
    tables = cursor.fetchall()
    tables = [table[0] for table in tables]
    table_name = {"ilmn": "Illumina",
                  "aapl": "Apple",
                  "nvda": "Nvidia"}
    companies = ", ".join(list(table_name.values()))
    engine = create_engine(f"sqlite:///{db_path}")
    
    ## Create index
    sql_database = SQLDatabase(engine, include_tables = tables)
    table_node_mapping = SQLTableNodeMapping(sql_database)
    table_schema_objs = [
        SQLTableSchema(table_name=table,
                    context_str = f"This table gives information regarding the {table_name[table]}'s stock metrics including closing and opening stock prices and volume.")
        for table in tables
    ]
    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex
    )
    # Create query engine tools
    query_engine = SQLTableRetrieverQueryEngine(
        sql_database,
        obj_index.as_retriever(similarity_top_k = 3),
        llm = llm
    )
    query_engine_tools = [
        QueryEngineTool(
            query_engine = query_engine,
            metadata = ToolMetadata(
                name="stocks",
                description = (
                    f"Provides stock information about these companies: '{companies}' including"
                    " open, high, low, and closing prices, as well as stock volumes on a"
                    " daily basis from 2020-2024. Use a detailed plain text question as"
                    " input to the tool."
                )
            )
        )
    ]
    
    # Return ReActAgent
    return ReActAgent.from_tools(
        tools = query_engine_tools,
        verbose = True,
        llm = llm,
        system_prompt = f"""\
        You are a skilled stock analyst designed to analyse stock price trends against
        public information. You are honest. If you do not know the answer, you will
        not make up an answer. You must always use at least one of the tools
        provided when answering a question."""
    )

#%%
if __name__ == "__main__":
    from llama_index.llms.bedrock import Bedrock
    from llama_index.core.callbacks import CallbackManager,LlamaDebugHandler
    
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    Settings.callback_manager = CallbackManager([llama_debug])
    llm = Bedrock(
        model = "anthropic.claude-3-opus-20240229-v1:0",
        aws_access_key_id = os.environ["AWS_ACCESS_KEY"],
        aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_region_name = os.environ["AWS_DEFAULT_REGION"]
    )
    agent = initialize_agent(llm=llm, db_path = "../database/stocks.db")
    query_str = "Describe the general closing price trend for Apple in 2020, 2021, 2022 and 2023. These years were plagued with economic recessions arising from COVID-19. Was Apple heavily affected?"
    response = agent.stream_chat(query_str)
    response.print_response_stream()
    
# %%
