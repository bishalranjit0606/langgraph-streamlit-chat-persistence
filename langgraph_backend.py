from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langchain_openrouter import ChatOpenRouter

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages

import sqlite3
import os

from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenRouter(
    model="openrouter/free",
    temperature=0,
    streaming=True
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):

    messages = state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }

db_path = os.path.join(os.path.dirname(__file__), "chatbot.db")

# check_same_thread=False lets Streamlit use this connection
conn = sqlite3.connect(db_path, check_same_thread=False)

checkpointer = SqliteSaver(conn)


def get_all_threads():
    query = """
        SELECT thread_id
        FROM checkpoints
        GROUP BY thread_id
        ORDER BY MAX(checkpoint_id) DESC
    """

    with checkpointer.cursor(transaction=False) as cur:
        cur.execute(query)
        return [row[0] for row in cur.fetchall()] 




graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(
    checkpointer=checkpointer
)