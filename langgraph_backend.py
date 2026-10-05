from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_openrouter import ChatOpenRouter

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages

import sqlite3
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Older LangChain names still work. Copy them to the current LangSmith names.
if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
    os.environ.setdefault("LANGSMITH_TRACING", "true")
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGSMITH_API_KEY", os.environ["LANGCHAIN_API_KEY"])
if os.getenv("LANGCHAIN_PROJECT"):
    os.environ.setdefault("LANGSMITH_PROJECT", os.environ["LANGCHAIN_PROJECT"])
if os.getenv("LANGCHAIN_ENDPOINT"):
    os.environ.setdefault("LANGSMITH_ENDPOINT", os.environ["LANGCHAIN_ENDPOINT"])

os.environ.setdefault("LANGSMITH_PROJECT", "langsmith-demo")


def _make_llm():
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if openrouter_key and openrouter_key != "your_openrouter_key_here":
        return ChatOpenRouter(
            model="openrouter/free",
            temperature=0,
            streaming=True,
        )

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key and groq_key != "your_groq_key_here":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0,
        )

    raise RuntimeError(
        "Add OPENROUTER_API_KEY or GROQ_API_KEY to .env before you start the chatbot."
    )


llm = _make_llm()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def make_config(thread_id: str) -> dict:
    # Same id for the saved chat and the LangSmith thread.
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat",
    }


def chat_node(state: ChatState, config: RunnableConfig):

    messages = state["messages"]

    response = llm.invoke(messages, config)

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