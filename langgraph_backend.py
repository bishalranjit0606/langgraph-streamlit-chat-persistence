from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

import ast
import json
import operator
import sqlite3
import os
import urllib.parse
import urllib.request

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

_CALC_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _calc_eval(node):
    if isinstance(node, ast.Expression):
        return _calc_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_calc_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](
            _calc_eval(node.left),
            _calc_eval(node.right),
        )
    raise ValueError("Use only numbers and + - * / % **.")


@tool
def calculator(expression: str) -> str:
    """Calculate a math expression.

    Use this for arithmetic, including the cost of many shares.
    Pass only the expression, for example 50 * 191.25 or (100 + 5) / 2.
    """
    try:
        value = _calc_eval(ast.parse(expression.strip(), mode="eval"))
    except Exception as exc:
        return f"Could not calculate that: {exc}"

    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


@tool
def get_stock_price(symbol: str) -> str:
    """Get the latest stock price for a ticker symbol.

    Use this when the user asks for a share price or stock quote.
    Pass the ticker, for example AAPL for Apple, TSLA for Tesla, or MSFT for Microsoft.
    """
    ticker = "".join(ch for ch in symbol.upper().strip() if ch.isalnum() or ch in ".-")
    if not ticker:
        return "Give a ticker symbol such as AAPL."

    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.parse.quote(ticker)}?interval=1d&range=1d"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.load(response)
    except Exception as exc:
        return f"Could not get the price for {ticker}: {exc}"

    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        return f"No stock price found for {ticker}. Use a ticker like AAPL."

    meta = result[0].get("meta") or {}
    price = meta.get("regularMarketPrice")
    if price is None:
        return f"No latest price found for {ticker}."

    name = meta.get("shortName") or ticker
    currency = meta.get("currency") or ""
    return f"{name} ({ticker}) latest price is {price} {currency}.".strip()


@tool
def duckduckgo_search(query: str) -> str:
    """Search the web with DuckDuckGo.

    Use this for current events, news, and facts that change.
    Input should be a search query.
    """
    from ddgs import DDGS

    query = query.strip()
    if not query:
        return "Give a search query."

    try:
        # Stay on DuckDuckGo. The auto backend can call Yahoo and time out.
        with DDGS(timeout=15) as ddgs:
            results = ddgs.news(query, max_results=5, backend="duckduckgo") or []
            if not results:
                results = ddgs.text(query, max_results=5, backend="duckduckgo") or []
    except Exception as exc:
        return f"Search failed. Please try again. ({type(exc).__name__})"

    if not results:
        return "No search results found."

    lines = []
    for item in results[:5]:
        title = item.get("title") or ""
        body = item.get("body") or ""
        source = item.get("source") or item.get("href") or ""
        lines.append(f"{title}. {body} ({source})".strip())
    return "\n".join(lines)


tools = [calculator, get_stock_price, duckduckgo_search]
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful chatbot. Answer normal questions directly. "
        "Use duckduckgo_search for current events, news, and facts that change. "
        "Use get_stock_price for the latest share price. Pass a ticker such as AAPL. "
        "Use calculator for arithmetic, including a follow-up like the cost of many shares. "
        "After a tool runs, answer in a normal sentence."
    )
)


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

    messages = [SYSTEM_PROMPT, *state["messages"]]

    response = llm_with_tools.invoke(messages, config)

    return {
        "messages": [response]
    }

def _chat_db_path() -> str:
    app_dir = os.path.dirname(os.path.abspath(__file__))
    # Community Cloud keeps the repo in /mount/src, and that folder is read-only.
    if app_dir.startswith("/mount/src"):
        data_dir = os.path.join(os.path.expanduser("~"), ".langgraph-chat")
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, "chatbot.db")
    return os.path.join(app_dir, "chatbot.db")


db_path = _chat_db_path()

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
graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(
    checkpointer=checkpointer
)