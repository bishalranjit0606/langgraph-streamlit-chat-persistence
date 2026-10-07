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


def _instant_answer(query):
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
        {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }
    )
    request = urllib.request.Request(url, headers={"User-Agent": "langgraph-chatbot"})
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.load(response)

    lines = []
    abstract = (payload.get("AbstractText") or "").strip()
    if abstract:
        heading = (payload.get("Heading") or query).strip()
        lines.append(f"{heading}: {abstract}")

    topics = list(payload.get("RelatedTopics") or [])
    added = 0
    while topics and added < 4:
        topic = topics.pop(0)
        if not isinstance(topic, dict):
            continue
        topics = list(topic.get("Topics") or []) + topics
        text = (topic.get("Text") or "").strip()
        if text:
            lines.append(text)
            added += 1
    return lines


def _web_results(query):
    from ddgs import DDGS

    last_error = None
    for kind in ("text", "news"):
        for _attempt in range(2):
            try:
                with DDGS(timeout=20) as ddgs:
                    if kind == "text":
                        rows = ddgs.text(query, max_results=5, backend="duckduckgo") or []
                    else:
                        rows = ddgs.news(query, max_results=5, backend="duckduckgo") or []
                if rows:
                    return rows
            except Exception as exc:
                last_error = exc
    if last_error and not last_error.__class__.__name__.endswith("Exception"):
        raise last_error
    return []


def _wiki_facts(url):
    marker = "/wiki/"
    if "wikipedia.org" not in url or marker not in url:
        return []

    title = urllib.parse.unquote(url.split(marker, 1)[1].split("#", 1)[0].split("?", 1)[0])
    if not title:
        return []

    api = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {
            "action": "parse",
            "page": title,
            "prop": "wikitext",
            "section": "0",
            "format": "json",
        }
    )
    request = urllib.request.Request(api, headers={"User-Agent": "langgraph-chatbot"})
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.load(response)

    wikitext = ((payload.get("parse") or {}).get("wikitext") or {}).get("*") or ""
    wanted = {"gold", "silver", "bronze", "rank", "competitors", "sports"}
    facts = []
    for line in wikitext.splitlines():
        if not line.startswith("|") or "=" not in line:
            continue
        key, value = line[1:].split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key in wanted and value:
            facts.append(f"{key}: {value}")
    return facts


def _wikipedia_backup(query):
    api = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 2,
            "format": "json",
        }
    )
    request = urllib.request.Request(api, headers={"User-Agent": "langgraph-chatbot"})
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.load(response)

    lines = []
    for item in ((payload.get("query") or {}).get("search") or [])[:2]:
        title = item.get("title") or ""
        if not title:
            continue
        slug = urllib.parse.quote(title.replace(" ", "_"))
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
        summary_request = urllib.request.Request(
            summary_url,
            headers={"User-Agent": "langgraph-chatbot"},
        )
        extract = ""
        try:
            with urllib.request.urlopen(summary_request, timeout=15) as response:
                extract = (json.load(response).get("extract") or "").strip()
        except Exception:
            extract = ""
        try:
            facts = _wiki_facts(f"https://en.wikipedia.org/wiki/{slug}")
        except Exception:
            facts = []
        line = title
        if extract:
            line += f": {extract}"
        if facts:
            line += " (" + "; ".join(facts) + ")"
        lines.append(line)
    return lines


@tool
def duckduckgo_search(query: str) -> str:
    """Search the web with DuckDuckGo.

    Use this for current events, news, office holders, sports results, and facts that change.
    Input should be a search query.
    """
    query = query.strip()
    if not query:
        return "Give a search query."

    queries = [query]
    simpler = query
    for prefix in ("who is the ", "who is ", "what is the ", "what is ", "current "):
        while simpler.lower().startswith(prefix):
            simpler = simpler[len(prefix):].strip()
    if simpler and simpler.lower() != query.lower():
        queries.append(simpler)

    parts = []
    instant = []
    for one_query in queries:
        try:
            instant = _instant_answer(one_query)
        except Exception:
            instant = []
        if instant:
            break
    if instant:
        parts.append("Instant answer:\n" + "\n".join(instant))

    results = []
    web_error = None
    for one_query in queries:
        try:
            results = _web_results(one_query)
        except Exception as exc:
            results = []
            web_error = exc
        if results:
            break
    if web_error and not results:
        parts.append(f"Web search failed ({type(web_error).__name__}).")

    pages = []
    wiki_notes = []
    for item in results[:5]:
        title = item.get("title") or ""
        body = item.get("body") or ""
        link = item.get("href") or item.get("url") or item.get("source") or ""
        pages.append(f"{title}. {body} ({link})".strip())
        if len(wiki_notes) < 2 and "wikipedia.org" in link:
            try:
                facts = _wiki_facts(link)
            except Exception:
                facts = []
            if facts:
                wiki_notes.append(f"{title}: " + "; ".join(facts))

    if pages:
        parts.append("Web results:\n" + "\n".join(pages))
    if wiki_notes:
        parts.append("Wikipedia facts:\n" + "\n".join(wiki_notes))

    if not pages and not wiki_notes:
        try:
            backup = _wikipedia_backup(queries[-1])
        except Exception:
            backup = []
        if backup:
            parts.append("Wikipedia:\n" + "\n".join(backup))

    if not parts:
        return (
            "Search failed. Tell the user you could not look this up. "
            "Do not guess from older memory."
        )
    return "\n\n".join(parts)


tools = [calculator, get_stock_price, duckduckgo_search]
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful chatbot. Answer normal questions directly. "
        "Use duckduckgo_search for current events, news, office holders, sports results, and facts that change. "
        "Use get_stock_price for the latest share price. Pass a ticker such as AAPL. "
        "Use calculator for arithmetic, including a follow-up like the cost of many shares. "
        "After a tool runs, answer from that tool result in a normal sentence. "
        "Use only names, dates, and numbers that appear in the tool result. "
        "Do not replace the tool result with older memory. "
        "If the tool says the search failed, say you could not look it up."
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