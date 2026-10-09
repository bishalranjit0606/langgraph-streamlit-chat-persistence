from langgraph.graph import START, StateGraph
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
try:
    from langgraph.prebuilt import tools_condition
except ImportError:

    def tools_condition(state):
        messages = state.get("messages") if isinstance(state, dict) else state
        last = messages[-1] if messages else None
        if getattr(last, "tool_calls", None):
            return "tools"
        return "__end__"

import ast
import asyncio
import operator
import os
import queue
import shutil
import sys
import threading
import time
import urllib.parse

import aiosqlite
import httpx
from dotenv import load_dotenv
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    MultiServerMCPClient = None

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
async def calculator(expression: str) -> str:
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


async def _get_json(url: str, timeout: float = 15, user_agent: str = "langgraph-chatbot"):
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url, headers={"User-Agent": user_agent})
        response.raise_for_status()
        return response.json()


@tool
async def get_stock_price(symbol: str) -> str:
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

    try:
        payload = await _get_json(url, timeout=10, user_agent="Mozilla/5.0")
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


async def _instant_answer(query):
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
        {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }
    )
    payload = await _get_json(url)

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

    # One short try. DuckDuckGo often stalls from a hosted server, and the
    # search library waits for that stall even after its own timeout.
    try:
        with DDGS(timeout=6) as ddgs:
            return ddgs.text(query, max_results=5, backend="duckduckgo") or []
    except Exception:
        return []


async def _web_results_capped(query, timeout=8):
    found = {}

    def run():
        found["rows"] = _web_results(query)

    worker = threading.Thread(target=run, name="duckduckgo-search", daemon=True)
    worker.start()
    deadline = time.monotonic() + timeout
    while worker.is_alive() and time.monotonic() < deadline:
        await asyncio.sleep(0.2)
    return found.get("rows") or []


async def _wiki_facts(url):
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
    payload = await _get_json(api)

    wikitext = ((payload.get("parse") or {}).get("wikitext") or {}).get("*") or ""
    wanted = {
        "gold",
        "silver",
        "bronze",
        "rank",
        "competitors",
        "sports",
        "incumbent",
        "incumbent_since",
    }
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


async def _wikipedia_backup(query):
    api = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 2,
            "format": "json",
        }
    )
    payload = await _get_json(api)

    lines = []
    for item in ((payload.get("query") or {}).get("search") or [])[:2]:
        title = item.get("title") or ""
        if not title:
            continue
        slug = urllib.parse.quote(title.replace(" ", "_"))
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
        extract = ""
        try:
            extract = ((await _get_json(summary_url)).get("extract") or "").strip()
        except Exception:
            extract = ""
        try:
            facts = await _wiki_facts(f"https://en.wikipedia.org/wiki/{slug}")
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
async def duckduckgo_search(query: str) -> str:
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
            instant = await asyncio.wait_for(_instant_answer(one_query), 8)
        except Exception:
            instant = []
        if instant:
            break
    if instant:
        parts.append("Instant answer:\n" + "\n".join(instant))

    # Wikipedia answers office-holder questions when DuckDuckGo is blocked.
    if not parts:
        try:
            backup = await asyncio.wait_for(_wikipedia_backup(queries[-1]), 10)
        except Exception:
            backup = []
        if backup:
            parts.append("Wikipedia:\n" + "\n".join(backup))

    # Skip the slow web search once another source already has an answer.
    results = []
    if not parts:
        results = await _web_results_capped(queries[0])

    pages = []
    wiki_notes = []
    for item in results[:5]:
        title = item.get("title") or ""
        body = item.get("body") or ""
        link = item.get("href") or item.get("url") or item.get("source") or ""
        pages.append(f"{title}. {body} ({link})".strip())
        if len(wiki_notes) < 2 and "wikipedia.org" in link:
            try:
                facts = await asyncio.wait_for(_wiki_facts(link), 8)
            except Exception:
                facts = []
            if facts:
                wiki_notes.append(f"{title}: " + "; ".join(facts))

    if pages:
        parts.append("Web results:\n" + "\n".join(pages))
    if wiki_notes:
        parts.append("Wikipedia facts:\n" + "\n".join(wiki_notes))

    if not parts:
        return (
            "Search failed. Tell the user you could not look this up. "
            "Do not guess from older memory."
        )
    return "\n\n".join(parts)


def _constitution_query(question: str) -> str:
    search_text = question.lower()
    for phrase in (
        "according to the constitution of nepal",
        "according to the constitution",
        "in the constitution of nepal",
        "in the constitution",
        "of the constitution",
    ):
        search_text = search_text.replace(phrase, " ")
    search_text = " ".join(search_text.split()).strip(" ?.")
    return search_text or question


@tool
async def search_constitution(question: str) -> str:
    """Search the Constitution of Nepal.

    Use this for questions about the Constitution of Nepal.
    Pass a short search phrase, such as official language of Nepal
    or national animal of Nepal.
    """
    from constitution_rag import format_passages, retrieve

    question = question.strip()
    if not question:
        return "Give a question about the Constitution of Nepal."

    try:
        docs = await asyncio.to_thread(retrieve, _constitution_query(question))
    except Exception as exc:
        return f"Could not search the constitution: {exc}"

    if not docs:
        return "No matching text found in the Constitution of Nepal."

    return (
        "Passages from the Constitution of Nepal. "
        "Answer only from these passages. "
        "If they do not contain the answer, say you could not find it in the constitution.\n\n"
        + format_passages(docs)
    )


tools = [calculator, get_stock_price, duckduckgo_search, search_constitution]


def _message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else (part.get("text") or "")
            for part in content
            if isinstance(part, (str, dict))
        )
    return str(content or "")


def _enabled_names(config: RunnableConfig) -> set[str] | None:
    raw = ((config or {}).get("configurable") or {}).get("enabled_tools")
    if raw is None:
        return None
    return {str(name) for name in raw}


def _selected_tools(config: RunnableConfig):
    enabled = _enabled_names(config)
    if enabled is None:
        return list(tools)
    return [item for item in tools if item.name in enabled]


def _general_prompt(enabled: set[str]) -> str:
    lines = [
        "You are a helpful chatbot. Decide what each question needs.",
        "Answer normal questions directly, with no tool.",
    ]
    if "search_constitution" in enabled:
        lines.append(
            "Use search_constitution for questions about the Constitution of Nepal, "
            "including its articles, rights, president, official language, national symbols, and government structure. "
            "Do not use web search for the constitution."
        )
    if "duckduckgo_search" in enabled:
        lines.append(
            "Use duckduckgo_search for current events, news, office holders, sports results, and facts that change."
        )
    if "get_stock_price" in enabled:
        lines.append(
            "Use get_stock_price for the latest share price. Pass a ticker such as AAPL."
        )
    if "calculator" in enabled:
        lines.append(
            "Use calculator for arithmetic, including a follow-up like the cost of many shares."
        )
    if any(name.startswith("time_") for name in enabled):
        lines.append(
            "Use the time MCP tools for the current time and for converting time between timezones. "
            "Nepal's timezone is Asia/Kathmandu."
        )
    if any(name.startswith("weather_") for name in enabled):
        lines.append(
            "Use the weather MCP tools for current weather, a forecast, and air quality. "
            "Pass a city name such as Kathmandu."
        )
    if any(name.startswith("sportscore_") for name in enabled):
        lines.append(
            "Use the SportScore MCP tools for live and recent football, cricket, basketball, and tennis scores, fixtures, and standings. "
            "The sport argument is football, cricket, basketball, or tennis."
        )
    lines.append(
        "After a tool runs, answer from that tool result in a normal sentence. "
        "When search_constitution was used, mention the article number when the passage includes one. "
        "Use only names, dates, and numbers that appear in the tool result. "
        "Do not replace the tool result with older memory. "
        "If the tool says the search failed, say you could not look it up. "
        "Call only a tool that is available."
    )
    return " ".join(lines)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def make_config(
    thread_id: str,
    mode: str | None = None,
    enabled_tools: list[str] | None = None,
) -> dict:
    # Same id for the saved chat and the LangSmith thread.
    configurable = {"thread_id": thread_id}
    metadata = {"thread_id": thread_id}
    if mode:
        configurable["mode"] = mode
        metadata["mode"] = mode
    if enabled_tools is not None:
        configurable["enabled_tools"] = list(enabled_tools)
    return {
        "configurable": configurable,
        "metadata": metadata,
        "run_name": "chat",
    }


def tool_groups():
    loaded = [item.name for item in tools]
    return [
        ("Calculator", [name for name in loaded if name == "calculator"]),
        ("Web search", [name for name in loaded if name == "duckduckgo_search"]),
        ("Stock price", [name for name in loaded if name == "get_stock_price"]),
        ("Constitution search", [name for name in loaded if name == "search_constitution"]),
        ("Time", [name for name in loaded if name.startswith("time_")]),
        ("Weather", [name for name in loaded if name.startswith("weather_")]),
        ("Sports", [name for name in loaded if name.startswith("sportscore_")]),
    ]


def route_turn(state: ChatState, config: RunnableConfig):
    mode = ((config or {}).get("configurable") or {}).get("mode") or "general"
    if mode == "rag":
        return "rag_node"
    return "chat_node"


async def chat_node(state: ChatState, config: RunnableConfig):
    selected = _selected_tools(config)
    enabled = {item.name for item in selected}
    model = llm.bind_tools(selected) if selected else llm
    messages = [SystemMessage(content=_general_prompt(enabled)), *state["messages"]]
    response = await model.ainvoke(messages, config)
    return {"messages": [response]}


async def rag_node(state: ChatState, config: RunnableConfig):
    from constitution_rag import format_passages, retrieve

    question = ""
    for message in reversed(state["messages"]):
        if getattr(message, "type", "") == "human":
            question = _message_text(message.content).strip()
            break

    try:
        docs = await asyncio.to_thread(retrieve, _constitution_query(question))
    except Exception as exc:
        passages = f"Could not search the constitution: {exc}"
    else:
        passages = (
            format_passages(docs)
            if docs
            else "No matching text found in the Constitution of Nepal."
        )

    prompt = SystemMessage(
        content=(
            "You answer questions about the Constitution of Nepal. "
            "Use only the passages below. "
            "If the passages do not contain the answer, say it was not found in the constitution. "
            "Do not use outside knowledge or other tools. "
            "Mention the article number when a passage includes one.\n\n"
            + passages
        )
    )
    response = await llm.ainvoke([prompt, *state["messages"]], config)
    return {"messages": [response]}


def _tool_result_text(result) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = []
        for item in result:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(result)


async def tools_node(state: ChatState, config: RunnableConfig):
    from langchain_core.messages import ToolMessage

    enabled = _enabled_names(config)
    allowed = {item.name: item for item in tools}
    if enabled is not None:
        allowed = {name: item for name, item in allowed.items() if name in enabled}

    outputs = []
    last = state["messages"][-1]
    for call in getattr(last, "tool_calls", None) or []:
        name = call.get("name") or ""
        call_id = call.get("id") or ""
        tool = allowed.get(name)
        if tool is None:
            outputs.append(
                ToolMessage(
                    content="That tool is turned off.",
                    tool_call_id=call_id,
                    name=name,
                )
            )
            continue
        try:
            result = await asyncio.wait_for(
                tool.ainvoke(call.get("args") or {}),
                25,
            )
        except TimeoutError:
            result = "That tool took too long. Say you could not look this up."
        except Exception as exc:
            result = f"Tool error: {exc}"
        text = _tool_result_text(result)
        if len(text) > 1800:
            text = text[:1800].rstrip() + "\n\n[truncated]"
        outputs.append(
            ToolMessage(
                content=text,
                tool_call_id=call_id,
                name=name,
            )
        )
    return {"messages": outputs}


def _chat_db_path() -> str:
    app_dir = os.path.dirname(os.path.abspath(__file__))
    # Community Cloud keeps the repo in /mount/src, and that folder is read-only.
    if app_dir.startswith("/mount/src"):
        data_dir = os.path.join(os.path.expanduser("~"), ".langgraph-chat")
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, "chatbot.db")
    return os.path.join(app_dir, "chatbot.db")


db_path = _chat_db_path()

_loop = asyncio.new_event_loop()


_loop_ready = threading.Event()


def _run_loop():
    asyncio.set_event_loop(_loop)
    _loop.call_soon(_loop_ready.set)
    _loop.run_forever()


threading.Thread(target=_run_loop, name="langgraph-async", daemon=True).start()
_loop_ready.wait()


def _run(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


def _iter_async(agen_factory):
    events = queue.Queue()

    async def produce():
        try:
            async for item in agen_factory():
                events.put(("item", item))
        except Exception as exc:
            events.put(("error", exc))
        else:
            events.put(("done", None))

    asyncio.run_coroutine_threadsafe(produce(), _loop)
    while True:
        kind, value = events.get()
        if kind == "done":
            break
        if kind == "error":
            raise value
        yield value


def _mcp_servers():
    # Use this project's Python so the MCP servers run inside the same virtual env.
    python = sys.executable
    servers = {
        "time": {
            "command": python,
            "args": ["-m", "mcp_server_time", "--local-timezone=Asia/Kathmandu"],
            "transport": "stdio",
        },
        "weather": {
            "command": python,
            "args": ["-m", "mcp_weather_server"],
            "transport": "stdio",
        },
    }
    # SportScore runs through Node. Streamlit Cloud has no npx, so skip it there.
    npx = shutil.which("npx")
    if npx:
        servers["sportscore"] = {
            "command": npx,
            "args": ["-y", "sportscore-mcp"],
            "transport": "stdio",
        }
    return servers


async def _load_mcp_tools():
    global _mcp

    if MultiServerMCPClient is None:
        return []

    servers = _mcp_servers()
    try:
        _mcp = MultiServerMCPClient(servers, tool_name_prefix=True)
    except Exception as exc:
        print(f"Skipped MCP: {exc}", file=sys.stderr)
        return []

    found = []
    for name in servers:
        try:
            found.extend(
                await asyncio.wait_for(_mcp.get_tools(server_name=name), 20)
            )
        except Exception as exc:
            print(f"Skipped MCP server {name}: {exc}", file=sys.stderr)
    return found


async def _open_chatbot():
    global tools

    mcp_tools = await _load_mcp_tools()
    tools = [calculator, get_stock_price, duckduckgo_search, search_constitution, *mcp_tools]

    conn = await aiosqlite.connect(db_path)
    saver = AsyncSqliteSaver(conn)
    await saver.setup()

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("tools", tools_node)
    graph.add_conditional_edges(
        START,
        route_turn,
        {"chat_node": "chat_node", "rag_node": "rag_node"},
    )
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
    graph.add_edge("rag_node", "__end__")
    return graph.compile(checkpointer=saver), saver


_compiled = None
checkpointer = None
_startup_error = None


class _Chatbot:
    # Streamlit stays synchronous. These methods run the async graph.

    def stream(self, inputs, config, stream_mode="messages"):
        if _startup_error is not None:
            raise _startup_error
        return _iter_async(
            lambda: _compiled.astream(inputs, config, stream_mode=stream_mode)
        )

    def get_state(self, config):
        if _startup_error is not None:
            raise _startup_error
        return _run(_compiled.aget_state(config))


chatbot = _Chatbot()


def get_all_threads():
    if _startup_error is not None or checkpointer is None:
        return []

    async def _threads():
        await checkpointer.setup()
        query = """
            SELECT thread_id
            FROM checkpoints
            GROUP BY thread_id
            ORDER BY MAX(checkpoint_id) DESC
        """
        async with checkpointer.conn.execute(query) as cursor:
            rows = await cursor.fetchall()
        return [row[0] for row in rows]

    return _run(_threads())


def startup_error():
    return _startup_error


try:
    _compiled, checkpointer = _run(_open_chatbot())
except Exception as exc:
    _startup_error = exc
    print(
        f"Chatbot startup failed: {type(exc).__name__}: {exc}",
        file=sys.stderr,
    )
