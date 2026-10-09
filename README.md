# LangGraph Streamlit Chat Persistence

A Streamlit chatbot built with LangGraph. It can answer with tools, or answer only from the Constitution of Nepal. Every conversation is its own thread. SQLite saves the chat, so you can close the app and open an old chat again.

Live app: https://langgraph-app-chat-persistence-fvm4mttbzpbvwzkwigncvi.streamlit.app/

The model is OpenRouter when `OPENROUTER_API_KEY` is set. If that key is missing, the app uses Groq. Replies stream as they are generated.

## Modes

The sidebar has two modes.

**General.** The model answers directly, or it calls a tool you leave turned on.

**Constitution RAG.** The model answers only from retrieved passages in the Constitution of Nepal. Calculator, web search, stock price, and MCP tools are not used in this mode.

## Tools

General mode can use:

- Constitution search, for questions about the Constitution of Nepal
- DuckDuckGo search, for news and current facts
- Stock price, from Yahoo Finance
- Calculator, for arithmetic

Time, weather, and sports scores come from MCP servers. If a server cannot start, that tool is skipped and the rest of the chat still works. SportScore needs Node (`npx`). Streamlit Cloud does not have it, so sports is skipped there.

The calculator, stock price, and web search do not need an API key.

## Constitution RAG

The PDF is split into short chunks, embedded with `sentence-transformers/all-MiniLM-L6-v2`, and stored in FAISS. The first constitution question builds the index. Later questions reuse it.

Locally the index is `faiss_constitution/` next to the code. On Streamlit Cloud the repo folder is read-only, so the index and the chat database are stored under the home directory instead. Neither folder is committed.

## Memory and tracing

Each chat has a `thread_id`. LangGraph saves that thread in SQLite with `AsyncSqliteSaver`, so the messages from one chat stay separate from the next. The sidebar lists previous conversations and loads the one you click.

The same `thread_id` is sent on every model call. With LangSmith tracing on, that chat stays in one LangSmith thread.

## What is in this repo

- `streamlitfe.py` is the Streamlit page.
- `langgraph_backend.py` is the LangGraph app, tools, and SQLite checkpointer.
- `constitution_rag.py` loads the PDF, builds the FAISS index, and retrieves passages.
- `Constitution of Nepal (2nd amd. English)_xf33zb3.pdf` is the source document.
- `requirements.txt` is the Python dependency list.
- `.env.example` lists the keys the app reads. Copy it to `.env`. Do not commit `.env`.

These stay on your machine and are listed in `.gitignore`:

- `.env`
- `chatbot.db`
- `faiss_constitution/`
- `.venv/` and `__pycache__/`

## Setup

You need Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the example env file and add your keys:

```bash
cp .env.example .env
```

You need one model key and a LangSmith key:

- `OPENROUTER_API_KEY` from [OpenRouter keys](https://openrouter.ai/keys), or `GROQ_API_KEY` from [Groq](https://console.groq.com/keys)
- `LANGSMITH_API_KEY` from [LangSmith](https://smith.langchain.com/)

`LANGSMITH_TRACING=true` sends each chat turn to the LangSmith project in `LANGSMITH_PROJECT`.

## Run

```bash
streamlit run streamlitfe.py
```

Streamlit opens a local page, usually at `http://localhost:8501`.

## Use the app

1. Pick **General** or **Constitution RAG** in the sidebar.
2. In General mode, turn tools on or off.
3. Type a message, or click a starter question.
4. Click **Start New Chat** to begin a new conversation.
5. Click a title under **Previous Conversations** to open an old chat.

Old chats stay in SQLite after you stop the app.
