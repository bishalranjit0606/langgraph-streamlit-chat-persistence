# LangGraph Streamlit Chat Persistence

A small Streamlit chatbot that keeps every conversation in a local SQLite file. You can start a new chat, close the app, and open an old chat again.

The chatbot uses OpenRouter when `OPENROUTER_API_KEY` is set. If that key is missing, it uses Groq.

The model decides what each question needs. It can answer directly, or it can use a tool:

- Constitution search for the Constitution of Nepal PDF
- DuckDuckGo search for news and current facts
- latest stock price from Yahoo Finance
- a calculator for arithmetic
- time, weather, and sports tools when those MCP servers are available

The constitution tool is traditional RAG. The PDF is split into short chunks, embedded with `sentence-transformers/all-MiniLM-L6-v2`, and stored in FAISS. The first constitution question builds `faiss_constitution/` next to the code. Later questions reuse that index. The model then answers from the retrieved passages.

The calculator, stock price, and web search do not need an API key.

## What is in this repo

- `streamlitfe.py` is the chat page.
- `langgraph_backend.py` is the LangGraph app and the SQLite checkpointer.
- `chatbot.db` is created next to the code when you run the app. It is not committed.

Each chat has its own `thread_id`. LangGraph saves messages for that id in `chatbot.db`. The same id is sent on every model call, so LangSmith keeps that chat in one thread. The sidebar reads those ids and loads the old messages.

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

`LANGSMITH_TRACING=true` sends each chat turn to the LangSmith project in `LANGSMITH_PROJECT`. Do not commit `.env`.

## Run

```bash
streamlit run streamlitfe.py
```

Streamlit opens a local page, usually at `http://localhost:8501`.

## Use the app

1. Type a message in the box at the bottom.
2. Click **Start New Chat** to begin a new conversation.
3. Click a thread id under **Previous Conversations** to open an old chat.

Old chats stay in `chatbot.db` after you stop the app.
