# LangGraph Streamlit Chat Persistence

A small Streamlit chatbot that keeps every conversation in a local SQLite file. You can start a new chat, close the app, and open an old chat again.

The model is `openrouter/free` through [OpenRouter](https://openrouter.ai/).

## What is in this repo

- `streamlitfe.py` is the chat page.
- `langgraph_backend.py` is the LangGraph app and the SQLite checkpointer.
- `chatbot.db` is created next to the code when you run the app. It is not committed.

Each chat has its own `thread_id`. LangGraph saves messages for that id in `chatbot.db`. The sidebar reads those ids and loads the old messages.

## Setup

You need Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the example env file and put your OpenRouter key in it:

```bash
cp .env.example .env
```

`.env` should look like this:

```
OPENROUTER_API_KEY=your_openrouter_key_here
```

Get a key from [OpenRouter keys](https://openrouter.ai/keys). Do not commit `.env`.

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
