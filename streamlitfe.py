import streamlit as st

from langgraph_backend import chatbot, get_all_threads, make_config
from langchain_core.messages import HumanMessage
from langsmith import uuid7


def generate_thread_id():
    return str(uuid7())


# Initialize current thread
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


# Initialize message history
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


# Sidebar
st.sidebar.title("LangGraph Chatbot")


# Show current thread ID
st.sidebar.subheader("Current Conversation")

st.sidebar.write(
    st.session_state["thread_id"]
)


TOOL_LABELS = {
    "duckduckgo_search": "DuckDuckGo search",
    "get_stock_price": "Stock price",
    "calculator": "Calculator",
}


def message_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
        )

    return ""


def tool_label(name):
    return TOOL_LABELS.get(name, (name or "Tool").replace("_", " "))


def show_message(message):
    with st.chat_message(message["role"]):
        for tool in message.get("tools") or []:
            st.status(f"{tool['label']} used", state="complete", expanded=False)

        if message.get("content"):
            st.write(message["content"])


def load_conversation(thread_id):
    state = chatbot.get_state(make_config(thread_id))

    messages = []
    pending_tools = []

    for msg in state.values.get("messages", []):
        if msg.type == "ai" and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                name = call.get("name") or "tool"
                pending_tools.append(
                    {
                        "name": name,
                        "label": tool_label(name),
                    }
                )
            continue

        if msg.type == "tool":
            continue

        if msg.type == "human":
            pending_tools = []
            messages.append(
                {
                    "role": "user",
                    "content": message_text(msg.content),
                    "tools": [],
                }
            )
            continue

        if msg.type == "ai":
            content = message_text(msg.content)
            if not content and not pending_tools:
                continue

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tools": pending_tools,
                }
            )
            pending_tools = []

    if pending_tools:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tools": pending_tools,
            }
        )

    return messages


# Start new chat
if st.sidebar.button("Start New Chat"):

    # Generate new thread
    st.session_state["thread_id"] = generate_thread_id()

    # Clear UI message history
    st.session_state["message_history"] = []

    st.rerun()


# Old chats come from SQLite, so they stay after a restart
previous_threads = [
    thread_id
    for thread_id in get_all_threads()
    if thread_id != st.session_state["thread_id"]
]


# Show previous conversations
st.sidebar.subheader("Previous Conversations")

for thread_id in previous_threads:
    if st.sidebar.button(thread_id, key=thread_id):
        st.session_state["thread_id"] = thread_id
        st.session_state["message_history"] = load_conversation(thread_id)

        st.rerun()


# Same thread id goes to SQLite and to LangSmith.
CONFIG = make_config(st.session_state["thread_id"])


# Display previous messages
for message in st.session_state["message_history"]:
    show_message(message)


# User input
user_input = st.chat_input("Type here")


if user_input:

    # Save user message
    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)


    # Generate assistant response
    with st.chat_message("assistant"):
        answer_box = {"widget": None}
        answer = ""
        tools_used = []
        running_tools = {}

        def show_answer(text):
            if answer_box["widget"] is None:
                answer_box["widget"] = st.empty()
            answer_box["widget"].markdown(text)

        def start_tool(name, call_id):
            if call_id and call_id in running_tools:
                return

            # The name can arrive before the call id. Keep one box for that tool.
            if call_id:
                for key, running in list(running_tools.items()):
                    if (
                        not running["done"]
                        and running["name"] == name
                        and str(key).startswith("tool-")
                    ):
                        running_tools[call_id] = running
                        return

            label = tool_label(name)
            status = st.status(f"Using {label}...", state="running", expanded=True)
            status.write(f"{label} is running.")
            running_tools[call_id or f"tool-{len(running_tools)}-{label}"] = {
                "name": name,
                "label": label,
                "status": status,
                "done": False,
            }

        def finish_tool(item):
            if item["done"]:
                return

            item["done"] = True
            item["status"].update(
                label=f"{item['label']} used",
                state="complete",
                expanded=False,
            )
            tools_used.append(
                {
                    "name": item["name"],
                    "label": item["label"],
                }
            )

        try:
            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages",
            ):
                if getattr(message_chunk, "type", "") == "tool":
                    call_id = getattr(message_chunk, "tool_call_id", None)
                    item = running_tools.get(call_id)
                    if item is None:
                        tool_name = getattr(message_chunk, "name", None)
                        item = next(
                            (
                                running
                                for running in running_tools.values()
                                if not running["done"] and running["name"] == tool_name
                            ),
                            None,
                        )
                    if item:
                        finish_tool(item)
                    continue

                calls = getattr(message_chunk, "tool_calls", None) or []
                if calls:
                    for call in calls:
                        start_tool(call.get("name") or "tool", call.get("id"))
                    continue

                for call in getattr(message_chunk, "tool_call_chunks", None) or []:
                    if call.get("name"):
                        start_tool(call.get("name"), call.get("id"))
                    if call.get("name") or call.get("id"):
                        break
                else:
                    content = message_text(message_chunk.content)
                    if content:
                        answer += content
                        show_answer(answer)

            for item in running_tools.values():
                finish_tool(item)

            if not answer:
                answer = "I could not finish that reply. Please try again."
                show_answer(answer)
        except Exception:
            for item in running_tools.values():
                if not item["done"]:
                    item["status"].update(
                        label=f"{item['label']} failed",
                        state="error",
                        expanded=False,
                    )
            answer = "Sorry, that request failed. Please try again."
            show_answer(answer)


    # Save assistant response
    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": answer,
            "tools": tools_used,
        }
    )