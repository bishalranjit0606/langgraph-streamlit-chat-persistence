import streamlit as st
from langchain_core.messages import HumanMessage
from langsmith import uuid7

st.set_page_config(page_title="LangGraph Chatbot", layout="wide")

try:
    import langgraph_backend
    from langgraph_backend import chatbot, get_all_threads, make_config, tool_groups
except Exception as exc:
    st.error("The chatbot could not start. " + type(exc).__name__ + ": " + str(exc))
    st.stop()

_start_fn = getattr(langgraph_backend, "startup_error", None)
_start_problem = _start_fn() if _start_fn else None
if _start_problem is not None:
    st.error(
        "The chatbot could not start. "
        + type(_start_problem).__name__
        + ": "
        + str(_start_problem)
    )
    st.stop()

st.markdown(
    """
    <style>
    .stApp, [data-testid="stHeader"], [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
        color: #1a1a1a;
        font-family: Helvetica, Arial, sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #f7f7f7;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] label {
        color: #1a1a1a;
    }
    [data-testid="stSidebar"] [data-testid^="stBaseButton"] {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #cccccc !important;
    }
    [data-testid="stSidebar"] [data-testid^="stBaseButton"]:hover,
    [data-testid="stSidebar"] [data-testid^="stBaseButton"]:focus,
    [data-testid="stSidebar"] [data-testid^="stBaseButton"]:active {
        background-color: #f3f3f3 !important;
        color: #1a1a1a !important;
        border-color: #1a1a1a !important;
    }
    [data-testid="stSidebar"] [data-testid^="stBaseButton"] p,
    [data-testid="stSidebar"] [data-testid^="stBaseButton"] span {
        color: #1a1a1a !important;
    }
    [class*="st-key-starter-"] [data-testid^="stBaseButton"] {
        background: #f7f7f7 !important;
        background-color: #f7f7f7 !important;
        color: #1a1a1a !important;
        -webkit-text-fill-color: #1a1a1a !important;
        border: 1px solid #cccccc !important;
        width: 100% !important;
        text-align: left !important;
        justify-content: flex-start !important;
    }
    [class*="st-key-starter-"] [data-testid^="stBaseButton"]:hover,
    [class*="st-key-starter-"] [data-testid^="stBaseButton"]:focus,
    [class*="st-key-starter-"] [data-testid^="stBaseButton"]:focus-visible,
    [class*="st-key-starter-"] [data-testid^="stBaseButton"]:active {
        background: #f7f7f7 !important;
        background-color: #f7f7f7 !important;
        color: #1a1a1a !important;
        -webkit-text-fill-color: #1a1a1a !important;
        border: 1px solid #cccccc !important;
    }
    [class*="st-key-starter-"] [data-testid^="stBaseButton"] p,
    [class*="st-key-starter-"] [data-testid^="stBaseButton"] span,
    [class*="st-key-starter-"] [data-testid^="stBaseButton"] div {
        color: #1a1a1a !important;
        -webkit-text-fill-color: #1a1a1a !important;
        text-align: left !important;
    }
    [data-testid="stChatMessage"],
    [data-testid="stChatMessageContent"] {
        background-color: #f7f7f7 !important;
        color: #1a1a1a !important;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span {
        color: #1a1a1a !important;
    }
    [data-testid="stChatMessage"] [data-testid="stExpander"],
    [data-testid="stChatMessage"] [data-testid="stExpander"] details,
    [data-testid="stChatMessage"] [data-testid="stExpander"] summary,
    [data-testid="stChatMessage"] [data-testid="stExpanderDetails"],
    [data-testid="stChatMessage"] pre,
    [data-testid="stChatMessage"] code {
        background: #f7f7f7 !important;
        background-color: #f7f7f7 !important;
        color: #1a1a1a !important;
        -webkit-text-fill-color: #1a1a1a !important;
    }
    [data-testid="stChatMessage"] [data-testid="stExpander"] p,
    [data-testid="stChatMessage"] [data-testid="stExpander"] span,
    [data-testid="stChatMessage"] [data-testid="stExpander"] div {
        color: #1a1a1a !important;
        -webkit-text-fill-color: #1a1a1a !important;
    }
    [data-testid="stAppScrollToBottomContainer"],
    [data-testid="stBottom"] > div {
        background-color: #ffffff !important;
    }
    [data-testid="stChatInput"] > div {
        background-color: #ffffff !important;
        border: 1px solid #cccccc !important;
    }
    [data-testid="stChatInputTextArea"] {
        color: #1a1a1a !important;
        -webkit-text-fill-color: #1a1a1a !important;
    }
    [data-testid="stChatInputSubmitButton"] {
        background-color: #f3f3f3 !important;
        color: #1a1a1a !important;
    }
    [data-testid="stChatInputSubmitButton"] svg {
        fill: #1a1a1a !important;
    }
    section.main > div.block-container {
        max-width: 980px;
        padding-top: 1.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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

mode_label = st.sidebar.radio(
    "Mode",
    ["General", "Constitution RAG"],
    horizontal=True,
    key="chat_mode",
)

if mode_label == "Constitution RAG":
    chat_mode = "rag"
    st.sidebar.caption("Answers only from the Constitution of Nepal.")
    selected_tools = []
else:
    chat_mode = "general"
    st.sidebar.caption("Answers directly, or uses the tools you turn on.")
    mcp_labels = {"Time", "Weather", "Sports"}
    grouped_tools = {"Tools": [], "MCP": []}
    for label, names in tool_groups():
        heading = "MCP" if label in mcp_labels else "Tools"
        grouped_tools[heading].append((label, names))
    selected_tools = []
    for heading in ("Tools", "MCP"):
        st.sidebar.subheader(heading)
        for label, names in grouped_tools[heading]:
            if not names:
                st.sidebar.checkbox(label, value=False, disabled=True)
                continue
            if st.sidebar.checkbox(label, value=True, key=f"enable-{label}"):
                selected_tools.extend(names)


# Show the first question as the chat title.
st.sidebar.subheader("Current Conversation")

current_title = st.sidebar.empty()


TOOL_LABELS = {
    "duckduckgo_search": "DuckDuckGo search",
    "get_stock_price": "Stock price",
    "calculator": "Calculator",
    "search_constitution": "Constitution search",
    "time_get_current_time": "Time",
    "time_convert_time": "Time conversion",
    "weather_get_current_weather": "Weather",
    "weather_get_weather_byDateTimeRange": "Weather forecast",
    "weather_get_weather_details": "Weather details",
    "weather_get_current_datetime": "Weather time",
    "weather_get_timezone_info": "Timezone",
    "weather_convert_time": "Weather time conversion",
    "weather_get_air_quality": "Air quality",
    "weather_get_air_quality_details": "Air quality details",
    "sportscore_get_matches": "Sports scores",
    "sportscore_get_match_detail": "Match details",
    "sportscore_get_team_schedule": "Team schedule",
    "sportscore_get_standings": "Standings",
    "sportscore_get_top_scorers": "Top scorers",
    "sportscore_get_player": "Player stats",
    "sportscore_get_bracket": "Bracket",
    "sportscore_get_tracker": "Live tracker",
}


def _part_text(part):
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        return str(part.get("text") or "")
    text = getattr(part, "text", "")
    return text if isinstance(text, str) else ""


def message_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(_part_text(part) for part in content)

    return ""


def tool_label(name):
    return TOOL_LABELS.get(name, (name or "Tool").replace("_", " "))


def short_title(text):
    text = " ".join(text.split())
    if len(text) <= 42:
        return text
    return text[:42].rstrip() + "..."


def title_from_messages(messages):
    for message in messages:
        if message.get("role") == "user" and (message.get("content") or "").strip():
            return short_title(message["content"])
    return "New chat"


def thread_title(thread_id):
    if thread_id == st.session_state.get("thread_id"):
        history = st.session_state.get("message_history") or []
        if history:
            return title_from_messages(history)

    return title_from_messages(load_conversation(thread_id))


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
    st.session_state["thread_id"] = generate_thread_id()
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

current_title.write(thread_title(st.session_state["thread_id"]))

for thread_id in previous_threads:
    if st.sidebar.button(thread_title(thread_id), key=thread_id):
        st.session_state["thread_id"] = thread_id
        st.session_state["message_history"] = load_conversation(thread_id)

        st.rerun()


# Same thread id goes to SQLite and to LangSmith.
CONFIG = make_config(
    st.session_state["thread_id"],
    mode=chat_mode,
    enabled_tools=selected_tools,
)


# Display previous messages
for message in st.session_state["message_history"]:
    show_message(message)


GENERAL_STARTERS = [
    "What is the official language of Nepal?",
    "What is 19 * 4?",
    "What is the weather in Kathmandu?",
    "What time is it in Nepal?",
    "What does the Constitution of Nepal say is the official language, what is the latest AAPL share price, and how many Nepali rupees does one share cost?",
]

RAG_STARTERS = [
    "What is the official language of Nepal?",
    "Who is the Head of State of Nepal?",
    "What is the national animal of Nepal?",
    "What is the national flower of Nepal?",
    "Where is the capital of Nepal?",
    "What happens if a law conflicts with the Constitution?",
]


# Empty chats offer questions. A click sends the same way as the input box.
starter_slot = st.empty()
picked_question = None
if not st.session_state["message_history"]:
    questions = RAG_STARTERS if chat_mode == "rag" else GENERAL_STARTERS
    with starter_slot.container():
        for index, question in enumerate(questions):
            if st.button(
                question,
                key=f"starter-{chat_mode}-{index}",
                width="stretch",
                wrap=True,
            ):
                picked_question = question


# Always register the input. A starter click must not skip it.
typed_question = st.chat_input("Type here")
user_input = picked_question or typed_question


if user_input:
    starter_slot.empty()

    # Save user message
    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    )
    current_title.write(title_from_messages(st.session_state["message_history"]))

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
            detail = status.empty()
            detail.write(f"{label} is running.")
            running_tools[call_id or f"tool-{len(running_tools)}-{label}"] = {
                "name": name,
                "label": label,
                "status": status,
                "detail": detail,
                "done": False,
            }

        def finish_tool(item):
            if item["done"]:
                return

            item["done"] = True
            if item.get("detail") is not None:
                item["detail"].write(f"{item['label']} finished.")
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

                content = message_text(message_chunk.content)
                if content:
                    answer += content
                    show_answer(answer)

            for item in running_tools.values():
                finish_tool(item)

            if not answer.strip():
                answer = "I could not finish that reply. Please try again."
                show_answer(answer)
        except Exception as exc:
            for item in running_tools.values():
                if not item["done"]:
                    item["status"].update(
                        label=f"{item['label']} failed",
                        state="error",
                        expanded=False,
                    )
            failed = ", ".join(item["label"] for item in running_tools.values())
            if type(exc).__name__ == "RateLimitError":
                answer = "The model is busy right now. Please try that question again."
            elif failed:
                answer = f"Sorry, {failed} failed. Please try again."
            else:
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