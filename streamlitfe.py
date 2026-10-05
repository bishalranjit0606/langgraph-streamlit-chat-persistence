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


def load_conversation(thread_id):
    state = chatbot.get_state(make_config(thread_id))

    messages = []

    for msg in state.values.get("messages", []):
        role = "user" if msg.type == "human" else "assistant"
        content = msg.content

        if isinstance(content, list):
            content = "".join(
                part if isinstance(part, str) else part.get("text", "")
                for part in content
            )

        messages.append(
            {
                "role": role,
                "content": content
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

    with st.chat_message(message["role"]):
        st.write(message["content"])


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

        def stream_response():

            for message_chunk, metadata in chatbot.stream(
                { 

                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages"
            ):

                if message_chunk.content:
                    yield message_chunk.content


        ai_message = st.write_stream(
            stream_response()
        )


    # Save assistant response
    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )