import time

from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv


# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()


# -----------------------------
# LLM Model
# -----------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)


# -----------------------------
# Page UI
# -----------------------------
st.title("AskBuddy - AI QnA Bot")
st.markdown("My QnA Bot with LangChain and Groq!")


# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    role = message["role"]
    content = message["content"]

    st.chat_message(role).markdown(content)


# -----------------------------
# Streaming Function
# -----------------------------
def stream_response(query):

    for chunk in llm.stream(query):

        text = chunk.content

        for char in text:

            yield char

            time.sleep(0.01)


# -----------------------------
# User Input
# -----------------------------
query = st.chat_input("Ask Anything?")


if query:

    # -------------------------
    # Display User Message
    # -------------------------
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    st.chat_message("user").markdown(query)


    # -------------------------
    # Display AI Response
    # -------------------------
    with st.chat_message("assistant"):

        response = st.write_stream(
            stream_response(query)
        )


    # -------------------------
    # Save AI Response
    # -------------------------
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })