from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_serpdive import SerpdiveSearch
import streamlit as st
import time


load_dotenv()

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature = 0.2,
    streaming=True
)

@tool
def search_help(query : str):
    """
    Search the web for up-to-date information relevant to the user's query.
    
    Args:
        query: The search query.
    """
    search = SerpdiveSearch()
    return search.run(query)


agent = create_agent(
    model = llm,
    tools = [search_help],
    system_prompt =("You are a helpful web search agent. "
        "Use the search_web tool whenever current or factual information "
        "needs to be searched on the internet. "
        "Use the search results to provide a clear and accurate answer."
    )
)

st.title("Search Agent Chatbot")
st.markdown("My Search Agent with LangChain, Groq and Streamlit")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message['role']
    content = message['content']
    st.chat_message(role).markdown(content)

def stream_response(messages):
    for message_chunk , metadata  in agent.stream({
        "messages":messages
    },
    stream_mode = "messages"
    ):
       if (
            metadata.get("langgraph_node") == "model"
            and message_chunk.content
        ):
                for char in message_chunk.content:
                    yield char
                    time.sleep(0.001)

query = st.chat_input('Ask me Anything!')  

if query:
    st.session_state.messages.append({"role": "user" , "content" : query})
    st.chat_message("user").markdown(query)
    
    with st.chat_message("assistant"):
        response = st.write_stream(
            stream_response(st.session_state.messages)
        )

    st.session_state.messages.append({'role' : "assistant" , "content" : response})
