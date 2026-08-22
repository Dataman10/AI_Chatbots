import uuid

import streamlit as st

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_serpdive import SerpdiveSearch

from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware


# =====================================
# LOAD ENVIRONMENT VARIABLES
# =====================================

load_dotenv()


# =====================================
# LLM
# =====================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2
)


# =====================================
# TOOL INPUT SCHEMA
# =====================================

class SearchInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "The exact search query to search on the internet. "
            "This field is required and must not be empty."
        )
    )


# =====================================
# SEARCH TOOL
# =====================================

@tool(args_schema=SearchInput)
def search_help(query: str) -> str:
    """
    Search the internet for current and up-to-date information.

    Use this tool when the user's question requires current,
    recent, or factual information from the web.
    """

    try:
        search = SerpdiveSearch()

        result = search.run(query)

        return result

    except Exception as e:
        return f"Search failed: {str(e)}"


# =====================================
# MEMORY (was missing before — this line is the fix for Bug 2)
# =====================================

checkpointer = InMemorySaver()


# =====================================
# AGENT
# =====================================

agent = create_agent(
    model=llm,
    tools=[search_help],
    system_prompt="""
You are a helpful AI search assistant.

You can answer general conversational questions directly.

Use the search_help tool ONLY when the user needs current,
recent, real-world, or factual information that should be searched
on the internet.

IMPORTANT TOOL RULES:

1. The search_help tool requires a mandatory argument named `query`.

2. Whenever you call search_help, ALWAYS provide a non-empty
   query argument.

3. Convert the user's question into a clear search query before
   calling the tool.

Example:

User asks:
"Who is the current President of the United States?"

Correct tool call:
search_help(query="current President of the United States")

Never call search_help without the required query argument.

After receiving search results, analyze them and provide a clean,
natural answer to the user.

Do not expose raw JSON, raw tool output, internal tool calls,
or agent reasoning to the user.

Use previous conversation messages to understand follow-up questions
and pronouns.

For example:

User: Who is the President of America?
Assistant: Donald Trump.

User: What is his age?

You should understand that "his" refers to Donald Trump.
""",  # <-- Bug 1 fix: comma added here
    middleware=[
        SummarizationMiddleware(
            # Bug 3 fix: use a Groq model (same provider as main llm),
            # not an Anthropic model with no matching API key.
            model="groq:llama-3.1-8b-instant",
            trigger=("tokens", 4000),
            keep=("messages", 20),
        )
    ],
    checkpointer=checkpointer,
)


# =====================================
# STREAMLIT PAGE
# =====================================

st.set_page_config(
    page_title="Search Agent Chatbot",
    page_icon="🔎"
)

st.title("🔎 Search Agent Chatbot")

st.markdown(
    "My Search Agent with LangChain, Groq and Streamlit"
)


# =====================================
# INITIALIZE CHAT HISTORY
# =====================================

if "messages" not in st.session_state:

    st.session_state.messages = []

# Needed by the checkpointer to know which conversation's memory
# to load/save. One thread_id per browser session.
if "thread_id" not in st.session_state:

    st.session_state.thread_id = str(uuid.uuid4())


# =====================================
# DISPLAY PREVIOUS CHAT HISTORY
# =====================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =====================================
# AGENT STREAMING FUNCTION
# =====================================

def stream_response(messages):

    # Checkpointer needs a thread_id in config on every call, else it
    # raises: "Checkpointer requires one or more of the following
    # 'configurable' keys: thread_id, checkpoint_ns, checkpoint_id"
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    for message_chunk, metadata in agent.stream(
        {
            "messages": messages
        },
        config=config,
        stream_mode="messages"
    ):

        # NOTE: verify "model" matches the actual node name in your
        # installed langgraph/langchain version — some versions name
        # this node "agent" instead of "model". If streaming shows
        # nothing, print(metadata) once to check the real node name.
        if (
            metadata.get("langgraph_node") == "model"
            and message_chunk.content
        ):

            for char in message_chunk.content:
                yield char


# =====================================
# USER INPUT
# =====================================

query = st.chat_input("Ask me anything...")


# =====================================
# PROCESS USER QUERY
# =====================================

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):

        st.markdown(query)

    with st.chat_message("assistant"):

        response = st.write_stream(
            stream_response(
                st.session_state.messages
            )
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )