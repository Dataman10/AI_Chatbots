from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv

## Reading the env variables
load_dotenv()

## LLM model
llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0.2
)
st.title("AskBuddy - AI QnA Bot")
st.markdown("My QnA Bot with Langchain and Groq!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask Anything ?")
if query:
    st.session_state.messages.append({"role" : "user" , "content" : query})
    st.chat_message("user").markdown(query)
    res = llm.invoke(query)
    st.chat_message("ai").markdown(res.content)
    st.session_state.messages.append({"role" : "ai" , "content" : res.content})
# while True:

#     user_input = input("You : ")
#     if user_input.lower() == 'exit':
#         print("Thanks for using the Chatbot")
#         break

#     res = llm.invoke(user_input)
#     print(f"AI : {res.content}")