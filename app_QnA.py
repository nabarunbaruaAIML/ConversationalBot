import streamlit as st

import os
from dotenv import load_dotenv

# from loguru import logger
from src import logger
from src.Mistral_QnA import MistralChat
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]= os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]= "true"
os.environ["LANGCHAIN_PROJECT"]= "QnA Bot"

## #Title of the app
st.title("Enhanced Q&A Chatbot With OpenAI")

## Select the OpenAI model
llm=st.sidebar.selectbox("Select Open Source model",["mistral"])

## Adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)
top_p = st.sidebar.slider("Top_P", min_value=0.0, max_value=1.00, value=0.3)
top_k = st.sidebar.slider("Top_K", min_value=1, max_value=100, value=10)

## MAin interface for user input
st.write("Goe ahead and ask any question")
user_input=st.text_input("You:")

if user_input :
    mist = MistralChat()
    logger.info("Mistral Got Instantiated")
    response= mist.get_answer(question=user_input,temperature=temperature,top_p=top_p,top_k=top_k)#generate_response(user_input,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provide the user input")





