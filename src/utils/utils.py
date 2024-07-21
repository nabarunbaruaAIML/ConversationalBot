from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_mistral_prompt_template()->ChatPromptTemplate:
    prompt=ChatPromptTemplate.from_template("""<s>[INST]You are helpful chatbot who answers all the question as best of your ability. First Analysis the complete Question and only answer to the Question being asked and do not try add extra information apart what is being asked.\n\nQuestion:{question}[/INST]""")
    return prompt

def get_mistral_RAG_prompt_template()->ChatPromptTemplate:
    prompt=ChatPromptTemplate.from_template("""<s>[INST]You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.\n\nContext:{context} Question:{input}[/INST]""")
    return prompt

def get_vector_embedding():
    if "db" not in st.session_state:
        st.session_state.embedding = OllamaEmbeddings(model='mxbai-embed-large')
        ## Data Ingestion step
        st.session_state.loader=PyPDFDirectoryLoader("artifacts/Online") 
        ## Document Loading
        st.session_state.docs=st.session_state.loader.load() 
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.db = Chroma.from_documents(documents=st.session_state.final_documents,embedding=st.session_state.embedding,persist_directory='artifacts/Online',collection_name='Online_Upload')
        # st.write('Vector Store Prepared')