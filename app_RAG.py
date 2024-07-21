import streamlit as st

import os
from dotenv import load_dotenv


from src import logger
from src.Mistral_RAG import MistralRAG
from src.utils.utils import get_vector_embedding
## #Title of the app
st.title("RAG based Q&A Chatbot")

st.sidebar.title("Settings")
## Adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)
top_p = st.sidebar.slider("Top_P", min_value=0.0, max_value=1.00, value=0.3)
top_k = st.sidebar.slider("Top_K", min_value=1, max_value=100, value=10)

upload_file = st.file_uploader('Choose your PDF file', type="pdf",accept_multiple_files=True)
if upload_file is not None:
    for i in upload_file:
        file_path = os.path.join('artifacts/Online',i.name)
        with open(file_path,"wb") as f:
            f.write(i.getbuffer())
        st.write("Upload Complete")
# print(upload_file.)
# vector_emb= st.button("Document Embedding")
# if vector_emb is not None:
if st.button("Document Embedding"):
    db_update = st.empty()
    get_vector_embedding()
    # st.write("Vector Database is ready")
    db_update.text("Vector Database is ready")
user_input=st.text_input("Enter your query from the Document")

if user_input:
    inst = MistralRAG()
    retriever=st.session_state.db.as_retriever()
    response = inst.get_answer(retriever,question=user_input,temperature=temperature,top_p=top_p,top_k=top_k)
    st.write(response['answer'])
    ## With a streamlit expander
    with st.expander("Document similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')
