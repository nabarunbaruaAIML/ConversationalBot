from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from src.Mistral_RAG_ChatHistory import MistralRAG_ChatHistory
import streamlit as st
import os
from src.utils.utils import get_vector_embedding
import uuid

from dotenv import load_dotenv
load_dotenv()

# store = {}

def main():
    # Check if 'session_id' is already in session_state
    if 'session_id' not in st.session_state:
        # Generate a new session ID
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.retriever = None
    if 'store' not in st.session_state:
        st.session_state.store={}    

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    ## set up Streamlit 
    st.title("Conversational RAG With PDF uplaods and chat history")
    st.sidebar.title("Settings")
    ## Adjust response parameter
    temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
    max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)
    top_p = st.sidebar.slider("Top_P", min_value=0.0, max_value=1.00, value=0.3)
    top_k = st.sidebar.slider("Top_K", min_value=1, max_value=100, value=10)
    sess = 'Session ID: ' + st.session_state.session_id 
    st.write(sess)
    st.write("Upload Pdf's and chat with their content")
    upload_file = st.file_uploader('Choose your PDF file', type="pdf",accept_multiple_files=True)
    if upload_file is not None:
        directory_path = 'artifacts/Online'
        all_files_and_dirs = os.listdir(directory_path)
        files = [f for f in all_files_and_dirs if os.path.isfile(os.path.join(directory_path, f)) and f.lower().endswith('.pdf')]
        # Delete the PDF files
        for pdf_file in files:
            os.remove(os.path.join(directory_path, pdf_file))
        for i in upload_file:
            file_path = os.path.join('artifacts/Online',i.name)
            with open(file_path,"wb") as f:
                f.write(i.getbuffer())
            st.write("Upload Complete")
    # retriever= None
    if st.button("Document Embedding") :
        db_update = st.empty()
        get_vector_embedding()
        # st.write("Vector Database is ready")
        db_update.text("Vector Database is ready")
        

    user_input=st.text_input("Enter your query from the Document")
    if user_input and st.session_state.retriever is not None :
        inst = MistralRAG_ChatHistory()
        # retriever=st.session_state.db.as_retriever() # You have to show different was of Retriver like K=1 or similarity etc
        chain = inst.get_chain(st.session_state.retriever,temperature=temperature,top_p=top_p,top_k=top_k) #question:str, question=user_input,
        conversational_rag_chain = RunnableWithMessageHistory(
                                                                chain,
                                                                get_session_history,
                                                                input_messages_key="input",
                                                                history_messages_key="chat_history",
                                                                output_messages_key="answer",
                                                            )
        
        response = conversational_rag_chain.invoke(
                                                        {"input": user_input},
                                                        config={
                                                                    "configurable": {"session_id": st.session_state.session_id}
                                                                },  # constructs a key "abc123" in `store`.
                                                    )
        st.write(response['answer'])

        st.write("Chat History")
        st.write(response['chat_history'])
        ## With a streamlit expander
        with st.expander("Document similarity Search"):
            for i,doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('------------------------')


if __name__=="__main__":
    main()