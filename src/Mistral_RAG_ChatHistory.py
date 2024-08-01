from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from src import logger
from src.utils.utils import get_mistral_chat_aware_prompt_template ,get_mistral_RAG_chat_aware_prompt_template
from singleton_decorator import singleton
from langchain_core.runnables import RunnablePassthrough
# from langchain.retrievers import BaseRetriever 
from langchain.chains import create_retrieval_chain , create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import Runnable

@singleton
class MistralRAG_ChatHistory:
    def __init__(self) -> None:
        self.chat_aware_Prompt_template = get_mistral_chat_aware_prompt_template()
        self.RAG_Prompt_Template = get_mistral_RAG_chat_aware_prompt_template()
        # self.output_parser = StrOutputParser()
        
    def get_model(self,temperature:float,top_p:float,top_k:int)->ChatOllama:
        llm = ChatOllama(model='mistral:instruct',temperature=temperature,top_p=top_p,top_k=top_k)
        return llm
    def get_chain(self,retriever,temperature:float,top_p:float,top_k:int)->Runnable:
        llm = self.get_model(temperature,top_p,top_k)
        logger.info("Mistral Model Got Loaded")
        history_aware_retriever=create_history_aware_retriever(llm,retriever,self.chat_aware_Prompt_template)
        question_answer_chain=create_stuff_documents_chain(llm,self.RAG_Prompt_Template)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
         
        logger.info(f"Chain got created:{rag_chain}")
        return rag_chain