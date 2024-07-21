from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from src import logger
from src.utils.utils import get_mistral_RAG_prompt_template
from singleton_decorator import singleton
from langchain_core.runnables import RunnablePassthrough
# from langchain.retrievers import BaseRetriever 
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

@singleton
class MistralRAG:
    def __init__(self) -> None:
        self.prompt= get_mistral_RAG_prompt_template()
        # self.output_parser = StrOutputParser()
        
    def get_model(self,temperature:float,top_p:float,top_k:int)->ChatOllama:
        llm = ChatOllama(model='mistral:instruct',temperature=temperature,top_p=top_p,top_k=top_k)
        return llm
    def get_answer(self,retriever,question:str,temperature:float,top_p:float,top_k:int)->str:
        llm = self.get_model(temperature,top_p,top_k)
        logger.info("Mistral Model Got Loaded")
        document_chain=create_stuff_documents_chain(llm,self.prompt)
        retrieval_chain=create_retrieval_chain(retriever,document_chain)
        # chain = (
        #         {"question":RunnablePassthrough()}| 
        #          self.prompt | 
        #          llm | 
        #          self.output_parser
        #         )
        answer = retrieval_chain.invoke({"input":question})
        logger.info(f"Chain got executed Answer:{answer}")
        return answer