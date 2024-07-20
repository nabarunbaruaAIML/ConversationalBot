from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from src import logger
from src.utils.utils import get_mistral_prompt_template
from singleton_decorator import singleton
from langchain_core.runnables import RunnablePassthrough

@singleton
class MistralChat:
    def __init__(self) -> None:
        self.prompt= get_mistral_prompt_template()
        self.output_parser = StrOutputParser()
        
    def get_model(self,temperature:float,top_p:float,top_k:int)->ChatOllama:
        llm = ChatOllama(model='mistral:instruct',temperature=temperature,top_p=top_p,top_k=top_k)
        return llm
    def get_answer(self,question:str,temperature:float,top_p:float,top_k:int)->str:
        llm = self.get_model(temperature,top_p,top_k)
        logger.info("Mistral Model Got Loaded")
        chain = (
                {"question":RunnablePassthrough()}| 
                 self.prompt | 
                 llm | 
                 self.output_parser
                )
        answer = chain.invoke(question)
        logger.info(f"Chain got executed Answer:{answer}")
        return answer