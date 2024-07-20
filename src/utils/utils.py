from langchain_core.prompts import ChatPromptTemplate



def get_mistral_prompt_template()->ChatPromptTemplate:
    prompt=ChatPromptTemplate.from_template("""<s>[INST]You are helpful chatbot who answers all the question as best of your ability. First Analysis the complete Question and only answer to the Question being asked and do not try add extra information apart what is being asked.\n\nQuestion:{question}[/INST]""")
    return prompt

