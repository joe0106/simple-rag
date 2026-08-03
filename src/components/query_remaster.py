from typing import List
from dataclasses import dataclass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

@dataclass
class RemasteredQuery:
    remastered_query: str
    keywords: List[str]

class QueryRemaster:
    def __init__(self, model: str = "gemini-3.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.7
        )
    def execute():
        ...
    def reformulate(self, query: str):
        prompt = """Given the user query, reformulate it to be more precise and extract key search terms.
Return your response in this JSON format:
{{
    "refined_query": "reformulated question",
    "keywords": ["key1", "key2", "key3"]
}}

Only return the JSON object, no other text.

User Query: {question}"""
        
        prompt = PromptTemplate.from_template(prompt)
        #todo: add retriver from vector db
        
        # rag_chain = (
        #     { "question": RunnablePassthrough() }
        #     | prompt
        #     | self.llm
        #     | JsonOutputParser()
        # )
        rag_chain = prompt | self.llm | JsonOutputParser()

        response = rag_chain.invoke({"question": query})

        return RemasteredQuery(
            remastered_query=response["refined_query"],
            keywords=response["keywords"]
        )