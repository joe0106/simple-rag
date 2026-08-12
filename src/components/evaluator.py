from typing import List
from models import SearchResult
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

class Evaluator():
    def __init__(self, model: str = "gemini-3.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.7
        )

    async def evaluate(self, query: str, context: List[SearchResult]):
        # combine context
        format_context = "\n\n".join([
            f"Context {index + 1}:\n{content.text}"
            for index, content in enumerate(context)
        ])
        prompt = """Analyze if the given context contains sufficient information to answer the query.
        Return ONLY a float number between 0 and 1, where:
        - 1.0 means the context perfectly contains all needed information
        - 0.0 means the context has no relevant information
        - Values in between represent partial information
        
        Consider:
        - Relevance of context to the query
        - Completeness of information
        - Reliability of information
        
        Context:
        {context}
        
        Query: {query}
        
        Score (0.0-1.0):"""

        prompt = PromptTemplate.from_template(prompt)

        rag_chain = prompt | self.llm

        result_score = await rag_chain.ainvoke({
            "context": format_context,
            "query": query
        })

        try:
            return float(result_score.text)
        except ValueError:
            return 0.0