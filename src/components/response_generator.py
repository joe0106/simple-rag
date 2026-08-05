from typing import List
from models import SearchResult, Citation, RAGResponse, RAGResponseStatus
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class LLMResponseGenerotor:
    def __init__(self, model: str = "gemini-3.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.7,
            max_output_tokens=1000, #test
        )

    def generate_answer(self, query: str, context: List[SearchResult]):
        # combine context
        format_context = "\n\n".join([
            f"Context {index + 1}:\n{content.text}"
            for index, content in enumerate(context)
        ])

        # prompt
        prompt = """Using the provided context, answer the question. Your response must be in JSON format with these fields:
        1. "answer": Your detailed response
        2. "citations": A list of objects, each with:
           - "text": The relevant quote from the context
           - "relevance_score": A float between 0-1 indicating how relevant this citation is
        3. "confidence_score": A float between 0-1 indicating your overall confidence
        
        Only use information from the provided context. If you're unsure, reflect that in the confidence score.
        
        Context:
        {context}
        
        Question: {query}
        
        Respond with only the JSON object, no other text."""

        prompt = PromptTemplate.from_template(prompt)

        # build chain
        rag_chain = prompt | self.llm | JsonOutputParser()

        # invoke chain
        response = rag_chain.invoke({
            "context": format_context,
            "query": query
        })

        citations = [
            Citation(
                text=res["text"],
                metadata={},
                relevance_score=res["relevance_score"]
            ) for res in response["citations"]
        ]

        return RAGResponse(
            answer=response["answer"],
            citations=citations,
            confidence_score=float(response["confidence_score"]),
            status=RAGResponseStatus.SUCCESS
        )