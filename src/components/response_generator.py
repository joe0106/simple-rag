from typing import List
from models import SearchResult, Citation, RAGResponse, RAGResponseStatus
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from abc import ABC
from collections.abc import AsyncIterator

class BaseResponseGenerator(ABC):
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm

    def _format_context(self, context: List[SearchResult]):
        """combine context into a formatted string"""
        format_context = "\n\n".join([
            f"Context {index + 1}:\n{content.text}"
            for index, content in enumerate(context)
        ])
        return format_context

    def _build_prompt(self, is_stream: bool = False) -> PromptTemplate:
        """return instruction for respective prompt mode(streaming/sync response)"""
        instruction = "Response with only the answer text." if is_stream \
            else """
            Your response must be in JSON format with these fields:
            1. "answer": Your detailed response
            2. "citations": A list of objects, each with:
                - "text": The relevant quote from the context
                - "relevance_score": A float between 0-1 indicating how relevant this citation is
            3. "confidence_score": A float between 0-1 indicating your overall confidence. If you're unsure, reflect that in the confidence score
            """
        return PromptTemplate.from_template(
            f"""
            Only use information from the provided context, answer the question.
            
            Context:
            {{context}}

            Question:
            {{query}}

            {instruction}
            """
            )

    def _build_chain(self, streaming: bool = False):
        prompt = self._build_prompt(is_stream=streaming)
        if streaming:
            return prompt | self.llm
        return prompt | self.llm | JsonOutputParser()

class LLMResponseGenerator(BaseResponseGenerator):
    def __init__(self, api_key: str, model: str = "gemini-3.5-flash"):
        llm = ChatGoogleGenerativeAI(
            api_key=api_key,
            model=model,
            temperature=0.7,
            max_output_tokens=1000,
            
        )
        super().__init__(llm)

    def generate_answer(self, query: str, context: List[SearchResult]):
        # build chain
        rag_chain = self._build_chain(streaming=False)

        # invoke chain
        response = rag_chain.invoke({
            "context": self._format_context(context),
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

    async def generate_answer_stream(self, query: str, context: List[SearchResult]) -> AsyncIterator:
        # build chain
        rag_chain = self._build_chain(streaming=True)

        try:
            # stream chain result
            async for chunk in rag_chain.astream({
                "context": self._format_context(context),
                "query": query
            }):
                content = chunk.content
                if isinstance(content, str):
                    if content:
                        print(repr(content))
                        yield content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, str):
                            print(repr(item))
                            yield item
                        elif isinstance(item, dict) and item.get("text"):
                            print(repr(item["text"]))
                            yield item["text"]
        except Exception as e:
            raise Exception(f"LLM streaming service error: {str(e)}")