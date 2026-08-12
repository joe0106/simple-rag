from components.query_remaster import QueryRemaster
from components.retriever import Retriever
from components.evaluator import Evaluator
from components.response_generator import LLMResponseGenerator
from models import RAGResponse, RAGResponseStatus

class RAGWorkFlow():
    def __init__(
            self, 
            query_remaster: QueryRemaster, 
            retriever: Retriever,
            evaluator: Evaluator,
            llm_res_gen: LLMResponseGenerator,
            completion_threshold: float = 0.7,
        ):
        self.remaster = query_remaster
        self.retriever = retriever
        self.evaluator = evaluator
        self.completion_threshold = completion_threshold
        self.llm_res_gen = llm_res_gen

    async def execute(self, query: str):
        """execute and return response once"""
        #get context
        context = await self.retrieve_context(query)

        #evaluate
        evaluate_score = await self.evaluator.evaluate(query, context)
        if self.completion_threshold > evaluate_score:
            return RAGResponse(
                answer="Failed to answer with reliable response due to insufficient context",
                citations=[],
                status=RAGResponseStatus.INSUFFICIENT_CONTEXT
            )
        
        #response generate
        rag_response = self.llm_res_gen.generate_answer(query, context)
        return rag_response

    async def execute_streaming(self, query: str):
        """execute with streaming mode"""
        #get context
        context = await self.retrieve_context(query)

        #evaluate
        evaluate_score = await self.evaluator.evaluate(query, context)
        if self.completion_threshold > evaluate_score:
            yield "Failed to answer with reliable response due to insufficient context"
            return

        try:
            #stream generated response
            async for chunk in self.llm_res_gen.generate_answer_stream(query, context):
                yield chunk
        except Exception:
            raise

    async def retrieve_context(self, query: str):
        #query remaster
        remastered_result = await self.remaster.reformulate(query)

        #retriever
        context = await self.retriever.retrieve(
            remastered_result.remastered_query,
            remastered_result.keywords
        )

        return context