from components.query_remaster import QueryRemaster
from components.retriever import Retriever
from components.evaluator import Evaluator
from components.response_generator import LLMResponseGenerotor

class RAGWorkFlow():
    def __init__(
            self, 
            query_remaster: QueryRemaster, 
            retriever: Retriever,
            evaluator: Evaluator,
            llm_res_gen: LLMResponseGenerotor,
            completion_threshold: float = 0.7,
        ):
        self.remaster = query_remaster
        self.retriever = retriever
        self.evaluator = evaluator
        self.completion_threshold = completion_threshold
        self.llm_res_gen = llm_res_gen

    def execute(self, query: str):
        #query remaster
        remastered_result = self.remaster.reformulate(query)

        #retriever
        context = self.retriever.retrieve(
            remastered_result.remastered_query,
            remastered_result.keywords
        )

        #evaluate
        evaluate_score = self.evaluator.evaluate(query, context)
        if self.completion_threshold > evaluate_score:
            return None
        
        #response generate
        rag_response = self.llm_res_gen.generate_answer(query, context)
        return rag_response