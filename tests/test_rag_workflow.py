from dataclasses import dataclass

import pytest

from components.query_remaster import RemasteredQuery
from components.rag_workflow import RAGWorkFlow
from models import RAGResponse, RAGResponseStatus, SearchResult


QUERY = "Where can I work remotely?"
REMASTERED_QUERY = "What is the company's remote work policy?"
KEYWORDS = ["remote work", "policy"]
CONTEXT = [SearchResult(text="Employees may work remotely two days per week.", metadata={}, score=0.9)]


class FakeQueryRemaster:
    def __init__(self, events):
        self.events = events

    async def reformulate(self, query):
        self.events.append(("remaster", query))
        return RemasteredQuery(REMASTERED_QUERY, KEYWORDS)


class FakeRetriever:
    def __init__(self, events, context=CONTEXT):
        self.events = events
        self.context = context

    async def execute(self, query, keywords):
        self.events.append(("retrieve", query, keywords))
        return self.context


class FakeEvaluator:
    def __init__(self, events, score):
        self.events = events
        self.score = score

    async def evaluate(self, query, context):
        self.events.append(("evaluate", query, context))
        return self.score


@dataclass
class FakeAnswerGenerator:
    events: list
    response: RAGResponse
    chunks: list[str]
    stream_error: Exception | None = None

    def generate_answer(self, query, context):
        self.events.append(("generate", query, context))
        return self.response

    async def generate_answer_stream(self, query, context):
        self.events.append(("generate_stream", query, context))
        if self.stream_error is not None:
            raise self.stream_error
        for chunk in self.chunks:
            yield chunk


def make_workflow(score=0.9, *, events=None, response=None, chunks=None, stream_error=None):
    events = [] if events is None else events
    response = response or RAGResponse(answer="Remote work is allowed.", citations=[])
    chunks = ["Remote ", "work ", "is allowed."] if chunks is None else chunks
    return (
        RAGWorkFlow(
            FakeQueryRemaster(events),
            FakeRetriever(events),
            FakeEvaluator(events, score),
            FakeAnswerGenerator(events, response, chunks, stream_error),
            completion_threshold=0.7,
        ),
        events,
    )


@pytest.mark.asyncio
async def test_execute_calls_dependencies_in_order_and_returns_generated_response():
    workflow, events = make_workflow()
    expected = RAGResponse(answer="Generated answer", citations=[])
    workflow.llm_res_gen.response = expected

    result = await workflow.execute(QUERY)

    assert result == expected
    assert [event[0] for event in events] == [
        "remaster",
        "retrieve",
        "evaluate",
        "generate",
    ]
    assert events[1][1:] == (REMASTERED_QUERY, KEYWORDS)
    assert events[2][1:] == (QUERY, CONTEXT)
    assert events[3][1:] == (QUERY, CONTEXT)


@pytest.mark.asyncio
async def test_retrieve_context_passes_remastered_query_and_keywords_to_retriever():
    workflow, events = make_workflow()

    result = await workflow.retrieve_context(QUERY)

    assert result == CONTEXT
    assert events == [
        ("remaster", QUERY),
        ("retrieve", REMASTERED_QUERY, KEYWORDS),
    ]


@pytest.mark.asyncio
async def test_execute_returns_insufficient_context_when_score_is_below_threshold():
    workflow, events = make_workflow(score=0.69)

    result = await workflow.execute(QUERY)

    assert result.status is RAGResponseStatus.INSUFFICIENT_CONTEXT
    assert result.answer == "Failed to answer with reliable response due to insufficient context"
    assert not any(event[0] == "generate" for event in events)


@pytest.mark.asyncio
async def test_execute_generates_answer_when_score_equals_threshold():
    workflow, events = make_workflow(score=0.7)

    result = await workflow.execute(QUERY)

    assert result.answer == "Remote work is allowed."
    assert events[-1] == ("generate", QUERY, CONTEXT)


@pytest.mark.asyncio
async def test_execute_streaming_returns_only_fallback_when_context_is_insufficient():
    workflow, events = make_workflow(score=0.69)

    result = [chunk async for chunk in workflow.execute_streaming(QUERY)]

    assert result == ["Failed to answer with reliable response due to insufficient context"]
    assert [event[0] for event in events] == ["remaster", "retrieve", "evaluate"]


@pytest.mark.asyncio
async def test_execute_streaming_yields_chunks_in_order():
    workflow, events = make_workflow(chunks=["first", "second", "third"])

    result = [chunk async for chunk in workflow.execute_streaming(QUERY)]

    assert result == ["first", "second", "third"]
    assert [event[0] for event in events] == [
        "remaster",
        "retrieve",
        "evaluate",
        "generate_stream",
    ]


@pytest.mark.asyncio
async def test_execute_streaming_propagates_generator_exception():
    error = RuntimeError("stream failed")
    workflow, events = make_workflow(stream_error=error)

    with pytest.raises(RuntimeError, match="stream failed"):
        [chunk async for chunk in workflow.execute_streaming(QUERY)]

    assert events[-1] == ("generate_stream", QUERY, CONTEXT)