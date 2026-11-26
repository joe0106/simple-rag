from dataclasses import dataclass

import pytest
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda

import components.response_generator as response_generator_module
from components.response_generator import LLMResponseGenerator
from models import Citation, RAGResponseStatus, SearchResult


CONTEXT = [
    SearchResult(text="First context", metadata={}, score=0.9),
    SearchResult(text="Second context", metadata={}, score=0.8),
]


class FakeChain:
    def __init__(self, response=None, chunks=None, error=None):
        self.response = response
        self.chunks = chunks or []
        self.error = error
        self.invocations = []

    def invoke(self, values):
        self.invocations.append(values)
        return self.response

    async def astream(self, values):
        self.invocations.append(values)
        if self.error is not None:
            raise self.error
        for chunk in self.chunks:
            yield chunk


@dataclass
class FakeChunk:
    content: object


def make_generator():
    generator = object.__new__(LLMResponseGenerator)
    generator.llm = RunnableLambda(lambda value: value)
    return generator


def test_format_context_includes_multiple_contexts_in_order():
    generator = make_generator()

    result = generator._format_context(CONTEXT)

    assert result == "Context 1:\nFirst context\n\nContext 2:\nSecond context"


def test_format_context_returns_empty_string_for_empty_context():
    generator = make_generator()

    assert generator._format_context([]) == ""


def test_build_prompt_contains_json_response_instruction_for_sync_mode():
    generator = make_generator()

    prompt = generator._build_prompt(is_stream=False).format(context="context", query="query")

    assert "Your response must be in JSON format" in prompt
    assert '"answer"' in prompt
    assert '"citations"' in prompt
    assert '"confidence_score"' in prompt
    assert "Response with only the answer text." not in prompt


def test_build_prompt_contains_text_only_instruction_for_streaming_mode():
    generator = make_generator()

    prompt = generator._build_prompt(is_stream=True).format(context="context", query="query")

    assert "Response with only the answer text." in prompt
    assert "Your response must be in JSON format" not in prompt


def test_build_chain_uses_json_output_parser_only_for_sync_mode():
    generator = make_generator()

    sync_chain = generator._build_chain(streaming=False)
    streaming_chain = generator._build_chain(streaming=True)

    assert isinstance(sync_chain.steps[-1], JsonOutputParser)
    assert not any(isinstance(step, JsonOutputParser) for step in streaming_chain.steps)


def test_generate_answer_builds_rag_response_and_converts_citations(monkeypatch):
    generator = make_generator()
    response = {
        "answer": "Remote work is allowed.",
        "citations": [
            {"text": "Employees may work remotely.", "relevance_score": 0.95},
            {"text": "Up to two days per week.", "relevance_score": "0.8"},
        ],
        "confidence_score": "0.87",
    }
    chain = FakeChain(response=response)
    monkeypatch.setattr(generator, "_build_chain", lambda streaming=False: chain)

    result = generator.generate_answer("remote work", CONTEXT)

    assert result.answer == "Remote work is allowed."
    assert result.status is RAGResponseStatus.SUCCESS
    assert result.confidence_score == 0.87
    assert result.citations == [
        Citation(text="Employees may work remotely.", metadata={}, relevance_score=0.95),
        Citation(text="Up to two days per week.", metadata={}, relevance_score=0.8),
    ]
    assert chain.invocations == [
        {
            "context": "Context 1:\nFirst context\n\nContext 2:\nSecond context",
            "query": "remote work",
        }
    ]


@pytest.mark.parametrize("missing_key", ["citations", "answer", "confidence_score"])
def test_generate_answer_raises_key_error_when_required_field_is_missing(monkeypatch, missing_key):
    generator = make_generator()
    response = {
        "answer": "An answer",
        "citations": [],
        "confidence_score": 0.8,
    }
    response.pop(missing_key)
    monkeypatch.setattr(
        generator,
        "_build_chain",
        lambda streaming=False: FakeChain(response=response),
    )

    with pytest.raises(KeyError, match=missing_key):
        generator.generate_answer("query", CONTEXT)


def test_generate_answer_currently_accepts_out_of_range_relevance_score(monkeypatch):
    generator = make_generator()
    monkeypatch.setattr(
        generator,
        "_build_chain",
        lambda streaming=False: FakeChain(
            response={
                "answer": "Answer",
                "citations": [{"text": "Quote", "relevance_score": 1.5}],
                "confidence_score": 0.8,
            }
        ),
    )

    result = generator.generate_answer("query", CONTEXT)

    assert result.citations[0].relevance_score == 1.5


def test_generate_answer_currently_accepts_out_of_range_confidence_score(monkeypatch):
    generator = make_generator()
    monkeypatch.setattr(
        generator,
        "_build_chain",
        lambda streaming=False: FakeChain(
            response={"answer": "Answer", "citations": [], "confidence_score": -0.1}
        ),
    )

    result = generator.generate_answer("query", CONTEXT)

    assert result.confidence_score == -0.1


@pytest.mark.asyncio
async def test_generate_answer_stream_yields_string_content(monkeypatch):
    generator = make_generator()
    chain = FakeChain(chunks=[FakeChunk("first"), FakeChunk("second")])
    monkeypatch.setattr(generator, "_build_chain", lambda streaming=False: chain)

    result = [chunk async for chunk in generator.generate_answer_stream("query", CONTEXT)]

    assert result == ["first", "second"]


@pytest.mark.asyncio
async def test_generate_answer_stream_yields_strings_and_text_from_list_content(monkeypatch):
    generator = make_generator()
    chain = FakeChain(
        chunks=[
            FakeChunk(["first", "second", {"text": "third"}, {"text": ""}]),
        ]
    )
    monkeypatch.setattr(generator, "_build_chain", lambda streaming=False: chain)

    result = [chunk async for chunk in generator.generate_answer_stream("query", CONTEXT)]

    assert result == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_generate_answer_stream_ignores_empty_chunks(monkeypatch):
    generator = make_generator()
    chain = FakeChain(chunks=[FakeChunk(""), FakeChunk([]), FakeChunk({"text": ""})])
    monkeypatch.setattr(generator, "_build_chain", lambda streaming=False: chain)

    result = [chunk async for chunk in generator.generate_answer_stream("query", CONTEXT)]

    assert result == []


@pytest.mark.asyncio
async def test_generate_answer_stream_ignores_unsupported_content_types(monkeypatch):
    generator = make_generator()
    chain = FakeChain(chunks=[FakeChunk(123), FakeChunk({"unexpected": "value"}), FakeChunk(None)])
    monkeypatch.setattr(generator, "_build_chain", lambda streaming=False: chain)

    result = [chunk async for chunk in generator.generate_answer_stream("query", CONTEXT)]

    assert result == []


@pytest.mark.asyncio
async def test_generate_answer_stream_wraps_llm_errors(monkeypatch):
    generator = make_generator()
    monkeypatch.setattr(
        generator,
        "_build_chain",
        lambda streaming=False: FakeChain(error=TimeoutError("upstream timeout")),
    )

    with pytest.raises(Exception, match="^LLM streaming service error: upstream timeout$"):
        [chunk async for chunk in generator.generate_answer_stream("query", CONTEXT)]
