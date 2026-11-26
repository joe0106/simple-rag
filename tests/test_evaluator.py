from dataclasses import dataclass

import pytest
from langchain_core.runnables import RunnableLambda

from components.evaluator import Evaluator
from models import SearchResult


QUERY = "What is the remote work policy?"
CONTEXT = [
    SearchResult(text="Employees may work remotely two days per week.", metadata={}, score=0.9),
    SearchResult(text="Remote work requires manager approval.", metadata={}, score=0.8),
]


@dataclass
class FakeLLMResponse:
    text: object


def make_evaluator(response, captured_inputs=None):
    captured_inputs = [] if captured_inputs is None else captured_inputs

    def invoke(prompt_value):
        captured_inputs.append(prompt_value.to_string())
        return response

    evaluator = object.__new__(Evaluator)
    evaluator.llm = RunnableLambda(invoke)
    return evaluator, captured_inputs


@pytest.mark.asyncio
async def test_evaluate_formats_context_with_indexes_and_original_text():
    evaluator, captured_inputs = make_evaluator(FakeLLMResponse("0.85"))

    result = await evaluator.evaluate(QUERY, CONTEXT)

    assert result == 0.85
    assert len(captured_inputs) == 1
    prompt = captured_inputs[0]
    assert "Context 1:\nEmployees may work remotely two days per week." in prompt
    assert "Context 2:\nRemote work requires manager approval." in prompt
    assert "Query: What is the remote work policy?" in prompt


@pytest.mark.asyncio
async def test_evaluate_converts_numeric_text_to_float():
    evaluator, _ = make_evaluator(FakeLLMResponse("0.85"))

    assert await evaluator.evaluate(QUERY, CONTEXT) == 0.85


@pytest.mark.asyncio
async def test_evaluate_returns_zero_for_non_numeric_text():
    evaluator, _ = make_evaluator(FakeLLMResponse("The context is sufficient."))

    assert await evaluator.evaluate(QUERY, CONTEXT) == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("score_text", [" 0.8 ", "\n0.8\n", "\t0.8\t"])
async def test_evaluate_parses_score_with_whitespace_or_newlines(score_text):
    evaluator, _ = make_evaluator(FakeLLMResponse(score_text))

    assert await evaluator.evaluate(QUERY, CONTEXT) == 0.8


@pytest.mark.asyncio
@pytest.mark.parametrize("score_text", ["-0.1", "1.1", "2.0"])
async def test_evaluate_currently_accepts_scores_outside_zero_to_one(score_text):
    evaluator, _ = make_evaluator(FakeLLMResponse(score_text))

    assert await evaluator.evaluate(QUERY, CONTEXT) == float(score_text)


@pytest.mark.asyncio
async def test_evaluate_formats_empty_context_without_context_entries():
    evaluator, captured_inputs = make_evaluator(FakeLLMResponse("0.0"))

    result = await evaluator.evaluate(QUERY, [])

    assert result == 0.0
    prompt = captured_inputs[0]
    assert "Context:" in prompt
    assert "Context 1:" not in prompt
    assert "Query: What is the remote work policy?" in prompt


@pytest.mark.asyncio
async def test_evaluate_raises_attribute_error_when_llm_response_has_no_text():
    evaluator, _ = make_evaluator(object())

    with pytest.raises(AttributeError, match="text"):
        await evaluator.evaluate(QUERY, CONTEXT)
