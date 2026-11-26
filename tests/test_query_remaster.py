import json

import pytest
from langchain_core.runnables import RunnableLambda

from components.query_remaster import QueryRemaster, RemasteredQuery


QUERY = "Where can I work remotely?"


def make_query_remaster(model_output):
    remaster = object.__new__(QueryRemaster)
    remaster.llm = RunnableLambda(lambda _: model_output)
    return remaster


@pytest.mark.asyncio
async def test_reformulate_converts_valid_json_to_remastered_query():
    remaster = make_query_remaster(
        json.dumps(
            {
                "refined_query": "What is the remote work policy?",
                "keywords": ["remote work", "policy"],
            }
        )
    )

    result = await remaster.reformulate(QUERY)

    assert result == RemasteredQuery(
        remastered_query="What is the remote work policy?",
        keywords=["remote work", "policy"],
    )


@pytest.mark.asyncio
async def test_reformulate_preserves_refined_query_whitespace():
    refined_query = "  What is the remote work policy?\n"
    remaster = make_query_remaster(
        json.dumps({"refined_query": refined_query, "keywords": []})
    )

    result = await remaster.reformulate(QUERY)

    assert result.remastered_query == refined_query


@pytest.mark.asyncio
async def test_reformulate_returns_keywords_list():
    keywords = ["remote work", "approval", "policy"]
    remaster = make_query_remaster(
        json.dumps({"refined_query": "remote work policy", "keywords": keywords})
    )

    result = await remaster.reformulate(QUERY)

    assert result.keywords == keywords
    assert isinstance(result.keywords, list)


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_key", ["refined_query", "keywords"])
async def test_reformulate_raises_key_error_when_required_key_is_missing(missing_key):
    response = {"refined_query": "remote work policy", "keywords": ["policy"]}
    response.pop(missing_key)
    remaster = make_query_remaster(json.dumps(response))

    with pytest.raises(KeyError, match=missing_key):
        await remaster.reformulate(QUERY)


@pytest.mark.asyncio
async def test_reformulate_raises_key_error_for_empty_json_object():
    remaster = make_query_remaster("{}")

    with pytest.raises(KeyError, match="refined_query"):
        await remaster.reformulate(QUERY)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_output", ["", "not valid json", "{\"refined_query\":"])
async def test_reformulate_raises_for_empty_or_malformed_json(model_output):
    remaster = make_query_remaster(model_output)

    with pytest.raises(Exception):
        await remaster.reformulate(QUERY)


@pytest.mark.asyncio
async def test_reformulate_currently_accepts_keywords_string_without_validation():
    remaster = make_query_remaster(
        json.dumps({"refined_query": "remote work policy", "keywords": "policy"})
    )

    result = await remaster.reformulate(QUERY)

    assert result.keywords == "policy"
    assert isinstance(result.keywords, str)
