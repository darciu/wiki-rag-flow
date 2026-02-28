from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from instructor.core.client import Instructor
from pydantic import BaseModel, Field

from llm.router import (
    RouteType,
    direct_query,
    further_questions_query,
    paraphase_query,
    rag_query,
    route_query,
    simplify_clean_query,
)

if TYPE_CHECKING:
    from backend.db.weaviate.connection import WeaviateManager
    from parser.nlp.toolkit import NLPToolkit

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Chat response")
    suggested_prompts: list[str] = Field(default_factory=list, description="Suggest")


def unique_chunks(results: list[dict]) -> list[dict]:
    """Filter unique chunks of given wiki article with the highest rank_score"""
    unique_map = {}

    for item in results:
        key = (item["source_id"], item["chunk_id"])

        current_score = item.get("rank_score", -float("inf"))
        if key not in unique_map or current_score > unique_map[key].get(
            "rank_score", -float("inf")
        ):
            unique_map[key] = item
    return sorted(unique_map.values(), key=lambda x: x["rank_score"], reverse=True)


def get_neighbour_context_keys(results: list[dict]) -> list[tuple[str, int]]:
    """For given list of wiki article chunks, get also N-1 and N+1 chunks that do not overlap with existing ones"""
    existing_keys = {(res["source_id"], res["chunk_id"]) for res in results}
    missing_keys = set()

    for res in results:
        s_id = res["source_id"]
        c_id = res["chunk_id"]

        if c_id - 1 >= 0:
            prev_key = (s_id, c_id - 1)
            if prev_key not in existing_keys:
                missing_keys.add(prev_key)

        next_key = (s_id, c_id + 1)
        if next_key not in existing_keys:
            missing_keys.add(next_key)

    return list(missing_keys)


def prepare_context_for_llm(sorted_chunks: list[dict], question) -> str:
    """Query for LLM containing found wiki chunks and also primary question"""
    if not sorted_chunks:
        return "Brak dostępnego kontekstu."

    blocks = []
    current_block = {
        "source_id": sorted_chunks[0]["source_id"],
        "source_title": sorted_chunks[0]["source_title"],
        "last_chunk_id": sorted_chunks[0]["chunk_id"],
        "text": sorted_chunks[0]["chunk_text"],
    }

    for i in range(1, len(sorted_chunks)):
        next_chunk = sorted_chunks[i]

        is_same_doc = next_chunk["source_id"] == current_block["source_id"]
        is_sequential = next_chunk["chunk_id"] == current_block["last_chunk_id"] + 1

        if is_same_doc and is_sequential:
            current_block["text"] += " " + next_chunk["chunk_text"]
            current_block["last_chunk_id"] = next_chunk["chunk_id"]
        else:
            blocks.append(current_block)
            current_block = {
                "source_id": next_chunk["source_id"],
                "source_title": next_chunk["source_title"],
                "last_chunk_id": next_chunk["chunk_id"],
                "text": next_chunk["chunk_text"],
            }

    blocks.append(current_block)

    xml_output = "<context>\n"
    for block in blocks:
        xml_output += (
            f'  <document id="{block["source_id"]}" title="{block["source_title"]}">\n'
            f"    {block['text'].strip()}\n"
            f"  </document>\n"
        )
    xml_output += "</context>"

    xml_output += f"\n\n<question>{question}</question>"

    return xml_output


def rag_search_answer_path(
    llm_client: Instructor,
    weaviate_client: WeaviateManager,
    nlp_toolkit: NLPToolkit,
    question: str,
    model_name: str,
    top_results: int = 6,
    extended_context: bool = True,
) -> ChatResponse:
    simplified_clean = simplify_clean_query(llm_client, question, model_name)
    logger.info(
        f"Simplified split and normalized user question: {simplified_clean.normalized_queries}"
    )

    # paraphase basic question
    paraphases = []
    for cleaned_query in simplified_clean.normalized_queries:
        paraphased = paraphase_query(llm_client, cleaned_query, model_name)
        paraphases.extend(paraphased.expanded_queries)

    logger.info(paraphases)

    queries = simplified_clean.normalized_queries + paraphases
    vectors = weaviate_client.embed_batch_natively(queries)

    basic_chunks = []
    for query_text, query_vector in zip(queries, vectors, strict=True):
        query_results = weaviate_client.single_wikichunk_hybrid_fetch(
            query_text, query_vector, 10, 0.5
        )
        scores = nlp_toolkit.rank(
            query_text, [elem["chunk_text"] for elem in query_results]
        )
        for score, elem in zip(scores, query_results, strict=True):
            elem["rank_score"] = score
        basic_chunks.extend(query_results)
    basic_chunks = unique_chunks(basic_chunks)

    basic_chunks = basic_chunks[:top_results]

    # N-1 and N+1 chunks
    if extended_context:
        missing_keys = get_neighbour_context_keys(basic_chunks)
        grouped_source_chunk_id = defaultdict(list)
        for s_id, c_id in missing_keys:
            grouped_source_chunk_id[s_id].append(c_id)

        extended_chunks = weaviate_client.batch_wikichunk_fetch(grouped_source_chunk_id)
        final_chunks = basic_chunks + extended_chunks
    else:
        final_chunks = basic_chunks

    final_sorted_context = sorted(
        final_chunks, key=lambda x: (x["source_id"], x["chunk_id"])
    )

    context_for_llm = prepare_context_for_llm(final_sorted_context, question)

    rag = rag_query(llm_client, context_for_llm, model_name)

    logger.info(f"Question: {question}\nAnswer: {rag.answer}")

    further_questions = further_questions_query(llm_client, context_for_llm, model_name)

    logger.info(f"Suggested prompts: {further_questions.questions}")
    logger.info("\n-----------\n----------\n----------")

    return ChatResponse(
        answer=rag.answer, suggested_prompts=further_questions.questions
    )


def get_chat_response(
    question: str, model_name: str, llm_client, weaviate_client, nlp_toolkit
) -> ChatResponse:

    route = route_query(llm_client, question, model_name)
    logger.info(f"ROUTE: {route.user_route.value}")

    if route.user_route == RouteType.CLARIFY:
        logger.info(f"Chat response: {route.clarify_message}")
        return ChatResponse(answer=route.clarify_message, suggested_prompts=[])

    elif route.user_route == RouteType.DIRECT:
        direct = direct_query(llm_client, question, model_name)
        logger.info(f"Chat response: {direct.answer}")
        return ChatResponse(answer=direct.answer, suggested_prompts=[])

    elif route.user_route == RouteType.RAG_SEARCH:
        return rag_search_answer_path(
            llm_client, weaviate_client, nlp_toolkit, question, model_name
        )
