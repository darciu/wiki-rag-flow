from llm.router import route_query, direct_query, simplify_clean_query, paraphase_query, RouteType, further_questions_query, answer_query
from openai import OpenAI
import instructor
from collections import defaultdict
from parser.nlp.toolkit import NLPToolkit
from backend.db.weaviate.connection import WeaviateManager
from config import WeaviateSettings
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def unique_chunks(results: list[dict]) -> list[dict]:
    unique_map = {}

    for item in results:
        key = (item['source_id'], item['chunk_id'])
        
        current_score = item.get('rank_score', -float('inf'))
        if key not in unique_map or current_score > unique_map[key].get('rank_score', -float('inf')):
            unique_map[key] = item
    return sorted(unique_map.values(), key=lambda x: x['rank_score'], reverse=True)


def get_neighbour_context_keys(results: list[dict]) -> list[tuple[str, int]]:

    existing_keys = {(res['source_id'], res['chunk_id']) for res in results}
    missing_keys = set()

    for res in results:
        s_id = res['source_id']
        c_id = res['chunk_id']

        if c_id - 1 >= 0:
            prev_key = (s_id, c_id - 1)
            if prev_key not in existing_keys:
                missing_keys.add(prev_key)

        next_key = (s_id, c_id + 1)
        if next_key not in existing_keys:
            missing_keys.add(next_key)

    return list(missing_keys)


def prepare_context_for_llm(sorted_chunks: list[dict], question) -> str:
    if not sorted_chunks:
        return "Brak dostępnego kontekstu."

    blocks = []
    current_block = {
        "source_id": sorted_chunks[0]["source_id"],
        "source_title": sorted_chunks[0]["source_title"],
        "last_chunk_id": sorted_chunks[0]["chunk_id"],
        "text": sorted_chunks[0]["chunk_text"]
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
                "text": next_chunk["chunk_text"]
            }
    
    blocks.append(current_block)

    xml_output = "<context>\n"
    for block in blocks:
        xml_output += (
            f'  <document id="{block["source_id"]}" title="{block["source_title"]}">\n'
            f'    {block["text"].strip()}\n'
            f'  </document>\n'
        )
    xml_output += "</context>"

    xml_output += f"\n\n<question>{question}</question>"

    return xml_output

def rag_search_answer_path(client_llm, question, model_name):
    result = simplify_clean_query(client_llm, question, model_name)
    print('-'*30)
    print('Znormalizowane pytanie: ', result.normalized_queries)
    print('-'*30)
    paraphases = []
    for cleaned_query in result.normalized_queries:
        result2 = paraphase_query(client_llm, cleaned_query, model_name)
        paraphases.extend(result2.expanded_queries)

    queries = result.normalized_queries + paraphases
    vectors = weaviate_client.embed_batch_natively(queries)
    all_results = []
    for query_text, query_vector in zip(queries, vectors, strict=True):
        query_results = weaviate_client.single_wikichunk_hybrid_fetch(query_text, query_vector, 10, 0.5)
        scores = nlp_toolkit.rank(query_text, [elem['chunk_text'] for elem in query_results])
        for score, elem in zip(scores, query_results):
            elem['rank_score'] = score
        all_results.extend(query_results)
    all_results = unique_chunks(all_results)

    top_results = all_results[:6]
    missing_keys = get_neighbour_context_keys(top_results)

    grouped_source_chunk_id = defaultdict(list)
    for s_id, c_id in missing_keys:
        grouped_source_chunk_id[s_id].append(c_id)

    fetched_chunks = weaviate_client.batch_wikichunk_fetch(grouped_source_chunk_id)

    final_chunks = top_results + fetched_chunks

    final_sorted_context = sorted(final_chunks, key=lambda x: (x['source_id'], x['chunk_id']))

    context_for_llm = prepare_context_for_llm(final_sorted_context, question)

    answer = answer_query(client_llm, context_for_llm, model_name)

    print(answer.answer)

    further_questions = further_questions_query(client_llm, context_for_llm, model_name)

    for further_question in further_questions.questions:
        print(further_question)



def get_chat_response(question: str, model_name: str):

    weaviate_settings = WeaviateSettings()
    weaviate_api_key = weaviate_settings.WEAVIATE_APIKEY_KEY

    weaviate_client = WeaviateManager(
        api_key=weaviate_api_key,
        host="127.0.0.1",
        native_embedding_url="http://127.0.0.1:8008/embed",
    )
    nlp_toolkit = NLPToolkit()


    client_llm = instructor.from_openai(
        OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        ),
        mode=instructor.Mode.JSON,
    )
    

    result = route_query(client_llm, question, model_name)
    
    logger.info(f"Decyzja (Route): {result.user_route.value}")
    
    if result.user_route == RouteType.CLARIFY:
        print(f"Wiadomość od LLM: {result.clarify_message}")
        
    elif result.user_route == RouteType.DIRECT:
        direct_answer = direct_query(client_llm, question, model_name)
        print(direct_answer.answer)

    elif result.user_route == RouteType.RAG_SEARCH:
        rag_search_answer_path(client_llm, question, model_name)
