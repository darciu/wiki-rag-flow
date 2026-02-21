import mwparserfromhell
import re
import html
import logging
import time
import os
import math
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_wiki_categories(wikicode):
    categories = [
        link.title.strip_code().strip().replace("Kategoria:", "")
        for link in wikicode.filter_wikilinks()
        if link.title.startswith("Kategoria:")
    ]
    return categories


def fetch_wiki_infobox_data(wikicode):

    infobox_data = {
        "imię i nazwisko": None,
        "imię": None,
        "data urodzenia": None,
        "miejsce urodzenia": None,
        "data śmierci": None,
        "miejsce śmierci": None,
        "obywatelstwo": None,
        "nazwa": None,
        "nazwa zwyczajowa": None,
        "państwo":None,
        "kraj": None,
        "miejscowość": None,
        "tytuł": None,
        "liczba ludności": None,
        "rok": None,
    }


    for template in wikicode.filter_templates():
        if "infobox" in str(template.name).lower():
            fields = list(infobox_data.keys())
            for field in fields:
                if template.has(field):
                    val = template.get(field).value.strip_code().strip()
                    if val and not infobox_data[field]:
                        infobox_data[field] = val
    infobox_data = {normalize_key(key):val for key, val in infobox_data.items() if val is not None}
    return infobox_data

def normalize_key(key):
    pl_map = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ ", "acelnoszzACELNOSZZ_")
    key = key.translate(pl_map)
    return key

def fetch_wiki_clean_sections(text):

    match = re.search(r'<text[^>]*>(.*?)</text>', text, re.DOTALL)
    wikitext = html.unescape(match.group(1))
    wikicode = mwparserfromhell.parse(wikitext)
    templates = wikicode.filter_templates()

    infoboxes = [t for t in templates if "infobox" in str(t.name).lower()]

    for ibox in infoboxes:
        try:
            if wikicode.contains(ibox):
                wikicode.remove(ibox)
        except ValueError:
            continue

    for link in wikicode.filter_wikilinks():
        title = str(link.title)
        if title.startswith(("Kategoria:", "Category:")):
            try:
                wikicode.remove(link)
            except ValueError:
                continue

    for template in wikicode.filter_templates():
        try:
            if wikicode.contains(template):
                wikicode.remove(template)
        except ValueError:
            continue

    # divide text into sections that starts with heading
    headings = wikicode.filter_headings()

    sections = {"Wstęp": []}
    current_key = "Wstęp"

    for node in wikicode.nodes:
        if node in headings:
            current_key = str(node.title).strip()
            sections[current_key] = []
        else:

            sections[current_key].append(node)
    final_dict = {
        title: mwparserfromhell.parse(nodes).strip_code().strip() 
        for title, nodes in sections.items() 
        if nodes
    }

    ignored = ["Uwagi", "Przypisy", "Zobacz też", "Linki zewnętrzne", "Bibliografia", "Statystyki"]
    cleaned_dict = {
        k: v.replace('\xa0k', '').strip() for k, v in final_dict.items() 
        if k not in ignored and v.strip()
    }
    
    return cleaned_dict


def process_batch(batch, batch_idx, expected_total_batches, time_start, mongodb_client, weaviate_client, nlp_toolkit):

    logger.info(f'Worker\'s PID: {os.getpid()}')
    logger.info(f'Start new unprocessed batch of full size {len(batch)} : {batch_idx}\n{get_progess_bar(batch_idx,expected_total_batches,time_start)}')
    weaviate_batch = []
    mongodb_batch = []
    batch_for_short = {}
    batch_for_long = {}
    common_structure_batch = {}
    for txt in batch:
        if any(x in txt['title'] for x in ['Kategoria:','Wątek:','Wikipedia:','Szablon:', 'Moduł:', 'Portal:', 'MediaWiki:', 'Pomoc:', 'Wikiprojekt:', 'Plik:']):
            pass
        else:
            
            source_id = txt['_id']
            source_title = txt['title']
            wikicode = mwparserfromhell.parse(txt['content'])
            if len(wikicode) < 100:
                continue
            wiki_categories = fetch_wiki_categories(wikicode)
            wiki_infobox_data = fetch_wiki_infobox_data(wikicode)
            wiki_sections = fetch_wiki_clean_sections(txt['content'])
            cleaned_text = "\n ".join([item for k, v in wiki_sections.items() for item in (k, v)])


            common_structure ={'source_id':source_id,
                            'source_title':source_title,
                            'wiki_categories':wiki_categories,
                            }
            common_structure.update(wiki_infobox_data)
            
            common_structure_batch[source_id] = common_structure

            mongodb_batch.append({"_id":source_id, "plain_article":cleaned_text})

            sections = [f"{k}|||{v}" for k, v in wiki_sections.items()]


            # optimization of chunking
            batch_for_short[source_id] = {}
            batch_for_long[source_id] = {}
            for positional_id, text in enumerate(sections):
                if len(text) < 1000:
                    batch_for_short[source_id][positional_id] =  [text]
                else:
                    batch_for_long[source_id][positional_id] = text
            # end of iteration

    # after batch loop 
    # chunk only long texts
    long_texts_to_process = []
    long_metadata = [] 
    for source_id, sections in batch_for_long.items():
        if sections:
            for positional_id, text in sections.items():
                long_texts_to_process.append(text)
                long_metadata.append((source_id, positional_id))

    del batch_for_long

        
    prefixes = [text.split('|||', 1)[0] for text in long_texts_to_process]
    postfixes = [text.split('|||', 1)[1] for text in long_texts_to_process]
    

    chunked_texts = nlp_toolkit.chunk_texts(postfixes, max_tokens=450)

    result = [
        [f"{p}|||{text}" for text in chunks] 
        for p, chunks in zip(prefixes, chunked_texts)
    ]

    processed_longs = {} 

    for (source_id, positional_id), chunks in zip(long_metadata, result):
        if source_id not in processed_longs:
            processed_longs[source_id] = {}
        processed_longs[source_id][positional_id] = chunks

    merged_all = {}

    for source_id in batch_for_short.keys():
        merged_source = {}
        if processed_longs.get(source_id):
            temp_dict = batch_for_short[source_id] | processed_longs[source_id]
            merged_source.update(temp_dict)
        else:
            merged_source.update(batch_for_short[source_id])
        merged_all[source_id] = merged_source
    
    del batch_for_short



    for sub_dict in merged_all.values():            
        for lista in sub_dict.values():       
            for i in range(len(lista)):      
                parts = lista[i].split("|||")
                lista[i] = ": ".join(parts)

    # gather all chunks
    for key, body in common_structure_batch.items():
        pieces = merged_all[key]
        cnt = 0
        for _, val in sorted(pieces.items()):
            for text in val:
                
                chunk_struct = body.copy()
                chunk_struct.update({'chunk_id':cnt
                                    ,'chunk_text':text})
                weaviate_batch.append(chunk_struct)
                cnt +=1

    del common_structure_batch

    weaviate_client.bulk_upsert(weaviate_batch)
    logger.info(f'Batch of size {len(weaviate_batch)} has been upserted into Weaviate database')
    del weaviate_batch

    mongodb_client.bulk_upsert('wiki_plain_articles', mongodb_batch)
    processed_ids = [doc["_id"] for doc in mongodb_batch]  # albo z oryginalnego batcha, ale tylko tych faktycznie zapisanych
    mongodb_client.mark_processed("wikipedia", processed_ids)
    logger.info(f'Batch of size {len(mongodb_batch)} has been upserted into MongoDB database')
    del mongodb_batch
    time.sleep(1)
    


def get_progess_bar(n: int, total_n: int, time_start: float, unit: str = 'it') -> str:
    progress_string = tqdm.format_meter(
        n=n, 
        total=total_n,
        elapsed=time.time() - time_start, 
        unit=unit,
        bar_format="{l_bar}{bar}{r_bar}" 
    )
    return progress_string



def get_optimal_workers(percentage=0.6):
    total_cores = os.cpu_count() or 3
    workers = math.floor(total_cores * percentage)
    
    return min(7, max(1, workers))

