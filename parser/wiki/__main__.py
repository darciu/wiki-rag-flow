import logging
import mwparserfromhell
import re
import html
from backend.db.mongodb.connection import MongoManager
from config import MongoDBSettings
from parser.nlp.wiki_tools import fetch_wiki_categories, fetch_wiki_infobox_data, fetch_wiki_clean_sections
from parser.nlp.toolkit import NLPToolkit

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

mongodb_settings = MongoDBSettings()

MONGODB_URI = mongodb_settings.mongodb_uri
mongodb_client = MongoManager(MONGODB_URI, "scraper_db")
nlp_toolkit = NLPToolkit()


if __name__ == "__main__":
    logger.info("PARSER WIKI")
    if not mongodb_client.is_healthy():
        logger.critical("PARSER WIKI Could not establish connection with MongoDB")
        exit(1)
    # analogicznie sprawdzenie połączenia z weaviate
        
    weaviate_batch = []
    mongodb_batch = []
    generator = mongodb_client.fetch_batches("wikipedia", batch_size=100)
    for batch in generator:

        for record in batch:
            if any(x in record['title'] for x in ['Kategoria:','Wątek:','Wikipedia:','Szablon:', 'Moduł:', 'Portal:', 'MediaWiki:', 'Pomoc:', 'Wikiprojekt:', 'Plik:']):
                pass
            else:
                

                source_id = record['_id']
                source_title = record['title']
                content = record['content']
                wikicode = mwparserfromhell.parse(content)
                wiki_categories = fetch_wiki_categories(wikicode)
                wiki_infobox_data = fetch_wiki_infobox_data(wikicode)
                wiki_sections = fetch_wiki_clean_sections(content)
                cleaned_text = "\n".join([item for k, v in wiki_sections.items() for item in (k, v)])
                mongodb_load = {'source_id':source_id, 'content':cleaned_text}
                mongodb_batch.append(mongodb_load)

                keys = list(wiki_sections.keys())
                values = list(wiki_sections.values())
                chunks = nlp_toolkit.chunk_texts(values, max_tokens=300)
                wiki_dict = {keys[i]: chunks[i] for i in range(len(keys))}

                wiki_chunks = [f"{name}: {desc}" for name, desc_list in wiki_dict.items() for desc in desc_list]

                entities = nlp_toolkit.extract_ner_entities(cleaned_text)
                personalia = [elem['entity'] for elem in entities.personalia]
                personalia = nlp_toolkit.lemmatize(personalia)
                locations = [elem['entity'] for elem in entities.locations]
                organizations = [elem['entity'] for elem in entities.organizations]

                keywords = nlp_toolkit.extract_keywords(wiki_chunks)
                keywords = [[kw[0] for kw in lst if kw[1]>=0.75] for lst in keywords]
                keywords = list(set([x for xs in keywords for x in xs]))

                readabilities = nlp_toolkit.texts_readability_fog(wiki_chunks)

                common_structure ={'source_id':source_id,
                                'source_title':source_title,
                                'wiki_categories':wiki_categories,
                                'wiki_sections':wiki_sections,
                                'personalia':personalia,
                                'locations':locations,
                                'organizations':organizations,
                                'keywords':keywords}

                common_structure.update(wiki_infobox_data)

                chunk_structures = []
                for chunk_id, (chunk_text, readability) in enumerate(zip(wiki_chunks, readabilities)):
                    chunk_struct = common_structure.copy()
                    chunk_struct.update({'chunk_id':chunk_id,
                                        'chunk_text':chunk_text,
                                        'readability':readability})
                    
                    chunk_structures.append(chunk_struct)

    mongodb_client.bulk_upsert(mongodb_batch)
