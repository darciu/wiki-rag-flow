from tqdm import tqdm
import xml.etree.ElementTree as ET
import requests
import re
import logging
import hashlib
from pathlib import Path
import bz2

logger = logging.getLogger(__name__)

def get_latest_dumpstatus_url():
    rss_url = "https://dumps.wikimedia.org/plwiki/latest/plwiki-latest-pages-articles-multistream-index.txt.bz2-rss.xml"
    
    try:
        logger.info(f'Download wiki metadata from: {rss_url}')
        response = requests.get(rss_url, timeout=10)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        
        # search all tree (.//) for tag item then tag link
        item_link = root.find(".//item/link").text
        
        # find eight numbers one afer another
        date_match = re.search(r'(\d{8})', item_link)
        if not date_match:
            return None
        
        dump_date = date_match.group(1)
        
        return f"https://dumps.wikimedia.org/plwiki/{dump_date}/dumpstatus.json"
        
    except Exception as e:
        logger.exception(f'Exception has occured: {e}')
        raise
    
def fetch_dumpstatus(dumpstatus_url):
    try:
        logger.info(f'Download wiki dumpstatus from: {dumpstatus_url}')
        response = requests.get(dumpstatus_url, timeout=10)
        response.raise_for_status()
        
        return response.json()
    
    except requests.exceptions.HTTPError as e:
        logger.error(f'HTTP error has occured. Status {response.status_code}: {e}')
        raise
    except requests.exceptions.JSONDecodeError as e:
        logger.error(f'JSON format error {e}')
        raise
    except Exception as e:
        logger.exception(f'Exception has occured: {e}')
        raise

def is_dump_done(articlesmultistreamdump):
    if articlesmultistreamdump.get('status','') == 'done':
        logger.info(f'Articles multistream dump is ready to download')
        return True
    else:
        logger.error(f'Articles multistream dump is not ready to download')

def get_download_urls(articlesmultistreamdump):
    multistream_urls = []
    for _, val in articlesmultistreamdump['files'].items():

        multistream_urls.append({
            'url':"https://dumps.wikimedia.org/"+val['url'],
            'md5':val['md5']
        })
    return multistream_urls

def check_md5(filepath, wiki_md5):
    with open(filepath, "rb") as f:
        digest = hashlib.file_digest(f, "md5")
        actual_md5 = digest.hexdigest()

    return actual_md5 == wiki_md5


def pair_wiki_files(folder_path):

    index = {}
    multistream = {}
    # example -p1p187037.bz2
    pattern = re.compile(r'-(p\d+p\d+)\.bz2$')
    
    for file in Path(folder_path).glob("*.bz2"):
        match = pattern.search(file.name)
        if match:
            file_id = match.group(1)
            if 'multistream-index' in file.name:
                index[file_id] = file.name
            else:
                multistream[file_id] = file.name

    pairs = []
    for filename in index:
        if filename in multistream:
            pairs.append({
                'index':index[filename],
                'multistream':multistream[filename]
            })
        else:
            logging.error(f'Filename {filename} has no pair')
            raise FileNotFoundError(f"Filename {filename} has no pair")
    return pairs


def get_unique_indices(filepath):
    logging.info(f"Get unique indices for: {filepath}")
    offsets = set()
    with bz2.open(filepath, 'rt', encoding='utf-8') as source:
        for line in source:
            parts = line.strip().split(':')
            if len(parts) >= 3:
                try:
                    offset = int(parts[0])
                    offsets.add(offset)
                except ValueError:
                    print(f'Error. Offset is not numeric: {parts[0]}')
                    continue

    return sorted(list(offsets))


def get_full_block(filepath, byte_offset):
    with open(filepath, 'rb') as f:
        f.seek(byte_offset)
        
        decompressor = bz2.BZ2Decompressor()
        parts = []
        
        while not decompressor.eof:
            chunk = f.read(64 * 1024) # chunk is 64KB
            if not chunk:
                break
            try:
                parts.append(decompressor.decompress(chunk))
            except EOFError:
                break 
                
        return b"".join(parts).decode('utf-8')
    
def get_title_id_from_page(page):
    tmp = page.partition('<revision>')[0]
    title_start = tmp.find('<title>') + 7
    title_end = tmp.find('</title>')
    title = tmp[title_start:title_end]

    id_start = tmp.find('<id>') + 4
    id_end = tmp.find('</id>')
    page_id = tmp[id_start:id_end]
    return title, page_id


def multistream_to_mongodb(mongodb_client, filepath, indices):
    logger.info(f"Upserting records to MongoDB scraper_db/wikipedia from file: {filepath}")
    batch = []
    batch_size = 30
    for offset in tqdm(indices):
        full_xml_block = get_full_block(filepath, offset)
        pages = full_xml_block.split('<page>')
        for page in pages:
            if not page.isspace():
                title, page_id = get_title_id_from_page(page)
                if page_id:
                    load = {
                        "_id": page_id,
                        "title": title,
                        "content": page
                    }
                    batch.append(load)
                if len(batch) >= batch_size:
                    mongodb_client.bulk_upsert("wikipedia", batch)
                    batch = []

    if batch:
        mongodb_client.bulk_upsert("wikipedia", batch)
    logger.info(f"Finished upserting file: {filepath}")