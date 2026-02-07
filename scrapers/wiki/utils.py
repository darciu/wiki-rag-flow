import bz2
import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import requests
from tqdm import tqdm

if TYPE_CHECKING:
    from backend.db.mongodb.connection import MongoManager

logger = logging.getLogger(__name__)


def get_latest_dumpstatus_url(rss_url) -> str | None:
    """
    Retrieves the latest Wikimedia dump date from RSS and returns the dumpstatus.json URL.
    """

    try:
        logger.info(f"Download wiki metadata from: {rss_url}")
        response = requests.get(rss_url, timeout=10)
        response.raise_for_status()

        root = ET.fromstring(response.content)

        # search all tree (.//) for tag item then tag link
        element = root.find(".//item/link")
        if element is not None:
            item_link = element.text
            # find eight numbers one afer another
            if item_link is not None:
                date_match = re.search(r"(\d{8})", item_link)
            else:
                return None
        else:
            return None

        

        if not date_match:
            return None

        dump_date = date_match.group(1)

        return f"https://dumps.wikimedia.org/plwiki/{dump_date}/dumpstatus.json"

    except Exception as e:
        logger.exception(f"Exception has occured: {e}")
        raise


def fetch_dumpstatus(dumpstatus_url) -> dict[str, Any]:
    """
    Downloads and parses the Wikimedia dumpstatus JSON metadata.
    """
    try:
        logger.info(f"Download wiki dumpstatus from: {dumpstatus_url}")
        response = requests.get(dumpstatus_url, timeout=10)
        response.raise_for_status()

        return cast(dict[str, Any], response.json())

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error has occured. Status {response.status_code}: {e}")
        raise
    except requests.exceptions.JSONDecodeError as e:
        logger.error(f"JSON format error {e}")
        raise
    except Exception as e:
        logger.exception(f"Exception has occured: {e}")
        raise


def is_dump_done(articlesmultistreamdump: dict[str, Any]) -> bool:
    """
    Checks if the Wikipedia articles multistream dump status is 'done'.
    """
    if articlesmultistreamdump.get("status", "") == "done":
        logger.info("Articles multistream dump is ready to download")
        return True

    logger.error("Articles multistream dump is not ready to download")
    return False


def get_download_urls(articlesmultistreamdump: dict[str, Any]) -> list[dict[str, str]]:
    """
    Extracts download URLs and their MD5 checksums from the dump metadata.
    """
    multistream_urls = []
    for _, val in articlesmultistreamdump["files"].items():
        multistream_urls.append(
            {"url": "https://dumps.wikimedia.org/" + val["url"], "md5": val["md5"]}
        )
    return multistream_urls


def check_md5(filepath: str, wiki_md5: str) -> bool:
    """
    Verifies the file integrity by comparing its MD5 hash with the provided checksum.
    """
    path = Path(filepath)
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "md5")
        actual_md5 = digest.hexdigest()

    return actual_md5 == wiki_md5


def pair_wiki_files(folder_path: str) -> list[dict[str, str]]:
    """
    Pairs Wikipedia multistream index files with their corresponding data files inside download folder.
    """

    index = {}
    multistream = {}
    # example -p1p187037.bz2
    pattern = re.compile(r"-(p\d+p\d+)\.bz2$")

    for file in Path(folder_path).glob("*.bz2"):
        match = pattern.search(file.name)
        if match:
            file_id = match.group(1)
            if "multistream-index" in file.name:
                index[file_id] = file.name
            else:
                multistream[file_id] = file.name

    pairs = []
    for filename in index:
        if filename in multistream:
            pairs.append(
                {"index": index[filename], "multistream": multistream[filename]}
            )
        else:
            logging.error(f"Filename {filename} has no pair")
            raise FileNotFoundError(f"Filename {filename} has no pair")
    return pairs


def get_unique_indices(filepath: str) -> list[int]:
    """
    Extracts and sorts unique byte offsets from a Wikipedia multistream index file.
    It is necessary for further reading bz2 files with articles.
    """
    logging.info(f"Get unique indices for: {filepath}")
    offsets = set()
    with bz2.open(filepath, "rt", encoding="utf-8") as source:
        for line in source:
            parts = line.strip().split(":")
            if len(parts) >= 3:
                try:
                    offset = int(parts[0])
                    offsets.add(offset)
                except ValueError:
                    print(f"Error. Offset is not numeric: {parts[0]}")
                    continue

    return sorted(offsets)


def get_full_block(filepath: str, byte_offset: int) -> str | None:
    """
    Extracts and decompresses a single BZ2 block from a specific byte offset (index) in a file.
    """
    path = Path(filepath)
    with path.open("rb") as f:
        f.seek(byte_offset)

        decompressor = bz2.BZ2Decompressor()
        parts = []

        while not decompressor.eof:
            chunk = f.read(64 * 1024)  # chunk is 64KB
            if not chunk:
                break
            try:
                parts.append(decompressor.decompress(chunk))
            except EOFError:
                break
        try:
            return b"".join(parts).decode("utf-8")
        except UnicodeDecodeError as e:
            logging.error(
                f"Unicode decode error in {filepath} at offset {byte_offset}. Skipping block. Error: {e}"
            )
            return None


def get_title_id_from_page(page: str) -> tuple[str, str]:
    """
    Extracts the page title and ID from a Wikipedia XML page (actually page is string) fragment using slicing.
    """
    tmp = page.partition("<revision>")[0]
    title_start = tmp.find("<title>") + 7
    title_end = tmp.find("</title>")
    title = tmp[title_start:title_end]

    id_start = tmp.find("<id>") + 4
    id_end = tmp.find("</id>")
    page_id = tmp[id_start:id_end]
    return title, page_id


def multistream_to_mongodb(
    mongodb_client: MongoManager, filepath: str, indices: list[int]
) -> None:
    """
    Processes a Wikipedia multistream xml blocks and performs bulk upserts to MongoDB.
    """
    logger.info(
        f"Upserting records to MongoDB scraper_db/wikipedia from file: {filepath}"
    )
    batch = []
    batch_size = 30
    for offset in tqdm(indices):
        full_xml_block = get_full_block(filepath, offset)
        if full_xml_block is None:
            continue
        pages = full_xml_block.split("<page>")
        for page in pages:
            if not page.isspace():
                title, page_id = get_title_id_from_page(page)
                if page_id:
                    load = {"_id": page_id, "title": title, "content": page}
                    batch.append(load)
                if len(batch) >= batch_size:
                    mongodb_client.bulk_upsert("wikipedia", batch)
                    batch = []

    if batch:
        mongodb_client.bulk_upsert("wikipedia", batch)
    logger.info(f"Finished upserting file: {filepath}")
