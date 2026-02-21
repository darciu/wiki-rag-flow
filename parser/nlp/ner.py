import logging
import re
from pathlib import Path
from statistics import mean

import stanza  # type: ignore
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from parser.entities import NEREntities

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HerbertNERClient:
    _pipeline = None

    def __init__(self):
        self.model_checkpoint = "pczarnik/herbert-base-ner"
        self.model_dir = Path("models") / "herbert-base-ner"

    def _get_pipeline(self):
        if HerbertNERClient._pipeline is None:
            required_files = [
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
            ]

            complete_dir_and_files = self.model_dir.exists() and all(
                (self.model_dir / f).exists() for f in required_files
            )
            if not complete_dir_and_files:
                logger.info(
                    f"Load model and tokenizer from remote: {self.model_checkpoint}..."
                )

                if self.model_dir.exists():
                    for item in sorted(self.model_dir.rglob("*"), reverse=True):
                        item.unlink() if item.is_file() else item.rmdir()
                else:
                    self.model_dir.mkdir(parents=True, exist_ok=True)

                    tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
                    model = AutoModelForTokenClassification.from_pretrained(
                        self.model_checkpoint
                    )
                    tokenizer.save_pretrained(self.model_dir)
                    model.save_pretrained(self.model_dir)

                    HerbertNERClient._pipeline = pipeline(
                        "ner", model=model, tokenizer=tokenizer
                    )
            else:
                logger.info(f"Loading model from local directory: {self.model_dir}...")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
                    model = AutoModelForTokenClassification.from_pretrained(
                        self.model_dir
                    )

                    HerbertNERClient._pipeline = pipeline(
                        "ner", model=model, tokenizer=tokenizer
                    )
                except Exception as e:
                    logger.info(
                        f"Error while loading: {e}. Try delete local directory manually: {self.model_dir}"
                    )
                    raise

        return HerbertNERClient._pipeline

    def extract_raw_entities(self, texts: list[str]) -> list[list[dict]]:
        """Extract all entities from given list of texts using batch processing"""

        ner_pipeline = self._get_pipeline()
        return ner_pipeline(texts)

    def group_entities(self, ner_output: list[dict]) -> dict:
        """Group NERs withing three categories: PER, LOC, ORG"""

        entities: dict[str, list[dict]] = {"PER": [], "LOC": [], "ORG": []}

        current_entity: list[str] = []
        current_type = None
        current_entity_score = []

        for token in ner_output:
            tag = token["entity"]
            word = token["word"].replace("</w>", " ")
            score = float(token["score"])

            if tag.startswith("B-"):
                if current_entity and current_type:
                    entities[current_type].append(
                        {
                            "entity": "".join(current_entity),
                            "score": mean(current_entity_score),
                        }
                    )
                current_type = tag[2:]
                current_entity = [word]
                current_entity_score = [score]

            elif tag.startswith("I-") and current_type == tag[2:]:
                current_entity.append(word)
                current_entity_score.append(score)

            else:
                if current_entity and current_type:
                    entities[current_type].append(
                        {
                            "entity": "".join(current_entity),
                            "score": mean(current_entity_score),
                        }
                    )
                current_entity = []
                current_entity_score = []
                current_type = None

        # last remaining entity
        if current_entity and current_type:
            entities[current_type].append(
                {"entity": "".join(current_entity), "score": mean(current_entity_score)}
            )

        return entities

    def fix_spacing_full_names(self, full_name: str) -> str:
        """Remove unnecessary spacing"""

        # tokenize string
        tokens = re.findall(r"\b\w+\b", full_name)

        if not tokens:
            return full_name

        result = tokens[0]
        for token in tokens[1:]:
            if token[0].isupper():
                result += " " + token
            else:
                result += token

        return result.strip()

    def parse_entities(self, texts: list[str]) -> list[NEREntities]:
        """Combined logic of extracting, cleaning and parsing NER entities"""

        batch_raw_entities = self.extract_raw_entities(texts)
        results = []

        for raw_entities in batch_raw_entities:
            grouped_entities = self.group_entities(raw_entities)

            personalia = [
                {
                    "entity": self.fix_spacing_full_names(elem["entity"]),
                    "score": elem["score"],
                }
                for elem in grouped_entities["PER"]
            ]
            locations = [
                {
                    "entity": self.fix_spacing_full_names(elem["entity"]),
                    "score": elem["score"],
                }
                for elem in grouped_entities["LOC"]
            ]
            organizations = [
                {
                    "entity": self.fix_spacing_full_names(elem["entity"]),
                    "score": elem["score"],
                }
                for elem in grouped_entities["ORG"]
            ]

            results.append(NEREntities(personalia, locations, organizations))
        return results


class StanzaNERClient:
    _nlp_stanza = None

    def __init__(self):
        self.model_dir = Path("models") / "stanza"

    def _get_model(self):
        if self._nlp_stanza is None:
            required_files = [
                "resources.json",
            ]

            complete_dir_and_files = self.model_dir.exists() and all(
                (self.model_dir / f).exists() for f in required_files
            )
            if not complete_dir_and_files:
                if self.model_dir.exists():
                    for item in sorted(self.model_dir.rglob("*"), reverse=True):
                        item.unlink() if item.is_file() else item.rmdir()
                else:
                    logger.info(
                        f"Model is downloaded from external resource to location {self.model_dir}"
                    )
                    stanza.download("pl", model_dir=str(self.model_dir))
            else:
                logger.info(f"Model already exists in location: {self.model_dir}")

            StanzaNERClient._nlp_stanza = stanza.Pipeline(
                "pl",
                processors="tokenize,ner",
                dir=str(self.model_dir),
            )
            logger.info("Model has been loaded")

        return StanzaNERClient._nlp_stanza

    def extract_raw_entities(self, texts: list[str]):
        """Extract all entities from given text"""

        ner_model = self._get_model()
        return ner_model(texts)

    def filter_entities(self, document, pos_type: str):
        """Filter entities with pos_type: persName, placeName or orgName"""

        findings = []
        current = []

        for sentence in document.sentences:
            for token in sentence.tokens:
                ner_tag = token.ner
                text = token.text

                if ner_tag == f"B-{pos_type}":
                    current.append(text)
                elif ner_tag == f"I-{pos_type}":
                    current.append(text)
                elif ner_tag == f"E-{pos_type}":
                    current.append(text)
                    full = " ".join(current)
                    findings.append(full)
                    current = []
                elif ner_tag == f"S-{pos_type}":
                    findings.append(text)

        return findings

    def parse_entities(self, texts: list[str]) -> list[NEREntities]:
        """
        Combined logic of extracting, cleaning and parsing NER entities.
        Stanza in fact does not have batch processing, so process is iterating as single
        model inferences
        """
        ner_model = self._get_model()
        results = []
        for text in texts:
            if not text or not text.strip():
                results.append(NEREntities([], [], []))
                continue

            doc = ner_model(text)

            mapping = {"persName": [], "placeName": [], "orgName": []}

            for ent in doc.entities:
                if ent.type in mapping:
                    mapping[ent.type].append({"entity": ent.text, "score": 1.0})

            results.append(
                NEREntities(
                    personalia=mapping["persName"],
                    locations=mapping["placeName"],
                    organizations=mapping["orgName"],
                )
            )
        return results
