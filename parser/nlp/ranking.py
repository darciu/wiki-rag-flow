

import logging
from pathlib import Path
from sentence_transformers import CrossEncoder
from typing import List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CrossEncoderMSMarcoClient:
    _model = None

    def __init__(self):
        self.model_checkpoint = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.model_dir = Path("models") / "cross-encoder-ms-marco-minilm"

    def _get_model(self):
        if CrossEncoderMSMarcoClient._model is None:
            required_files = [
                "tokenizer.json",
                "tokenizer_config.json",
                "config.json",
                "model.safetensors",
            ]

            complete_dir_and_files = self.model_dir.exists() and all(
                (self.model_dir / f).exists() for f in required_files
            )

            if not complete_dir_and_files:
                logger.info(f"Load model from remote: {self.model_checkpoint}...")

                if self.model_dir.exists():
                    for item in sorted(self.model_dir.rglob("*"), reverse=True):
                        item.unlink() if item.is_file() else item.rmdir()
                else:
                    self.model_dir.mkdir(parents=True, exist_ok=True)

                model = CrossEncoder(self.model_checkpoint, max_length=512)

                self.model_dir.mkdir(parents=True, exist_ok=True)
                model.save(str(self.model_dir))

                CrossEncoderMSMarcoClient._model = model
            else:
                logger.info(f"Load model from local directory: {self.model_dir}")
                try:
                    CrossEncoderMSMarcoClient._model = CrossEncoder(
                        str(self.model_dir)
                    )
                except Exception as e:
                    logger.info(
                        f"Error while loading: {e}. Try delete local directory manually: {self.model_dir}"
                    )
                    raise

            logger.info("Model has been loaded successfully.")

        return CrossEncoderMSMarcoClient._model

    def rank(self, query: str, texts: List[str]) -> List[float]:
        """
        Ranks text candidates against a query using a cross-encoder model.

        Args:
            query: The search query string.
            texts: A list of document strings to be scored.

        Returns:
            A list of relevance scores (higher is better) for each text.
        """
        model = self._get_model()
        pairs = [[query, text] for text in texts]
        scores = model.predict(pairs)

        return scores.tolist()
    


class UnicampMiniLMMultiClient:
    _model = None

    def __init__(self):
        self.model_checkpoint = "unicamp-dl/mMiniLM-L6-v2-mmarco-v2"
        self.model_dir = Path("models") / "unicamp-ms-marco-minilm-multilangual"

    def _get_model(self):
        if UnicampMiniLMMultiClient._model is None:
            required_files = [
                "tokenizer.json",
                "tokenizer_config.json",
                "config.json",
                "model.safetensors",
            ]

            complete_dir_and_files = self.model_dir.exists() and all(
                (self.model_dir / f).exists() for f in required_files
            )

            if not complete_dir_and_files:
                logger.info(f"Load model from remote: {self.model_checkpoint}...")

                if self.model_dir.exists():
                    for item in sorted(self.model_dir.rglob("*"), reverse=True):
                        item.unlink() if item.is_file() else item.rmdir()
                else:
                    self.model_dir.mkdir(parents=True, exist_ok=True)

                model = CrossEncoder(self.model_checkpoint, max_length=512)

                self.model_dir.mkdir(parents=True, exist_ok=True)
                model.save(str(self.model_dir))

                UnicampMiniLMMultiClient._model = model
            else:
                logger.info(f"Load model from local directory: {self.model_dir}")
                try:
                    UnicampMiniLMMultiClient._model = CrossEncoder(
                        str(self.model_dir)
                    )
                except Exception as e:
                    logger.info(
                        f"Error while loading: {e}. Try delete local directory manually: {self.model_dir}"
                    )
                    raise

            logger.info("Model has been loaded successfully.")

        return UnicampMiniLMMultiClient._model

    def rank(self, query: str, texts: List[str]) -> List[float]:
        """
        Ranks text candidates against a query using a cross-encoder model.

        Args:
            query: The search query string.
            texts: A list of document strings to be scored.

        Returns:
            A list of relevance scores (higher is better) for each text.
        """
        model = self._get_model()
        pairs = [[query, text] for text in texts]
        scores = model.predict(pairs)

        return scores.tolist()