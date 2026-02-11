from pathlib import Path
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from itertools import groupby

from semantic_chunkers import StatisticalChunker
from semantic_router.encoders import HuggingFaceEncoder


class LangchainSplitterClient:
    _tokenizer = None

    def __init__(self):
        self.model_checkpoint = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        self.model_dir = Path("models") / "langchain-splitter"

    def _get_tokenizer(self):
        if LangchainSplitterClient._tokenizer is None:
            
            required_files = [
                "tokenizer.json", 
                "tokenizer_config.json",
            ]
            
            complete_dir_and_files = self.model_dir.exists() and all((self.model_dir / f).exists() for f in required_files)

            if not complete_dir_and_files:
                print(f"Load tokenizer from remote: {self.model_checkpoint}...")
                
                if self.model_dir.exists():
                    for item in sorted(self.model_dir.rglob("*"), reverse=True):
                        item.unlink() if item.is_file() else item.rmdir()
                else:
                    self.model_dir.mkdir(parents=True, exist_ok=True)
                
                tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
                
                self.model_dir.mkdir(parents=True, exist_ok=True)
                tokenizer.save_pretrained(str(self.model_dir))
                
                LangchainSplitterClient._tokenizer = tokenizer
            else:
                print(f"Load tokenizer from local directory: {self.model_dir}")
                try:
                    LangchainSplitterClient._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
                except Exception as e:
                    print(f"Error while loading: {e}. Try delete local directory manually: {self.model_dir}")
                    raise

            print("Tokenizer has been loaded successfully.")

        return LangchainSplitterClient._tokenizer
    
    def chunk_texts(self, texts, max_tokens):
        tokenizer = self._get_tokenizer()
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=max_tokens,
            chunk_overlap=50
        )

        ids = list(range(len(texts)))
        metadatas = [{"id": i} for i in ids]
        docs = splitter.create_documents(texts, metadatas=metadatas)

        result = [[d.page_content for d in group] 
                    for _, group in groupby(docs, key=lambda d: d.metadata["id"])]
        
        return result
    




class StatisticalChunkerClient:
    _encoder = None

    def __init__(self):
        self.model_checkpoint = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        self.model_dir = Path("models") / "statistical-chunker"

    def _get_encoder(self):
        if StatisticalChunkerClient._encoder is None:
            
            required_files = [
                "config.json", 
                "model.safetensors",
            ]
            
            complete_dir_and_files = self.model_dir.exists() and all((self.model_dir / f).exists() for f in required_files)

            if not complete_dir_and_files:
                print(f"Load encoder from remote: {self.model_checkpoint}...")
                
                if self.model_dir.exists():
                    for item in sorted(self.model_dir.rglob("*"), reverse=True):
                        item.unlink() if item.is_file() else item.rmdir()
                else:
                    self.model_dir.mkdir(parents=True, exist_ok=True)
                
                encoder = HuggingFaceEncoder(name=self.model_checkpoint)
                
                self.model_dir.mkdir(parents=True, exist_ok=True)
                encoder._model.save_pretrained(str(self.model_dir))
                
                StatisticalChunkerClient._encoder = encoder
            else:
                print(f"Load encoder from local directory: {self.model_dir}")
                try:
                    StatisticalChunkerClient._tokenizer = HuggingFaceEncoder(name=str(self.model_dir))
                except Exception as e:
                    print(f"Error while loading: {e}. Try delete local directory manually: {self.model_dir}")
                    raise

            print("Encoder has been loaded successfully.")

        return LangchainSplitterClient._tokenizer
    

    def chunk_texts(self, texts, max_tokens):
        encoder = self._get_encoder()
        chunker = StatisticalChunker(
            encoder=encoder,   
            max_split_tokens=max_tokens,
        )
        chunks = chunker(texts)
        results = [[elem.content for elem in chunk] for chunk in chunks]
        return results

    
