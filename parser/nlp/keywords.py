from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import torch
from typing import List

class VLT5KeywordsClient:
    _tokenizer = None
    _model = None

    def __init__(self):
        self.model_dir = Path("models") / "vlt5"
        self.model_checkpoint = "Voicelab/vlt5-base-keywords"

    def _get_model_tokenizer(self):
        if VLT5KeywordsClient._tokenizer is None or VLT5KeywordsClient._model is None:
            
            
            required_files = [
                "config.json", 
                "model.safetensors", 
                "tokenizer.json", 
                "tokenizer_config.json",
                "generation_config.json",
            ]
            
            complete_dir_and_files = self.model_dir.exists() and all((self.model_dir / f).exists() for f in required_files)

            if not complete_dir_and_files:
                print(f"Load model and tokenizer from remote: {self.model_checkpoint}...")
                
                if self.model_dir.exists():
                    for item in sorted(self.model_dir.rglob("*"), reverse=True):
                        item.unlink() if item.is_file() else item.rmdir()
                else:
                    self.model_dir.mkdir(parents=True, exist_ok=True)
                
                tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
                model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint)
                
                self.model_dir.mkdir(parents=True, exist_ok=True)
                tokenizer.save_pretrained(str(self.model_dir))
                model.save_pretrained(str(self.model_dir))
                
                VLT5KeywordsClient._tokenizer = tokenizer
                VLT5KeywordsClient._model = model
            else:
                print(f"Load model and tokenizer from local directory: {self.model_dir}")
                try:
                    VLT5KeywordsClient._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
                    VLT5KeywordsClient._model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_dir))
                except Exception as e:
                    print(f"Error while loading: {e}. Try delete local directory manually: {self.model_dir}")
                    raise

            print("Model and tokenizer has been loaded successfully.")

        return VLT5KeywordsClient._model, VLT5KeywordsClient._tokenizer
    

    def extract_keywords(self, texts: List[str]):
        model, tokenizer = self._get_model_tokenizer()
        batch_size = 8
        task_prefix = "Keywords: "
        result = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = [task_prefix + t for t in texts[i:i+batch_size]]
                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt" 
                )
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    max_new_tokens=36
                )
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                result.extend(decoded)
        return [[(elem,1.0) for elem in row.split(',')] for row in result]


class KeyBERTKeywordsClient:
    _model = None

    def __init__(self):
        self.model_dir = Path("models") / "keybert-st-paraphrase"
        self.model_checkpoint = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    def _get_model(self):
        if KeyBERTKeywordsClient._model is None:
            
            
            required_files = [
                "model.safetensors ",
                "sentence_bert_config.json",
                "config_sentence_transformers.json",
                "modules.json",
                "tokenizer_config.json",
                "config.json",
                "tokenizer.json"
            ]
            
            complete_dir_and_files = self.model_dir.exists() and all((self.model_dir / f).exists() for f in required_files)

            if not complete_dir_and_files:
                print(f"Load model from remote: {self.model_checkpoint}...")
                
                if self.model_dir.exists():
                    for item in sorted(self.model_dir.rglob("*"), reverse=True):
                        item.unlink() if item.is_file() else item.rmdir()
                else:
                    self.model_dir.mkdir(parents=True, exist_ok=True)
                
                st_model = SentenceTransformer(self.model_checkpoint)
                model = KeyBERT(model=st_model)
                self.model_dir.mkdir(parents=True, exist_ok=True)
                st_model.save(self.model_dir)

                KeyBERTKeywordsClient._model = model
            else:
                print(f"Load model from local directory: {self.model_dir}")
                try:
                    st_model = SentenceTransformer(self.model_dir)
                    model = KeyBERT(model=st_model)
                    KeyBERTKeywordsClient._model = model
                except Exception as e:
                    print(f"Error while loading: {e}. Try delete local directory manually: {self.model_dir}")
                    raise

            print("Model has been loaded successfully.")

        return KeyBERTKeywordsClient._model
    
    def extract_keywords(self, texts: List[str]):
        model = self._get_model()
        output = model.extract_keywords(
        texts,
        keyphrase_ngram_range=(1, 3),  
        use_mmr=True,
        diversity=0.6,
        top_n=2
        )

        return output
        