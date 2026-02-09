from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
    

    def extract_keywords(self, texts: List[str], batch_size=8, max_input_length=512,
        max_new_tokens=32, num_beams=4, no_repeat_ngram_size=3):
        model, tokenizer = self._get_model_tokenizer()
        task_prefix = "Keywords: "
        results = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = [task_prefix + t for t in texts[i:i+batch_size]]
                inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_input_length,
                return_tensors="pt" 
                )
                outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_new_tokens=max_new_tokens
                )
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                results.extend(decoded)
        return results