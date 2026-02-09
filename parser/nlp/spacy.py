
import spacy
from typing import List

class SpacyUtils:
    _nlp_spacy = None

    def _get_nlp_spacy(self):
        if self._nlp_spacy is None:
            print("Loading spacy NLP...")
            SpacyUtils._nlp_spacy = spacy.load("pl_core_news_lg")

        return SpacyUtils._nlp_spacy


    def lemmatize_names(self, names: List[str], batch_size: int = 512) -> List[str]:
        """Return the basic forms of a list of names"""

        nlp_spacy = self._get_nlp_spacy()
        disable_components = ["ner", "parser", "senter", "textcat"]
        to_disable = [c for c in disable_components if c in nlp_spacy.pipe_names]
        lemmatized = []
        docs = nlp_spacy.pipe(names, batch_size=batch_size, disable=to_disable)
        for doc in docs:
            lemmatized.append(" ".join([token.lemma_ for token in doc]))

        return lemmatized
    
    def count_syllables_pl(self, word: str) -> int:
        word = word.lower()
        pl_vowels = "aeiouyąęó"
        count = 0
        
        for i in range(len(word)):
            if word[i] in pl_vowels:
                # i before other pl_vowel
                if word[i] == 'i' and i + 1 < len(word) and word[i+1] in pl_vowels:
                    continue
                count += 1
                
        return max(1, count)
    

    def texts_readability_fog(self, texts: list[str], batch_size: int = 100) -> list[float]:
        if not texts:
            return []

        nlp_spacy = self._get_nlp_spacy()
        disable_components = ["ner", "lemmatizer", "textcat"]
        to_disable = [c for c in disable_components if c in nlp_spacy.pipe_names]

        results = []
        for doc in nlp_spacy.pipe(texts, batch_size=batch_size, disable=to_disable):
            if not doc or len(doc.text.strip()) == 0:
                results.append(0.0)
                continue
                
            sentences = list(doc.sents)
            words = [token.text for token in doc if token.is_alpha]
            
            if not sentences or not words:
                results.append(0.0)
                continue

            # hard words with minimum 4 syllabes
            hard_words_count = sum(1 for w in words if self.count_syllables_pl(w) >= 4)
            
            avg_sentence_length = len(words) / len(sentences)
            percentage_hard_words = (hard_words_count / len(words)) * 100
            
            fog_index = 0.4 * (avg_sentence_length + percentage_hard_words)
            
            results.append(round(fog_index, 2))
            
        return results