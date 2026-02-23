import ollama
from typing import List
from pydantic import BaseModel, Field, model_validator
from enum import Enum

from instructor.exceptions import InstructorRetryException
from instructor.core.client import Instructor

ROUTE_QUERY_SYSTEM_PROMPT = """Jesteś inteligentnym klasyfikatorem zapytań dla systemu opartego na bazie wiedzy z Wikipedii.
    Twoim jedynym zadaniem jest ocena wejścia użytkownika i przypisanie go do jednej z trzech kategorii, oraz zwrócenie poprawnego obiektu JSON.
    KATEGORIE:
    
    1. "RAG_SEARCH" - Każda wypowiedź użytkownika (pytanie, prośba lub polecenie) dotycząca JAKICHKOLWIEK faktów, zjawisk, historii, kultury, biografii czy nauki lub innych dziedzin wiedzy, która wymaga szczegółowej wiedzy. Przykłady: "Kto wygrał bitwę pod Waterloo?", "Wymień wszystkich królów Polski", "Opisz proces fotosyntezy". Wybierz tę opcję, gdy temat jest złożony i wymaga wgłębienia się w artykuły Wikipedii lub nie jesteś pewien czy model będzie w stanie odpowiedzieć na pytanie samodzielnie.
    2. "DIRECT" - Używaj TYLKO I WYŁĄCZNIE wtedy, gdy wypowiedź ABSOLUTNIE NIE WYMAGA opierania się na faktach, historii, nauce ani wiedzy encyklopedycznej. Kategoria ta jest zarezerwowana dla zapytań o samego bota (np. "Kim jesteś?", "Jak działasz?"), prostych zadań kreatywnych (np. "Napisz mi wierszyk", "Opowiedz żart") lub podstawowej logiki ("Ile to 2+2?"). Jeśli zapytanie dotyka jakiejkolwiek dziedziny wiedzy o świecie rzeczywistym – użycie tej flagi jest SUROWO ZABRONIONE (wybierz wtedy RAG_SEARCH).
    3. "CLARIFY" - Wypowiedź pozbawiona jasnego celu informacyjnego. Zalicza się do tego: zwykłe przywitania ("Cześć", "Hej"), niezrozumiały bełkot ("asdsdf"), ucięte zdania, luźne opinie lub wypowiedzi zbyt ogólnikowe, by można było na nie merytorycznie odpowiedzieć (np. "co o tym myślisz?", "a on co zrobił?"). Zwróć tę flagę, gdy musisz dopytać użytkownika, czego dokładnie szuka.

    ### ZASADY DLA KATEGORII 'CLARIFY':
    Jeśli wybierasz CLARIFY, musisz sformułować pomocną odpowiedź w polu 'clarify_message', nie zostawiaj tego pola pustego.
    1. WYKORZYSTAJ KONTEKST: Jeśli użytkownik pyta o coś ogólnego (np. "kto jest królem?"), wykorzystaj tą część informacji w pytaniu doprecyzującym. Odpisz: "Chętnie pomogę, ale potrzebuję doprecyzowania: o jaki kraj lub okres historyczny pytasz?".
    2. BĄDŹ NATURALNY: Unikaj powtarzania frazy "Jestem botem Wikipedii". Zamiast tego reaguj na to, co napisał użytkownik.
    3. REAKCJA NA BEŁKOT: Jeśli wpis to losowe znaki, poproś o ponowne zadanie pytania w sposób przyjazny i luźny.
    4. TYLKO PO POLSKU: zadane przez ciebie pytanie musi być wyłącznie w języku polskim, w prostych słowach.

    ### PRZYKŁADY (FEW-SHOT):
    Użytkownik: "kiedy on zmarł?"
    JSON: {"user_route": "CLARIFY", "clarify_message": "Z chęcią sprawdzę tę datę, ale powiedz mi proszę, o jaką postać Ci chodzi?"}

    Użytkownik: "hej, co tam?"
    JSON: {"user_route": "CLARIFY", "clarify_message": "Cześć! Jestem gotowy do przeszukania Wikipedii dla Ciebie. O czym chciałbyś się dziś dowiedzieć?"}

    Użytkownik: "asdfghjkl"
    JSON: {"user_route": "CLARIFY", "clarify_message": "Nie rozumiem. Czy mógłbyś zadań pytanie ponownie?"}

    Użytkownik: "bitwa pod Grunwaldem"
    JSON: {"user_route": "RAG_SEARCH", "clarify_message": null}
    """

DIRECT_ANSWER_SYSTEM_PROMPT = """Jesteś modelem językowym o ogromnej wiedzy ogólnej. 
    Odpowiadaj konkretnie na zadane pytanie.
    
    ## ZASADY:
    1. Jeśli znasz odpowiedź: Zwróć ją w polu 'answer' oraz ustaw 'knows_answer' na True.
    2. Jeśli NIE JESTEŚ PEWIEN lub pytanie dotyczy faktów, na których model nie był trenowany (np. wydarzenia z wczoraj): 
       W polu 'answer' napisz: Niestety, moja wiedza wewnętrzna nie obejmuje szczegółów na ten temat, a pytanie nie wymagało przeszukania bazy artykułów Wikipedii.
       Ustaw 'knows_answer' na False oraz 'confidence_score' blisko 0.
    3. Nigdy nie zmyślaj faktów (nie halucynuj).
    4. Zawsze odpowiedzi udzielaj wyłącznie w języku polskim.
    """

CLEAN_DATA_SYSTEM_PROMPT= """Jesteś edytorem tekstu. Masz za zadanie:
    1. Usunąć z tekstu zbędne ozdobniki, takie jak przywitania ("Hej!", "Cześć","Jak się masz")
    2. Zastąpić zwroty grzecznościowe takie jak "proszę", "czy mógłbyś coś zrobić", neutralną formą czasownika "zrób to".
    3. Rozbić złożone zdania na proste i zwrócić każde tak rozbite zdanie jako kolejny element listy.

    ZASADY KRYTYCZNE:
    1. ZACHOWAJ SENS: Nie wolno Ci dodawać nowych informacji, zmieniać podmiotu ani modyfikować intencji pytania.
    2. BRAK FANTZJOWANIA: Jeśli zapytanie dotyczy "bitwy pod Grunwaldem", nie twórz wariacji o "wojnie z Zakonem", jeśli oryginalne zdanie o tym nie wspomina. Nie dodawaj żadnego dodatkowego kontekstu.
    3. POPRAWNOŚĆ GRAMATYCZNA I JĘZYKOWA: Przemyśl czy wygenerowany przez Ciebie tekst jest poprawny zarówno gramatycznie jak i pod względem językowym według zasad języka polskiego.
    4. NIE ZAMIENIAJ PYTAŃ NA ZDANIA OZNAJMIAJĄCE: Niedozwolone jest zamienianie pytań na zdania oznajmiające i odwrotnie. Jeśli coś jest pytaniem, musi nim pozostać.
    5. NIE ODPOWIADAJ NA POLECENIA CZY PYTANIA: Jeśli w tekście jest polecenie lub pytanie, nie możesz na nie odpowiadać. Twoim celem jest jedynie czyszczenie i dzielenie danych. 
    """

PARAPHASE_SENTENCE_SYSTEM_PROMPT = """Jesteś analitykiem lingwistycznym specjalizującym się w systemach wyszukiwania i przetwarzania informacji.

    TWOJE ZADANIE:
    Wygeneruj od 1 do 3 alternatywnych wersji podanego zdania, które są identyczne pod względem merytorycznym, ale różnią się konstrukcją gramatyczną lub słownictwem.
    
    ZASADY KRYTYCZNE:
    1. ZACHOWAJ SENS: Nie wolno Ci dodawać nowych informacji, zmieniać podmiotu ani modyfikować intencji pytania.
    2. SYNONYMY I STRUKTURA: Używaj synonimów (np. "zmarł" zamiast "odszedł"), zamieniaj stronę czynną na bierną i zmieniaj szyk zdania.
    3. BRAK FANTZJOWANIA: Jeśli zapytanie dotyczy "bitwy pod Grunwaldem", nie twórz wariacji o "wojnie z Zakonem", jeśli oryginalne zdanie o tym nie wspomina.
    4. POPRAWNOŚĆ GRAMATYCZNA I JĘZYKOWA: Przemyśl czy wygenerowany przez Ciebie tekst jest poprawny zarówno gramatycznie jak i pod względem językowym według zasad języka polskiego.
    5. FORMAT: Musisz zwrócić wyłącznie obiekt JSON z kluczem 'expanded_queries'.


    PRZYKŁADY:
    Użytkownik: "Kto wynalazł telefon?"
    JSON: {"expanded_queries": ["Przez kogo został wynaleziony telefon?", "Twórca wynalazku telefonu", "Kto jest autorem technologii telefonicznej?"]}

    Użytkownik: "Wymień skutki bitwy pod Waterloo."
    JSON: {"expanded_queries": ["Jakie były konsekwencje starcia pod Waterloo?", "Bitwa pod Waterloo i jej następstwa", "Podaj rezultaty bitwy pod Waterloo."]}
    """




class RouteType(str, Enum):
    RAG_SEARCH = "RAG_SEARCH"
    DIRECT = "DIRECT"
    CLARIFY = "CLARIFY"

class QueryDecision(BaseModel):
    user_route: RouteType = Field(
        ..., 
        description="Przypisz zapytanie do jednej z trzech kategorii."
    )
    clarify_message: str | None = Field(
        default=None, 
        description="""Wypełnij to pole tylko jeśli wartość pola user_route to CLARIFY. W takim przypadku nigdy nie zostawiaj tego pola pustego.
                    Dla user_route RAG_SEARCH i DIRECT zostaw to pole puste (null)."""
    )
    @model_validator(mode='after')
    def validate_clarify_message(self) -> 'QueryDecision':
        if self.user_route == RouteType.CLARIFY:
            if not self.clarify_message or not self.clarify_message.strip():
                raise ValueError(
                    """BŁĄD KRYTYCZNY: Skoro user_route to CLARIFY, pole clarify_message nie może być puste. Musisz wygenerować wiadomość dopytującą użytkownika."""
                )
        return self
    
class DirectQuestion(BaseModel):
    answer: str = Field(
        ..., 
        description="Merytoryczna i dosyć zwięzła odpowiedź na pytanie użytkownika."
    )
    knows_answer: bool = Field(
        ..., 
        description="Czy model LLM posiada wystarczającą wiedzę, aby odpowiedzieć na to pytanie? True jeśli tak, False jeśli musi przyznać, że nie wie."
    )
    confidence_score: float = Field(
        ..., 
        ge=0.0, le=1.0, 
        description="Ocena pewności odpowiedzi od 0.0 do 1.0."
    )

    @model_validator(mode='after')
    def validate_content(self) -> 'DirectQuestion':
        if not self.answer or len(self.answer.strip()) < 5:
            raise ValueError("Pole answer musi zawierać sensowną treść, nawet jeśli przyznajesz, że nie potrafisz udzielić odpowiedzi.")

        if not self.knows_answer and self.confidence_score > 0.5:
             raise ValueError("Niespójność: knows_answer jest False, ale confidence_score jest wysoki.")
             
        return self

    @model_validator(mode='after')
    def force_honesty(self) -> 'DirectQuestion':
        is_unsure = not self.knows_answer or self.confidence_score < 0.5
        
        if is_unsure and len(self.answer) > 150:
            raise ValueError(
                "Twoja pewność jest niska, ale odpowiedź jest zbyt długa. "
                "Zredukuj odpowiedź do krótkiej, elastycznej informacji o braku wiedzy."
            )
            
        if self.knows_answer and self.confidence_score < 0.5:
            raise ValueError(
                "Niespójność: twierdzisz, że znasz odpowiedź, ale Twoja pewność (confidence) jest niska. "
                "Zmień knows_answer na False i podaj komunikat o braku wiedzy."
            )
            
        return self


def route_query(client: Instructor, user_query: str, model_name: str) -> QueryDecision:
    try:
        system_prompt = ROUTE_QUERY_SYSTEM_PROMPT
        
        decision = client.chat.completions.create(
            model=model_name,
            response_model=QueryDecision,
            max_retries=3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
        )
        return decision
    except InstructorRetryException as e:
        print(f"Ostrzeżenie: Model nie wygenerował poprawnej wiadomości po {e.n_attempts} próbach.")

        return QueryDecision(
            user_route=RouteType.CLARIFY,
            clarify_message="Jestem botem Wikipedii. Twoje pytanie jest dla mnie trochę niejasne. Czy mógłbyś je sformułować inaczej lub podać więcej szczegółów?"
        )

class QueryCleaner(BaseModel):
    normalized_queries: List[str] = Field(
        ..., 
        description="Lista uproszczonych, jednoznacznych zdań twierdzących lub pytań."
    )

class QueryExpander(BaseModel):
    expanded_queries: List[str] = Field(
        ..., 
        description="Lista 1-3 różnych parafraz zapytania bazowego."
    )

def direct_query(client: Instructor, user_query: str, model_name: str) -> DirectQuestion:
    
    try:
        return client.chat.completions.create(
            model=model_name,
            response_model=DirectQuestion,
            max_retries=3,
            messages=[
                {"role": "system", "content": DIRECT_ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
        )
    except InstructorRetryException as e:
        return DirectQuestion(
            answer="Przepraszam, ale nie jestem w stanie odpowiedzieć na to pytanie.",
            knows_answer=False,
            confidence_score=0.0
        )


def simplify_clean_query(client: Instructor, user_query: str, model_name: str) -> QueryCleaner:
    
    try:
        return client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            response_model=QueryCleaner,
            messages=[{"role": "system", "content": CLEAN_DATA_SYSTEM_PROMPT},
                    {"role": "user", "content": user_query}]
        )
    
    except InstructorRetryException as e:
        print(f"Ostrzeżenie: Model nie wygenerował poprawnej wiadomości po {e.n_attempts} próbach.")
        return QueryCleaner(
            normalized_queries=[user_query]
        )


def paraphase_query(client: Instructor, user_query: str, model_name: str) -> QueryExpander:

    try:
        return client.chat.completions.create(
            model=model_name,
            response_model=QueryExpander,
            temperature=0.0,
            max_retries=3,
            messages=[{"role": "system", "content": PARAPHASE_SENTENCE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_query}]
        )

    except InstructorRetryException as e:
        print(f"Ostrzeżenie: Model nie wygenerował poprawnej wiadomości po {e.n_attempts} próbach.")
        return QueryExpander(
            expanded_queries=[]
        )