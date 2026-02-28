from typing import List
from pydantic import BaseModel, Field, model_validator
from enum import Enum

from instructor.exceptions import InstructorRetryException
from instructor.core.client import Instructor

from llm.prompts import ROUTE_QUERY_SYSTEM_PROMPT, DIRECT_ANSWER_SYSTEM_PROMPT, CLEAN_DATA_SYSTEM_PROMPT, PARAPHASE_SENTENCE_SYSTEM_PROMPT, FURTHER_QUESTIONS_SYSTEM_PROMPT, RAG_QUERY_SYSTEM_PROMPT
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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

class RAGAnswer(BaseModel):
    is_found: bool = Field(
        description="Czy w dostarczonym kontekście znajduje się odpowiedź na pytanie?"
    )
    answer: str = Field(
        description="Zwięzła i merytoryczna odpowiedź na pytanie użytkownika. Wypełnij tylko w przypadku gdy is_found przyjmuje wartość True."
    )


class RAGQuestions(BaseModel):
    questions: List[str | None] = Field(
        description="Jedno do trzech pytań wygenerowanych na podstawie podanego kontekstu zwrócone jako lista pytań."
    )


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
        logger.info(f"Warning! Model could not generate reply after {e.n_attempts} retires.")

        return QueryDecision(
            user_route=RouteType.CLARIFY,
            clarify_message="Jestem botem Wikipedii. Twoje pytanie jest dla mnie trochę niejasne. Czy mógłbyś je sformułować inaczej lub podać więcej szczegółów?"
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
        logger.info(f"Warning! Model could not generate reply after {e.n_attempts} retires.")
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
        logger.info(f"Warning! Model could not generate reply after {e.n_attempts} retires.")
        return QueryExpander(
            expanded_queries=[]
        )
    

def rag_query(client: Instructor, user_query: str, model_name: str) -> RAGAnswer:
    try:
        system_prompt = RAG_QUERY_SYSTEM_PROMPT
        
        decision = client.chat.completions.create(
            model=model_name,
            response_model=RAGAnswer,
            max_retries=3,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
        )
        return decision
    except InstructorRetryException as e:
        logger.info(f"Warning! Model could not generate reply after {e.n_attempts} retires.")

        return RAGAnswer(
            is_found=False,
            answer="Nie udało mi się znaleźć odpowiedzi na zadaną kwestię."
        )
    

def further_questions_query(client: Instructor, contenxt: str, model_name: str) -> RAGQuestions:
    try:
        system_prompt = FURTHER_QUESTIONS_SYSTEM_PROMPT
        
        questions = client.chat.completions.create(
            model=model_name,
            response_model=RAGQuestions,
            max_retries=3,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": contenxt},
            ],
        )
        return questions
    except InstructorRetryException as e:
        logger.info(f"Warning! Model could not generate reply after {e.n_attempts} retires.")

        return RAGQuestions(
            questions=[]
        )