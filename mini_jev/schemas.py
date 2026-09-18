from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

class NoulQuestion(BaseModel):
    type: Literal["noul"] = "noul"
    instructions: str

class ChoiceQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: str
    criteria: Union[Dict[str, str], List[str]]

class ScoreQuestion(BaseModel):
    type: Literal["score"] = "score"
    instructions: str
    criteria: List[str]

QuestionType = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]

class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(..., description="Probability of truth/positive, between 0.0 and 1.0")
    confidence: float = Field(..., description="Model confidence score")

class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str = Field(..., description="Selected option key")
    probabilities: Dict[str, float] = Field(..., description="Probability distribution across all options")
    confidence: float = Field(..., description="Probability of the top choice")

class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float = Field(..., description="Continuous expected score based on rubric weights")
    legend: Dict[str, str] = Field(..., description="Index to criteria description mapping")
    probabilities: Dict[str, float] = Field(..., description="Probability of each rubric level")
    confidence: float = Field(..., description="Probability of the highest likelihood level")

AnswerType = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]

class UsageStats(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 1
    latency_ms: float = 0.0

class EvaluateResponse(BaseModel):
    model: str
    answers: Dict[str, AnswerType]
    usage: UsageStats
