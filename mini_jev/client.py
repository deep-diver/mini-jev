from typing import Any, Dict, Optional, Union
import torch

from mini_jev.engine import Gemma3JevEngine
from mini_jev.schemas import EvaluateResponse, QuestionType


class MiniJevClient:
    """
    Client for MiniJev, matching the TypeSafe Jev API interface.
    """

    def __init__(
        self,
        model: str = "google/gemma-3-270m-it",
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.engine = Gemma3JevEngine(model_id=model, device=device, dtype=dtype)

    def evaluate(
        self,
        state: Any,
        questions: Dict[str, Union[QuestionType, dict]],
    ) -> EvaluateResponse:
        """
        Evaluates a set of typed questions against a state context.

        Args:
            state: The input context (string, dict, or list).
            questions: Dictionary mapping question names to question definitions.
                       Question types can be 'noul', 'choice', or 'score'.

        Returns:
            EvaluateResponse containing typed answers, probabilities, and latency stats.
        """
        return self.engine.evaluate(state=state, questions=questions)
