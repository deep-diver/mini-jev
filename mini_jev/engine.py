import json
import string
import time
from typing import Any, Dict, List, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_jev.schemas import (
    AnswerType,
    ChoiceAnswer,
    ChoiceQuestion,
    EvaluateResponse,
    NoulAnswer,
    NoulQuestion,
    QuestionType,
    ScoreAnswer,
    ScoreQuestion,
    UsageStats,
)


class Gemma3JevEngine:
    """
    Mini-Jev Engine powered by Google Gemma 3 270M-IT.
    Executes single-forward-pass logit scoring (non-autoregressive)
    to achieve sub-30ms typed decisions on Apple Silicon MPS.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-3-270m-it",
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_id = model_id
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load weights on CPU first, then transfer to target device (avoids MPS device_map malloc bug)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self._cache_token_ids()

    def _cache_token_ids(self):
        """Precompute token IDs for Yes/No, A-Z, and 0-9."""
        # Yes / No
        self.tok_yes_upper = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.tok_yes_lower = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        self.tok_no_upper = self.tokenizer.encode("No", add_special_tokens=False)[0]
        self.tok_no_lower = self.tokenizer.encode("no", add_special_tokens=False)[0]

        # Letters A-Z
        self.letters = list(string.ascii_uppercase)
        self.letter_tokens = {}
        for letter in self.letters:
            t_upper = self.tokenizer.encode(letter, add_special_tokens=False)[0]
            t_lower = self.tokenizer.encode(letter.lower(), add_special_tokens=False)[0]
            self.letter_tokens[letter] = (t_upper, t_lower)

        # Digits 0-9
        self.digit_tokens = {}
        for d in range(10):
            d_str = str(d)
            self.digit_tokens[d_str] = self.tokenizer.encode(d_str, add_special_tokens=False)[0]

    def _format_state(self, state: Any) -> str:
        if isinstance(state, str):
            return state.strip()
        elif isinstance(state, (dict, list)):
            return json.dumps(state, ensure_ascii=False, indent=2)
        return str(state)

    def _build_prompt(self, user_content: str) -> str:
        messages = [{"role": "user", "content": user_content}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _get_next_token_logits(self, prompt: str) -> Tuple[torch.Tensor, int]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Logits of the very next token after generation prompt
            next_token_logits = outputs.logits[0, -1, :].float()
        return next_token_logits, input_len

    def evaluate_noul(self, state_str: str, question: NoulQuestion) -> Tuple[NoulAnswer, int]:
        content = (
            f"State / Context:\n{state_str}\n\n"
            f"Question:\n{question.instructions}\n\n"
            f"Answer with Yes or No only."
        )
        prompt = self._build_prompt(content)
        logits, input_tokens = self._get_next_token_logits(prompt)

        logit_yes = max(logits[self.tok_yes_upper].item(), logits[self.tok_yes_lower].item())
        logit_no = max(logits[self.tok_no_upper].item(), logits[self.tok_no_lower].item())

        tensor_pair = torch.tensor([logit_yes, logit_no], dtype=torch.float32)
        probs = torch.softmax(tensor_pair, dim=-1)
        p_yes = probs[0].item()
        confidence = max(p_yes, 1.0 - p_yes)

        return NoulAnswer(
            type="noul",
            noul=round(p_yes, 4),
            confidence=round(confidence, 4),
        ), input_tokens

    def evaluate_choice(self, state_str: str, question: ChoiceQuestion) -> Tuple[ChoiceAnswer, int]:
        if isinstance(question.criteria, dict):
            keys = list(question.criteria.keys())
            options_text = [
                f"{self.letters[i]}: {key} - {question.criteria[key]}"
                for i, key in enumerate(keys)
            ]
        else:
            keys = question.criteria
            options_text = [
                f"{self.letters[i]}: {opt}"
                for i, opt in enumerate(keys)
            ]

        content = (
            f"State / Context:\n{state_str}\n\n"
            f"Question:\n{question.instructions}\n\n"
            f"Options:\n" + "\n".join(options_text) + "\n\n"
            f"Which option is the best answer? Respond with only the option letter ({', '.join(self.letters[:len(keys)])})."
        )
        prompt = self._build_prompt(content)
        logits, input_tokens = self._get_next_token_logits(prompt)

        cand_logits = []
        for i in range(len(keys)):
            letter = self.letters[i]
            t_upper, t_lower = self.letter_tokens[letter]
            l_val = max(logits[t_upper].item(), logits[t_lower].item())
            cand_logits.append(l_val)

        tensor_logits = torch.tensor(cand_logits, dtype=torch.float32)
        probs = torch.softmax(tensor_logits, dim=-1).tolist()

        best_idx = int(torch.argmax(tensor_logits).item())
        best_choice = keys[best_idx]
        confidence = probs[best_idx]

        prob_map = {keys[i]: round(probs[i], 4) for i in range(len(keys))}

        return ChoiceAnswer(
            type="choice",
            choice=best_choice,
            probabilities=prob_map,
            confidence=round(confidence, 4),
        ), input_tokens

    def evaluate_score(self, state_str: str, question: ScoreQuestion) -> Tuple[ScoreAnswer, int]:
        levels = question.criteria
        levels_text = [f"{i}: {desc}" for i, desc in enumerate(levels)]

        content = (
            f"State / Context:\n{state_str}\n\n"
            f"Question:\n{question.instructions}\n\n"
            f"Rubric / Scale:\n" + "\n".join(levels_text) + "\n\n"
            f"Which level best describes the context? Respond with only the level number (0 to {len(levels) - 1})."
        )
        prompt = self._build_prompt(content)
        logits, input_tokens = self._get_next_token_logits(prompt)

        cand_logits = []
        for i in range(len(levels)):
            d_str = str(i)
            t_id = self.digit_tokens[d_str]
            cand_logits.append(logits[t_id].item())

        tensor_logits = torch.tensor(cand_logits, dtype=torch.float32)
        probs = torch.softmax(tensor_logits, dim=-1).tolist()

        continuous_score = sum(i * probs[i] for i in range(len(levels)))
        confidence = max(probs)
        legend = {str(i): levels[i] for i in range(len(levels))}
        prob_map = {str(i): round(probs[i], 4) for i in range(len(levels))}

        return ScoreAnswer(
            type="score",
            score=round(continuous_score, 4),
            legend=legend,
            probabilities=prob_map,
            confidence=round(confidence, 4),
        ), input_tokens

    def evaluate(
        self,
        state: Any,
        questions: Dict[str, Union[QuestionType, dict]],
    ) -> EvaluateResponse:
        start_time = time.perf_counter()
        state_str = self._format_state(state)

        answers: Dict[str, AnswerType] = {}
        total_input_tokens = 0

        for q_name, q_spec in questions.items():
            if isinstance(q_spec, dict):
                q_type = q_spec.get("type", "choice").lower()
                if q_type == "noul":
                    question_obj = NoulQuestion(**q_spec)
                elif q_type == "score":
                    question_obj = ScoreQuestion(**q_spec)
                else:
                    question_obj = ChoiceQuestion(**q_spec)
            else:
                question_obj = q_spec

            if question_obj.type == "noul":
                ans, in_tok = self.evaluate_noul(state_str, question_obj)
            elif question_obj.type == "choice":
                ans, in_tok = self.evaluate_choice(state_str, question_obj)
            elif question_obj.type == "score":
                ans, in_tok = self.evaluate_score(state_str, question_obj)
            else:
                raise ValueError(f"Unknown question type: {question_obj.type}")

            answers[q_name] = ans
            total_input_tokens += in_tok

        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return EvaluateResponse(
            model=f"mini-jev({self.model_id})",
            answers=answers,
            usage=UsageStats(
                input_tokens=total_input_tokens,
                output_tokens=len(answers),
                latency_ms=round(elapsed_ms, 2),
            ),
        )
