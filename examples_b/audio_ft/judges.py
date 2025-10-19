"""Pairwise judge for online DPO."""

import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel

from trl import BasePairwiseJudge


# _SYSTEM_PROMPT = """You are a careful, impartial evaluator for pairwise comparison.
# You will be given a prompt and two candidate responses.
# Your job is to choose which response better answers the prompt according to:
# - Helpfulness, correctness, clarity, safety, and adherence to the prompt.
# - Prefer concise, truthful, and non-speculative answers.
# - Penalize hallucinations, policy violations, and evasiveness.

_SYSTEM_PROMPT = """You are a careful, impartial evaluator for pairwise comparison.
You will be given a prompt of summarization task to summarize a dialogue between an agent and a client in call center, and two candidate summaries.

Evaluate the candidate summaries according to the following criteria:
- Fidelity: Does the summary accurately reflect the spirit and content of the original transcription? Does it respect the tone and nuances of the exchange?
- Completeness: Are all essential points and critical information present? Are any important elements missing?
- Accuracy: Are the facts accurate and consistent with the source? The summary must not contain any erroneous data or introduce elements absent from the original transcription.
- Structural coherence: Does the summary form a logical and well-organized whole? Do the ideas flow naturally, creating a fluid narrative rather than a simple accumulation of information?

You MUST return a single JSON object ONLY (no commentary, no markdown), with this exact schema:
{
  "reasoning": "A few sentences explaining your decision.",
  "choice": "0" | "1"
}
Where "0" means Response 0 is better, and "1" means Response 1 is better.
"""

# todo: better way to handle the template
_USER_TEMPLATE = """[PROMPT]
Tu es un expert en résumé de conversations. Ta mission est de générer un résumé abstractif en français d'une conversation.

- Générez uniquement le résumé, sans ajouter de phrases comme "Voici le résumé de la conversation".
- Ne mentionnez pas le nom de la société représentée par l'agent.
- Ne mentionnez pas le nom du client ni celui de l'agent.
- Le résumé doit rester neutre : utilisez toujours les termes "l'agent" et "le client".
- Le résumé doit avoir une longueur d'environ 100 à 200 mots.

Voici la transcription de la conversation :
{prompt}

[CANDIDATE SUMMARY 0]
{resp0}

[CANDIDATE SUMMARY 1]
{resp1}

Return ONLY the required JSON object. Do not include additional text.
"""


class PairwiseJudgeReasoningSchema(BaseModel):
    reasoning: str
    choice: Literal[0, 1]


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        parts = s.split("```")
        # choose the largest JSON-looking chunk
        for chunk in parts:
            chunk = chunk.strip()
            if chunk.startswith("{") and chunk.endswith("}"):
                return chunk
        return parts[-1]
    return s


def _parse_json_choice(content: str) -> dict[str, Any]:
    raw = _strip_code_fences(content.strip())
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise

# todo: also check OpenAIPairwiseJudge implementation
class VLLMPairwiseJudge(BasePairwiseJudge):
    """
    A BasePairwiseJudge that queries an external LLM served by vLLM (OpenAI-compatible API).
    Parallelized with ThreadPoolExecutor for throughput.
    The LLM must return JSON: { "reasoning": "...", "choice": "0" | "1" }.
    """

    def __init__(
        self,
        # model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: int = 300,
        max_retries: int = 1,
        # request_kwargs: dict[str, Any] | None = None,
        system_prompt: str = _SYSTEM_PROMPT,
        user_template: str = _USER_TEMPLATE,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        max_workers: int | None = None,  # threads for parallel calls
        retry_backoff_s: float = 0.5,  # base backoff per attempt
    ):
        # self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:8022/v1")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        # self.request_kwargs = request_kwargs or {}
        self.request_kwargs = defaultdict(dict)
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        # good default: a modest pool (IO-bound), not CPU-bound
        self.max_workers = max_workers or min(16, (os.cpu_count() or 4) * 2)
        self.retry_backoff_s = retry_backoff_s

        if self.max_tokens is not None:
            self.request_kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            self.request_kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            self.request_kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            self.request_kwargs["extra_body"]["top_k"] = self.top_k
        if self.min_p is not None:
            self.request_kwargs["extra_body"]["min_p"] = self.min_p
        self.request_kwargs["extra_body"]["guided_json"] = PairwiseJudgeReasoningSchema.model_json_schema()
        self.request_kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}

        self.init_openai_client()

    def get_default_model_name(self):
        """Get the first model name returned by the OpenAI server."""
        model_list = self.client.models.list()

        if not getattr(model_list, "data", None):
            raise RuntimeError("No models available from the server - cannot pick a default model.")

        self.model_name = model_list.data[0].id

    def init_openai_client(self):
        """Init the OpenAI client."""
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_s,
        )

        self.get_default_model_name()

    def _build_messages(self, prompt: str, resp0: str, resp1: str):
        """Build the messages for the OpenAI client."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(prompt=prompt, resp0=resp0, resp1=resp1)},
        ]

    def _judge_once(self, prompt: str, resp0: str, resp1: str) -> int:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                messages = self._build_messages(prompt, resp0, resp1)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **self.request_kwargs,
                )
                content = response.choices[0].message.content
                # print(f"Judge response: {response}")
                obj = _parse_json_choice(content)
                choice = str(obj.get("choice", "")).strip()
                if choice not in ("0", "1"):
                    raise ValueError(f"Invalid 'choice' value: {choice!r}")
                return int(choice)
            except Exception as e:
                last_error = e
                # exponential backoff with small cap
                time.sleep(min(self.retry_backoff_s * (2**attempt), 2.0))
        raise RuntimeError(f"Judge failed after {self.max_retries} attempts: {last_error}")

    def judge(self, prompts: list[str], completions: list[list[str]], shuffle_order: bool = True) -> list[int]:
        """
        Returns a list of indices (0 or 1) for each prompt; -1 on failure.
        """
        assert len(prompts) == len(completions), "prompts and completions must have the same length"
        n = len(prompts)

        # Prepare tasks with optional shuffling to mitigate positional bias.
        tasks = []
        for i, (p, pair) in enumerate(zip(prompts, completions)):
            assert len(pair) == 2, "each completion pair must have exactly two strings"
            a, b = pair[0], pair[1]
            if shuffle_order and random.random() < 0.5:
                r0, r1 = b, a
                unshuffle = (1, 0)  # model's 0->orig1, 1->orig0
            else:
                r0, r1 = a, b
                unshuffle = (0, 1)
            tasks.append((i, p, r0, r1, unshuffle))

        results = [-1] * n  # default to -1 on failure

        # Parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_map = {}
            for i, p, r0, r1, unshuffle in tasks:
                fut = pool.submit(
                    self._judge_once,
                    p,
                    r0,
                    r1,
                )
                future_map[fut] = (i, unshuffle)

            for fut in as_completed(future_map.keys()):
                i, unshuffle = future_map[fut]
                try:
                    choice_shuffled = fut.result()
                    if choice_shuffled in (0, 1):
                        results[i] = unshuffle[choice_shuffled]
                except Exception as e:
                    print(f"Judge failed for prompt {p}: {e}")
                    results[i] = -1

        print(f"Judge results: {results}")
        return results
