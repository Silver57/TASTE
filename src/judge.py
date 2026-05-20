"""Claude Sonnet 4.6 pairwise judge for the LLM-as-judge sub-study.

Reads ANTHROPIC_API_KEY from the environment. Each call is temperature=0 and
asks for a JSON object with exactly two keys: choice ("A" or "B") and reason
(a short string). Retries on transient errors with exponential backoff.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

import anthropic


SYSTEM_PROMPT = (
    "You are an expert book recommendation evaluator. "
    "You will be given a reader's preference profile, the ground-truth outcome "
    "for one held-out pair of books (which the reader actually liked vs. "
    "disliked), and two candidate one-sentence recommendations (A and B) for "
    "that pair. Choose the recommendation that better matches the ground truth "
    "and is more useful to the reader.\n\n"
    "Respond with a single JSON object and nothing else. The object must have "
    "exactly two keys:\n"
    '  "choice": either "A" or "B"\n'
    '  "reason": a short string (one sentence) explaining the choice.\n'
    "Do not wrap the JSON in code fences."
)


USER_TEMPLATE = """Reader profile (a sample of titles they have rated):
  Liked: {liked}
  Disliked: {disliked}

Held-out pair (ground truth):
  Reader LIKED:    {gt_liked}
  Reader DISLIKED: {gt_disliked}

Candidate recommendations for this pair:
  Recommendation A: {rec_a}
  Recommendation B: {rec_b}

Which recommendation better matches the reader's actual preference for this pair? Reply with JSON only."""


@dataclass
class JudgeResponse:
    choice: str           # "A" or "B"
    reason: str
    raw: str              # full text returned by the model


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(raw: str) -> JudgeResponse:
    """Parse the model's text into a JudgeResponse. Tolerates stray prose."""
    m = _JSON_RE.search(raw)
    if not m:
        raise ValueError(f"No JSON object in judge response: {raw!r}")
    data = json.loads(m.group(0))
    choice = str(data.get("choice", "")).strip().upper()
    if choice not in {"A", "B"}:
        raise ValueError(f"choice must be 'A' or 'B', got {choice!r} in {raw!r}")
    reason = str(data.get("reason", "")).strip()
    return JudgeResponse(choice=choice, reason=reason, raw=raw)


class Judge:
    """Thin Anthropic wrapper. Construct once, reuse across all matchups."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 400,
        max_retries: int = 5,
    ):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before running the judge."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def judge(
        self,
        liked: list[str],
        disliked: list[str],
        gt_liked: str,
        gt_disliked: str,
        rec_a: str,
        rec_b: str,
    ) -> JudgeResponse:
        user_msg = USER_TEMPLATE.format(
            liked="; ".join(liked) or "(none provided)",
            disliked="; ".join(disliked) or "(none provided)",
            gt_liked=gt_liked,
            gt_disliked=gt_disliked,
            rec_a=rec_a,
            rec_b=rec_b,
        )

        delay = 1.0
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                msg = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=0,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw = "".join(
                    block.text for block in msg.content if getattr(block, "type", None) == "text"
                )
                return _parse(raw)
            except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                last_err = e
                # 4xx other than 429 should not be retried.
                status = getattr(e, "status_code", None)
                if isinstance(e, anthropic.APIStatusError) and status is not None and 400 <= status < 500 and status != 429:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
            except ValueError as e:
                # Malformed JSON — retry once with the same temperature=0; if it
                # repeats, surface it.
                last_err = e
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

        assert last_err is not None
        raise last_err
