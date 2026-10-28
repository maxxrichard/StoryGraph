"""LLM-as-judge scoring for generated text, plus agreement with human ratings.

We score each summary on four criteria with a strong judge model, average to a 1-5 score, and
calibrate the judge against a human-rated subset (Spearman + mean absolute gap) before trusting it
on the full set.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean

import numpy as np

from storygraph.genai.prompts import JUDGE_SYSTEM, JUDGE_USER
from storygraph.genai.summaries import TextLLM

CRITERIA = ("faithfulness", "spoiler_safety", "coverage", "fluency")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeScore:
    faithfulness: float
    spoiler_safety: float
    coverage: float
    fluency: float
    rationale: str = ""

    @property
    def overall(self) -> float:
        return mean([self.faithfulness, self.spoiler_safety, self.coverage, self.fluency])


def parse_judge(raw: str) -> JudgeScore:
    m = _JSON_RE.search(raw)
    if not m:
        raise ValueError("judge returned no JSON")
    d = json.loads(m.group(0))
    vals = {c: float(np.clip(float(d.get(c, 1)), 1, 5)) for c in CRITERIA}
    return JudgeScore(**vals, rationale=str(d.get("rationale", "")))


class SummaryJudge:
    def __init__(self, llm: TextLLM, max_chars: int = 12_000):
        self.llm = llm
        self.max_chars = max_chars

    def score(self, transcript_text: str, summary: str) -> JudgeScore:
        text = transcript_text[: self.max_chars]
        raw = self.llm.complete(
            JUDGE_SYSTEM, JUDGE_USER.format(text=text, summary=summary), max_tokens=200
        )
        return parse_judge(raw)


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    # average ties
    out = ranks.astype(float)
    for v in np.unique(x):
        idx = np.where(x == v)[0]
        if len(idx) > 1:
            out[idx] = ranks[idx].mean()
    return out


def spearman(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    if len(a) != len(b) or len(a) < 3:
        raise ValueError("need >= 3 paired observations")
    ra, rb = _rank(np.asarray(a, float)), _rank(np.asarray(b, float))
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


@dataclass
class Calibration:
    n: int
    judge_mean: float
    human_mean: float
    spearman: float
    mean_abs_gap: float

    @property
    def trustworthy(self) -> bool:
        return self.n >= 50 and self.spearman >= 0.6 and self.mean_abs_gap <= 0.5


def calibrate(judge_scores: Sequence[float], human_scores: Sequence[float]) -> Calibration:
    j, h = np.asarray(judge_scores, float), np.asarray(human_scores, float)
    return Calibration(
        n=len(j),
        judge_mean=float(j.mean()),
        human_mean=float(h.mean()),
        spearman=spearman(j, h),
        mean_abs_gap=float(np.abs(j - h).mean()),
    )
