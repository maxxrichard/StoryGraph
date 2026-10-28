"""Agreement between model tags and editor gold labels (multi-label per field)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from storygraph.schemas import ContentSignals


@dataclass
class FieldAgreement:
    field: str
    exact_match: float  # share of titles where the label sets are identical
    macro_f1: float
    micro_f1: float
    kappa: float  # Cohen's kappa on per-(title,label) binary decisions
    n: int


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 1.0


def field_agreement(
    pred: Sequence[ContentSignals], gold: Sequence[ContentSignals], field: str, vocab: Sequence[str]
) -> FieldAgreement:
    if len(pred) != len(gold):
        raise ValueError("pred and gold must align")
    per_label = {lab: [0, 0, 0] for lab in vocab}  # tp, fp, fn
    exact = 0
    tp = fp = fn = tn = 0
    for p, g in zip(pred, gold):
        ps, gs = set(p.label_names(field)), set(g.label_names(field))
        exact += int(ps == gs)
        for lab in vocab:
            a, b = lab in ps, lab in gs
            if a and b:
                per_label[lab][0] += 1
                tp += 1
            elif a:
                per_label[lab][1] += 1
                fp += 1
            elif b:
                per_label[lab][2] += 1
                fn += 1
            else:
                tn += 1
    present = [v for v in per_label.values() if sum(v) > 0]
    macro = sum(_f1(*v) for v in present) / len(present) if present else 1.0
    micro = _f1(tp, fp, fn)
    n = tp + fp + fn + tn
    po = (tp + tn) / n if n else 1.0
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n) if n else 1.0
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return FieldAgreement(
        field=field,
        exact_match=exact / len(pred) if pred else 1.0,
        macro_f1=macro,
        micro_f1=micro,
        kappa=kappa,
        n=len(pred),
    )


def overall_agreement(
    pred: Sequence[ContentSignals], gold: Sequence[ContentSignals]
) -> dict[str, FieldAgreement]:
    from storygraph.understanding.taxonomy import GENRES, MOODS, THEMES

    return {
        "genres": field_agreement(pred, gold, "genres", GENRES),
        "themes": field_agreement(pred, gold, "themes", THEMES),
        "moods": field_agreement(pred, gold, "moods", MOODS),
    }
