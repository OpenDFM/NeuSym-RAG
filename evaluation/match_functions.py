#coding=utf8
from typing import Any


def eval_exact_string_match(pred: str, gold: str = '', lowercase: bool = True) -> float:
    """ Evaluate the predicted answer against the gold answer using exact string match.
    """
    pred, gold = str(pred).strip(), str(gold).strip()
    return pred.lower() == gold.lower() if lowercase else pred == gold