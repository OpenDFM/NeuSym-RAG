#coding=utf8
from typing import Any, List, Dict, Union, Optional
import json, math, re
from fuzzywuzzy import fuzz


def eval_paper_relevance_with_reference_answer(pred: Any, question: str, reference_answer: Union[str, List[str]], threshold: int = 95) -> float:
    """Evaluate the relevance of the predicted paper with the question and reference answer.
    @param:
        pred: The predicted paper title.
        question: The input question.
        reference_answer: The reference answer.
        **kwargs: The kwargs for eval_paper_relevance_with_llm.
    @return:
        The evaluation score, 0.0 or 1.0.
    """
    if (isinstance(reference_answer, str) and fuzz.ratio(pred.lower(), reference_answer.lower()) >= threshold) or \
        (isinstance(reference_answer, list) and any(fuzz.ratio(pred.lower(), ra.lower()) >= threshold for ra in reference_answer)):
        return 1.0
    return 0.0