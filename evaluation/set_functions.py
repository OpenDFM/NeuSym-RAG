#coding=utf8

from typing import Any, List
from .match_functions import eval_string_exact_match, eval_string_fuzzy_match, eval_int_exact_match, eval_float_exact_match, eval_structured_object_exact_match, try_parse_list


def eval_element_included(
        pred: Any,
        gold: List[Any],
        element_type: str = 'str',
        ndigits: int = 2,
        tolerance: float = 1e-6,
        fuzz_method: str = 'ratio',
        threshold: int = -1,
        lowercase: bool = False,
        ignore_blank: bool = False,
        **kwargs
    ) -> float:
    """ Evaluate whether the predicted answer is included in the gold answer list.
    @param:
        pred: Any, the predicted answer
        gold: List[Any], the gold answer
        element_type: str, the type of the element, by default, 'str'
        other parameters: see `match_functions.py`
    @return
        float, 1.0 or 0.0
    """
    if element_type == 'str':
        if threshold > 0: # fuzzy matching
            return float(any(eval_string_fuzzy_match(pred, g, fuzz_method=fuzz_method, threshold=threshold, lowercase=lowercase, ignore_blank=ignore_blank, **kwargs) for g in gold))
        else: # exact matching
            return float(any(eval_string_exact_match(pred, g, lowercase=lowercase, ignore_blank=ignore_blank, **kwargs) for g in gold))
    elif element_type == 'int':
        return float(any(eval_int_exact_match(pred, g, **kwargs) for g in gold))
    elif element_type == 'float':
        return float(any(eval_float_exact_match(pred, g, ndigits=ndigits, tolerance=tolerance, **kwargs) for g in gold))
    else:
        return float(any(eval_structured_object_exact_match(pred, g, ndigits=ndigits, tolerance=tolerance, fuzz_method=fuzz_method, threshold=threshold, lowercase=lowercase, ignore_blank=ignore_blank, **kwargs) for g in gold))


def eval_element_list_included(
        pred: List[Any],
        gold: List[Any],
        **kwargs
    ) -> float:
    """ Evaluate whether each element in the predicted answer list is included in the gold answer list.
    @param:
        pred: List[Any], the predicted answer list
        gold: List[List[Any]], the gold answer
    @return:
        float, 1.0 or 0.0
    """
    pred = try_parse_list(pred)
    return float(all(eval_element_included(p, gold, **kwargs) for p in pred))


def eval_element_list_overlap(
        pred: List[Any],
        gold: List[Any],
        count: int = 1,
        **kwargs
    ) -> float:
    """ Evaluate whether the predicted answer list overlaps with the gold answer list (at least `count` distinct elements).
    @param:
        pred: List[Any], the predicted answer list
        gold: List[List[Any]], the gold answer
        overlap: int, the minimum number of distinct elements that should overlap with the golden answer list
    @return:
        float, 1.0 or 0.0
    Examples:
        pred = ['a', 'b', 'c']
        gold = ['d', 'a', 'b']
        eval_element_list_overlap(pred, gold, count=2) -> 1.0
        eval_element_list_overlap(pred, gold, count=3) -> 0.0
    Note that, if we want the `pred` to fully contain `gold`, we can also use this overlap function with `count=len(gold)`.
    """
    pred = try_parse_list(pred)
    distinct = set()
    for p in pred:
        if eval_element_included(p, gold, **kwargs) > 0.5:
            distinct.add(str(p))
            if len(distinct) >= count:
                return 1.0
    return 0.0