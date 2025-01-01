#coding=utf8
from typing import Any, List, Dict, Union, Optional
import json, math, re
from fuzzywuzzy import fuzz, process


def eval_string_exact_match(
        pred: str,
        gold: str,
        lowercase: bool = False,
        ignore_blank: bool = False,
        **kwargs
    ) -> float:
    """ Evaluate the predicted answer against the gold answer using exact string match.
    @param:
        pred: str, the predicted answer
        gold: str, the gold answer
        ignore_blank: bool, whether to ignore the blank spaces, by default, False
        lowercase: bool, whether to convert the strings to lowercase before comparison, by default, False
    @return
        float, 1.0 or 0.0
    """
    pred, gold = str(pred).strip(), str(gold).strip()
    if ignore_blank:
        pred, gold = re.sub(r'\s+', '', pred), re.sub(r'\s+', '', gold)
    return float(pred.lower() == gold.lower()) if lowercase else float(pred == gold)


def eval_string_fuzzy_match(
        pred: str,
        gold: str,
        fuzz_method: str = 'ratio',
        threshold: int = 95,
        ignore_blank: bool = False,
        lowercase: bool = False,
        **kwargs
    ) -> float:
    """ Evaluate the predicted answer against the gold answer using fuzzy string match.
    @param:
        pred: str, the predicted answer
        gold: str, the gold answer
        fuzz_method: str, the method for fuzzy string matching, by default, 'ratio'
        threshold: int, the threshold for fuzzy string matching, by default, 95
        ignore_blank: bool, whether to ignore the blank spaces, by default, False
        lowercase: bool, whether to convert the strings to lowercase before comparison, by default, False
    @return
        float, 1.0 or 0.0
    """
    pred, gold = str(pred).strip(), str(gold).strip()
    # fuzz_method chosen from ['ratio', 'partial_ratio', 'token_sort_ratio', 'token_set_ratio']
    fuzz_function = getattr(fuzz, fuzz_method)
    if ignore_blank:
        if fuzz_method in ['token_sort_ratio', 'token_set_ratio']: # for tokens, preserve the blank spaces
            pred, gold = re.sub(r'\s+', ' ', pred), re.sub(r'\s+', ' ', gold)
        else:
            pred, gold = re.sub(r'\s+', '', pred), re.sub(r'\s+', '', gold)
    return float(fuzz_function(pred.lower(), gold.lower()) >= threshold) if lowercase else float(fuzz_function(pred, gold) >= threshold)


def eval_bool_exact_match(pred: Any, gold: bool, **kwargs) -> float:
    """ Evaluate the predicted answer against the gold answer using exact boolean match.
    """
    try:
        if isinstance(pred, bool):
            return float(bool(pred) == bool(gold))
        elif isinstance(pred, int):
            if int(pred) not in [0, 1]:
                return 0.0
            return float(bool(pred) == bool(gold))
        elif isinstance(pred, float):
            if float(pred) not in [0.0, 1.0]:
                return 0.0
            return float(bool(pred) == bool(gold))
        else:
            pred = str(pred)
            if pred.lower() in ['true', '1', 'yes', 'y', 't']:
                pred = True
            elif pred.lower() in ['false', '0', 'no', 'n', 'f']:
                pred = False
            else:
                return 0.0
            return float(bool(pred) == bool(gold))
    except Exception:
        return 0.0


def eval_int_exact_match(pred: Any, gold: int, **kwargs) -> float:
    """ Evaluate the predicted answer against the gold answer using exact integer match.
    """
    try:
        return float(int(pred) == int(gold))
    except Exception:
        return 0.0


def eval_float_exact_match(pred: Any, gold: float, ndigits: Optional[int] = None, tolerance: float = 1e-6, **kwargs) -> float:
    """ Evaluate the predicted answer against the gold answer using exact float match.
    """
    try:
        pred_float = float(pred)
        if ndigits is not None:
            pred_float = round(pred_float, ndigits)
            gold = round(gold, ndigits)
        return float(math.isclose(pred_float, gold, rel_tol=tolerance))
    except Exception:
        return 0.0


def try_parse_list(s: Any) -> List[Any]:
    """ Try to parse an object s into a list.
    """
    if isinstance(s, list):
        return s
    elif isinstance(s, tuple):
        return list(s)
    elif isinstance(s, set):
        return list(s)
    try:
        s_list = eval(s)
        if isinstance(s_list, list):
            return s_list
        elif isinstance(s_list, tuple):
            return list(s_list)
        elif isinstance(s_list, set):
            return list(s_list)
        else:
            return [s]
    except Exception:
        return [s]


def try_parse_dict(s: Any) -> Dict[str, Any]:
    """ Try to parse an object s into a dictionary.
    """
    if isinstance(s, dict):
        return s
    try:
        s_dict = json.loads(s)
        return s_dict
    except Exception:
        pass
    try:
        s_dict = eval(s)
        if isinstance(s_dict, dict):
            return s_dict
        else:
            return {}
    except Exception:
        return {}


def eval_structured_object_exact_match(pred: Union[str, Any], gold: Any, **kwargs) -> float:
    """ Evaluate the predicted answer against the gold answer using exact object match with optional arguments.
    Note that, structured objects here refer to list or dictionary, since json format `gold` does not support tuple and set.
        tuple -> list , set -> unordered list (`ignore_order=True` in kwargs)
    @param
        pred: str, the predicted answer
        gold: Any, the gold answer
        kwargs: dict, additional keyword arguments for comparing different types of objects
            - ignore_order: bool, whether to ignore the order of the elements in the list, by default, False
            - tolerance: float, the tolerance for comparing the float numbers, by default, 1e-6
            - ndigits: int, the number of digits to round the float numbers, by default, None
            - lowercase: bool, whether to convert the strings to lowercase before comparison, by default, False
            - ignore_blank: bool, whether to ignore the blank spaces, by default, False
            - threshold: int, the threshold for fuzzy string matching, by default, -1, means using exact match
            - fuzz_method: str, the method for fuzzy string matching, by default, 'ratio'
    @return
        float, 1.0 or 0.0
    """
    if isinstance(gold, list):
        gold_len = len(gold)
        pred = try_parse_list(pred)
        if not isinstance(pred, list):
            return 0.0
        pred_len = len(pred)
        if pred_len != gold_len:
            return 0.0
        ignore_order = kwargs.get('ignore_order', False)
        if ignore_order:
            pred = sorted(pred, key=lambda x: str(x))
            gold = sorted(gold, key=lambda x: str(x))
        for i, (p, g) in enumerate(zip(pred, gold)):
            if eval_structured_object_exact_match(p, g, **kwargs) < 0.5:
                return 0.0
        return 1.0

    elif isinstance(gold, dict):
        gold_len = len(gold)
        pred = try_parse_dict(pred)
        if not isinstance(pred, dict):
            return 0.0
        pred_len = len(pred)
        if pred_len != gold_len:
            return 0.0
        lowercase = kwargs.get('lowercase', False)
        if lowercase:
            gold = {k.lower(): v for k, v in gold.items()}
            pred = {k.lower(): v for k, v in pred.items()}
        for k, v in gold.items():
            if not isinstance(k, str) and k in pred:
                pred_v = pred[k]
                if eval_structured_object_exact_match(pred_v, v, **kwargs) < 0.5:
                    return 0.0
            elif str(k) in pred:
                pred_v = pred[str(k)]
                if eval_structured_object_exact_match(pred_v, v, **kwargs) < 0.5:
                    return 0.0
            else: # key not exists in pred
                return 0.0
        return 1.0

    elif isinstance(gold, int):
        return eval_int_exact_match(pred, gold, **kwargs)
    elif isinstance(gold, float):
        return eval_float_exact_match(pred, gold, **kwargs)
    else:
        threshold = kwargs.get('threshold', -1)
        if threshold > 0:
            return eval_string_fuzzy_match(pred, gold, **kwargs)
        else:
            return eval_string_exact_match(pred, gold, **kwargs)