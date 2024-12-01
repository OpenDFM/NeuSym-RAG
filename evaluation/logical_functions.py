#coding=utf8
from typing import Any, List, Dict, Union, Optional
from .match_functions import try_parse_list


def eval_conjunction(pred: List[Any], eval_func_list: List[str], eval_kwargs_list: List[Dict[str, Any]]) -> float:
    """ Evaluate the conjunction of multiple evaluation functions.
    @param:
        pred: The predicted answer list, passed sequentially to the eval_func_list.
        eval_func_list: The list of evaluation function names.
        eval_kwargs_list: The list of evaluation function kwargs.
    @return:
        The evaluation score, 0.0 or 1.0.
    """
    pred = try_parse_list(pred)
    if len(pred) != len(eval_func_list):
        return 0.0

    import evaluation
    for pred_i, eval_func, eval_kwargs in zip(pred, eval_func_list, eval_kwargs_list):
        function_call = getattr(evaluation, eval_func, None)
        assert function_call is not None, f"Evaluation function `{eval_func}` not found in the evaluation module. Remember to import it in the evaluation/__init__.py file."
        if float(function_call(pred_i, **eval_kwargs)) < 0.5:
            return 0.0
    return 1.0


def eval_disjunction(pred: List[Any], eval_func_list: List[str], eval_kwargs_list: List[Dict[str, Any]]) -> float:
    """ Evaluate the disjunction of multiple evaluation functions.
    @param:
        pred: The predicted answer list, passed sequentially to the eval_func_list.
        eval_func_list: The list of evaluation function names.
        eval_kwargs_list: The list of evaluation function kwargs.
    @return:
        The evaluation score, 0.0 or 1.0.
    """
    pred = try_parse_list(pred)
    if len(pred) != len(eval_func_list):
        return 0.0

    import evaluation
    for pred_i, eval_func, eval_kwargs in zip(pred, eval_func_list, eval_kwargs_list):
        function_call = getattr(evaluation, eval_func, None)
        assert function_call is not None, f"Evaluation function `{eval_func}` not found in the evaluation module. Remember to import it in the evaluation/__init__.py file."
        if float(function_call(pred_i, **eval_kwargs)) > 0.5:
            return 1.0
    return 0.0


def eval_negation(pred: Any, eval_func: str, eval_kwargs: Dict[str, Any]) -> float:
    """ Evaluate the negation of an evaluation function.
    @param:
        pred: The predicted answer
        eval_func: The evaluation function name
        eval_kwargs: The evaluation function kwargs
    @return:
        The evaluation score, 0.0 or 1.0.
    """
    import evaluation
    function_call = getattr(evaluation, eval_func, None)
    assert function_call is not None, f"Evaluation function `{eval_func}` not found in the evaluation module. Remember to import it in the evaluation/__init__.py file."
    result = function_call(pred, **eval_kwargs)
    return 1.0 - result