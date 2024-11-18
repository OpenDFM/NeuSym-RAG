#coding=utf8
import os, sys
from typing import Dict, Any
# Add the parent directory to the path so that we can import the evaluation module
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import evaluation


def evaluate_airqa(pred_answer: str, gold: Dict[str, Any]) -> float:
    """ Evaluate the predicted answer against the gold answer. The predicted answer is a string (from LLM response), and the gold answer is included in the gold data dictionary.
    """
    function_name = gold['evaluator']['eval_func']
    eval_func = getattr(evaluation, function_name, None)
    assert eval_func is not None, f"Evaluation function `{function_name}` not found in the evaluation module. Remember to import it in the evaluation/__init__.py file."
    eval_kwargs = gold['evaluator']['eval_kwargs']
    return eval_func(pred_answer, **eval_kwargs)
