#coding=utf8
import os, sys, logging, traceback
from typing import Dict, Any
# Add the parent directory to the path so that we can import the evaluation module
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import evaluation


logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt='[%(asctime)s][%(filename)s - %(lineno)d][%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def evaluate_airqa(pred_answer: str, gold: Dict[str, Any]) -> float:
    """ Evaluate the predicted answer against the gold answer. The predicted answer is a string (from LLM response), and the gold answer is included in the gold data dictionary.
    """
    if str(pred_answer).startswith('[ERROR]:'): return 0.
    function_name = gold['evaluator']['eval_func']
    eval_func = getattr(evaluation, function_name, None)
    assert eval_func is not None, f"Evaluation function `{function_name}` not found in the evaluation module. Remember to import it in the evaluation/__init__.py file."
    eval_kwargs = gold['evaluator']['eval_kwargs']
    try:
        score = eval_func(pred_answer, **eval_kwargs)
    except Exception as e:
        score = 0.0
        logger.error(traceback.format_exc())
        logger.error('Error occurred during evaluation: %s', str(e))
    return score
