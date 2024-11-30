#coding=utf8
import re
from typing import Any, Dict, List, Tuple, Optional
from utils.functions.common_functions import call_llm, call_llm_with_message


def eval_answer_with_llm_scoring_points(pred: Any, scoring_points: List[str], question: str, llm_model: str = 'gpt-4o', temperature: float = 0.0, ignore_order: bool = True, **kwargs) -> float:
    """ Evaluate the answer with LLM scoring points.
    By default, the LLM model is GPT-4o with temperature 0.0.
    """
    # Prepare the input
    num_scoring_points = len(scoring_points)
    scoring_points_str = '\n'.join(['- ' + sp.strip() for sp in scoring_points])
    template = f"""You are an intelligent judgement system who is expert in determining whether a predicted answer exactly mentions all required scoring points for the input question. You will be given the raw question, the predicted answer, and all required scoring points. And you need to provide the final decision with the following format:
```txt
True/False
```
Notice that:
1. Remember to wrap the final judgement with triple backticks.
2. The predicted answer string must exactly be "True" or "False" without any extra character or punctuation. Any other text will be considered as incorrect.
3. The given answer is only considered as correct IF AND ONLY IF the required scoring points are ALL mentioned in the predicted answer.
4. The {'structure, format and order' if ignore_order else 'structure and format'} of the scoring points does not matter. We only care about the semantics and content.
Now, let's start!

[Question]: {question}
[Predicted Answer]: {str(pred)}
[Scoring points]: In total, there are {num_scoring_points} scoring points:
{scoring_points_str}

Let's think step-by-step, and then provide the final judgement.
"""
    # Call the LLM model
    llm_output = call_llm(template, llm_model, temperature=temperature)
    # Extract the final judgement
    matched = re.findall(r'```(txt)?\s(.*?)\s```', llm_output, re.DOTALL)
    if len(matched) == 0:
        return 0.0
    final_judgement = matched[-1][1].strip().lower()
    # Return the final judgement
    return 1.0 if final_judgement == 'true' else 0.0