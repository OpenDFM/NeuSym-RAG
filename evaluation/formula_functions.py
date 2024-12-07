#coding=utf8
import re
from typing import Any, Dict, List, Union, Tuple, Optional
from .llm_functions import DEFAULT_LLM_MODEL, DEFAULT_TEMPERATURE, _eval_with_llm


def eval_complex_math_formula_with_llm(pred: Any, formulas: Union[str, List[str]], question: str, llm_model: str = DEFAULT_LLM_MODEL, temperature: float = DEFAULT_TEMPERATURE, ignore_order: bool = True, **kwargs) -> float:
    """ Evaluate the complex math formula with LLM.
    @param:
        pred: The predicted answer.
        formulas: The list of complex math formulas.
        question: The input question.
        llm_model: The LLM model name.
        temperature: The temperature parameter for LLM.
        ignore_order: Whether to ignore the order of formulas.
    @return:
        The evaluation score, 0.0 or 1.0.
    """
    # Prepare the input
    formulas_str = '\n'.join(['- ' + formula.strip() for formula in formulas]) if isinstance(formulas, list) else formulas
    template = f"""You are an intelligent symbolic math solver who is expert in in determining whether the predicted LaTeX code is mathematically equivalent to the golden math formula provided, based on the given input question. You will be provided with the following information:
- [Question]: A raw question describing the problem context.
- [Reference Math Formula]: The correct math formula(s), expressed in LaTeX.
- [Predicted Latex Code]: The answer generated in LaTeX format.
**Your task is to:**
1. Determine if the predicted LaTeX code is mathematically equivalent to the reference math formula(s).
2. If the reference math formula is a list of formulas, ensure the predicted answer contains all of these formulas in {'any' if ignore_order else 'sequential'} order.
3. Your judgment must be based on mathematical equivalence, not string-level similarity.
**Final Output Format:**
- You must provide the final decision in the following format:
```txt
True/False
```
- Wrap the output decision (only "True" or "False") with triple backticks. Any additional text or characters outside this format will be considered invalid.
**Suggested Evaluation Process:**
- Analyze the input question and reference math formula step-by-step.
- Check each component of the predicted LaTeX code against the reference formula(s).
- Confirm mathematical equivalence or identify discrepancies.
Now, let's start!

[Question]: {question}
[Reference Math Formula]: Here is the complex math formula:
{formulas_str}
[Predicted LaTeX Code]: {str(pred)}

Let's reason step-by-step, then provide the final judgment.
Now, let's start!
"""
    return _eval_with_llm(template, llm_model, temperature)