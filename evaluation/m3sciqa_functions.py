#coding=utf8
from typing import Any
import re, json, os, sys
from .llm_functions import call_llm_with_message


DEFAULT_M3SCIQA_SYSTEM_PROMPT = "You are the sole expert in this field and you can understand scientific papers."
DEFAULT_M3SCIQA_LLM_MODEL = 'gpt-4-0125-preview'
DEFAULT_M3SCIQA_TEMPERATURE = 0.1


def eval_m3sciqa(pred: Any, question: str, reference_answer: str, model: str = DEFAULT_M3SCIQA_LLM_MODEL, temperature: float = DEFAULT_M3SCIQA_TEMPERATURE, **kwargs) -> float:
    """ Evaluate the predicted answer for the M3SciQA dataset.
    @args:
        pred_ans: str, predicted answer
        question: str, question
        reference_answer: str, gold answer
    @return:
        score: float, evaluation score
    """
    
    def extract_response(text):
        import re
        pattern = r'\{.*?\}'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(0) if match else None
    
    prompt = f"""I am testing a model performance on open-ended questions, I want you to help me in checking if the candidate answer has the same meaning with the reference answer given the question. If you think the reference answer and the candidate answer have the same meaning, respond {{"selection": "1"}}; else, respond by {{"selection": "0"}}; if you think the candidate is partially correct, respond by {{"selection": "0.5"}}.

    <QUESTION>
    {question}
    </QUESTION>

    <REFERENCE>
    {reference_answer}
    </REFERENCE>

    <CANDIDATE>
    {str(pred)}
    </CANDIDATE>

    Again, if you think they have the same meaning, respond {{"selection": "1"}}; if you think they are totally irrelevant, respond by {{"selection": "0"}} only; if you think the candidate is partially correct, respond by {{"selection": "0.5"}}.
    
    Do not use other format.
"""
    
    response = call_llm_with_message(
        messages=[
            {"role": "system", "content": DEFAULT_M3SCIQA_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        model=model,
        temperature=temperature
    )
    response = extract_response(response)
    response = json.loads(response)['selection']
    
    # return 1.0 if float(response) > 0.9 else 0.0
    return float(response)
