#coding=utf8
import re, json, os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import Any, Dict, List, Tuple, Optional, Union
from utils.functions.common_functions import call_llm, call_llm_with_message
from utils.airqa_utils import get_relevant_papers_by_title

def eval_m3sciqa(pred: Any, question: str, reference_answer: str, model: str = "gpt-4-0125-preview", temperature: float = 0.0, **kwargs) -> float:
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
    
    prompt = f"""You are the sole expert in this field and you can understand scientific papers.
    
    I am testing a model performance on open-ended questions, I want you to help me in checking if the candidate answer has the same meaning with the reference answer given the question. If you think the reference answer and the candidate answer have the same meaning, respond {{"selection": "1"}}; else, respond by {{"selection": "0"}}; if you think the candidate is partially correct, respond by {{"selection": "0.5"}}.

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
    
    response = call_llm(prompt, model=model, temperature=temperature)
    response = extract_response(response)
    response = json.loads(response)['selection']
    
    # return 1.0 if float(response) > 0.9 else 0.0
    return float(response)