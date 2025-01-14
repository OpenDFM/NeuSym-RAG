#coding=utf8
import os, sys, json
from abc import ABC, abstractmethod
from typing import List, Dict, Any
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from annotation.moderator_prompt import MODERATE_PROMPT, EVALUATOR_PROMPT, USECASE_PROMPT
from utils.functions.common_functions import call_llm_with_pattern, call_llm

EVALUATIONS_FILE = os.path.join('evaluation', 'evaluations.json')
EVALUATIONS = json.load(open(EVALUATIONS_FILE, 'r'))
EVALUATORS_PROMPT = ""
for eval_func in EVALUATIONS:
    evaluation = EVALUATIONS[eval_func]
    EVALUATORS_PROMPT += EVALUATOR_PROMPT.format(
        function = eval_func,
        description = evaluation['description'],
        parameters = evaluation['parameters'],
        use_cases = "\n".join([
            USECASE_PROMPT.format(
                index = idx,
                example = usecase['example'],
                explanation = usecase['explanation']
            ) for idx, usecase in enumerate(evaluation['use_cases'], start=1)
        ])
    )

class BaseModerator(ABC):
    """ Moderate the question and fill the other parameters.
    1. Reform the question and the answer.
    2. Consider `evaluator`, `answer_format` and `tags`.
    TODO: Moderate human examples.
    """
    model: str = None
    temperature: float = None
    
    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature
    
    def _moderate_with_llm(
            self,
            template: str
        ) -> List[Any]:
        pattern = r"```(txt)?\s*\[question\]:\s*(.*?)\s*\[evaluator\]:\s*(.*?)\s*\[answer_format\]:\s*(.*?)\s*\[answer\]:\s*(.*?)\s*\[tag\]:\s*(.*?)```"
        response = call_llm_with_pattern(template, pattern, self.model, self.temperature)
        if not response: raise ValueError(f"Failed to Parse the Response. {response}")
        return response[1:]
    
    def moderate(self, question: str, answer: str) -> List[Any]:
        template = MODERATE_PROMPT.format(
            evaluator = EVALUATORS_PROMPT,
            question = question,
            answer = answer
        )
        return self._moderate_with_llm(template)