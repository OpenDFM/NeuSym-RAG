#coding=utf8
import os, sys, json
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from annotation.moderator_prompt import MODERATE_PROMPT, EVALUATOR_PROMPT, USECASE_PROMPT
from utils.functions.common_functions import call_llm_with_pattern, call_llm, convert_to_message, call_llm_with_message

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
            messages: List[Dict[str, Any]],
            template: str
        ) -> List[Any]:
        messages = str(messages)
        if len(messages) >= 50000:
            messages = messages[:50000]
        trajectory = {
            "role": "user", 
            "content": f"Here are original trajectory where the question and answer are generated:\n```json\n{messages}\n```"
        }
        messages = convert_to_message(template)
        messages.append(trajectory)
        response = call_llm_with_message(messages, model=self.model, temperature=self.temperature)
        messages.append({"role": "assistant", "content": response})
        return messages
    
    def moderate(self, messages: List[Dict[str, Any]], question: str, answer: str) -> List[Any]:
        template = MODERATE_PROMPT.format(
            evaluator = EVALUATORS_PROMPT,
            question = question,
            answer = answer
        )
        return self._moderate_with_llm(messages, template)