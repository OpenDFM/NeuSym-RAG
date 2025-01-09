#coding=utf
import os, sys, logging, json, random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from annotation.explorer import BaseExplorer, SingleExplorer
from annotation.moderator import BaseModerator
from evaluation.evaluator import evaluate_airqa
from utils.airqa_utils import generate_airqa_example_template
from utils.functions.common_functions import call_llm

logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt='[%(asctime)s][%(filename)s - %(lineno)d][%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

DEFAULT_LLM_MODEL = 'gpt-4o' # may be changed to other open-source models
DEFAULT_TEMPERATURE = 0.7

uuid_file = os.path.join("data", "dataset", "airqa", "used_uuids_100.json")
used_uuids = json.load(open(uuid_file, "r", encoding="utf-8"))

def check_llm_baseline(
        title: str,
        question: str,
        answer_format: str,
        evaluator: Dict[str, Any],
        model: str = "gpt-4o", 
        top_p: float = 0.95, 
        temperature: float = 0.7
    ):
    template = """You are an intelligent system who is expert in answering AI research related questions.

We're talking about the paper "{title}", and you should answer the following question in given format:
[Question]: {question}
[Answer Format]: {answer_format}"""
    output = call_llm(
        template.format(
            title = title, 
            question = question, 
            answer_format = answer_format
        ), 
        model = model, 
        top_p = top_p,
        temperature = temperature
    )
    return bool(evaluate_airqa(output, {"evaluator": evaluator}) > 0.5)

class BaseAnnotator(ABC):
    pid: str = None
    model: str = None
    temperature: float = None
    explorer: BaseExplorer = None
    moderator: BaseModerator = None

    def __init__(
            self, 
            model: str, 
            temperature: float,
            explorer_cls: Type[BaseExplorer],
            moderator_cls: Type[BaseModerator],
            **kwargs
        ):
        self.model = model
        self.temperature = temperature
        pid = kwargs.get("pid", None)
        if pid:
            try:
                self.explorer = explorer_cls(pid=pid, model=model, temperature=temperature)
            except Exception as e:
                logger.info(f"Failed to create an explorer with given paper uuid {pid}. {e}")
                raise RuntimeError(f"Failed to create an explorer with given paper uuid {pid}. {e}")
        else:
            for _ in range(100):
                pid = random.choice(used_uuids)
                try:
                    self.explorer = explorer_cls(pid=pid, model=model, temperature=temperature)
                    break
                except Exception as e:
                    continue
            else:
                logger.info(f"Failed to create an explorer after 100 attemps.")
                raise RuntimeError(f"Failed to create an explorer after 100 attemps.")
        logger.info(f"Explorer created successfully with paper uuid {pid}.")
        self.pid = pid
        self.moderator = moderator_cls(model=model, temperature=temperature)
    
    def _annotate(self) -> Dict[str, Any]:
        # Explore the paper
        try:
            question, answer, reasoning_steps, tags = self.explorer.explore()
            reasoning_steps = list(reasoning_steps.split("\n"))
        except Exception as e:
            logger.info(f"Failed to explore the paper {self.pid}. {str(e)}")
            return None
        
        # Moderate the question
        question, evaluator, answer_format, answer, eval_tag = self.moderator.moderate(question, answer)
        evaluator = eval(evaluator)
        tags.append(eval_tag)
        
        try:
            if evaluate_airqa(answer, {'evaluator': evaluator}) < 0.5:
                logger.info(f"Answer not valid.")
                logger.info(f"Gold Answer: {answer}")
                logger.info(f"Question: {question}")
                logger.info(f"Answer Format: {answer_format}")
                logger.info(f"Evaluator: {evaluator}")
                return None
        except Exception as e:
            logger.info(f"Failed to evaluate the answer. {str(e)}")
            return None
        
        try:
            state = {"gpt-4o-2024-08-06": check_llm_baseline(self.explorer.get_title(), question, answer_format, evaluator)}
        except Exception as e:
            logger.info(f"Failed to check the LLM baseline. {str(e)}")
            return None
        
        return {
            "question": question,
            "answer_format": answer_format,
            "tags": tags,
            "reasoning_steps": reasoning_steps,
            "evaluator": evaluator,
            "state": state,
            "annotator": self.model
        }
        

class SingleAnnotator(BaseAnnotator):
    def __init__(
            self, 
            model: str, 
            temperature: float,
            explorer_cls: Type[BaseExplorer],
            moderator_cls: Type[BaseModerator],
            **kwargs
        ):
        super().__init__(model, temperature, explorer_cls, moderator_cls, **kwargs)
    
    def annotate(self, write_to_json: bool = False):
        example = self._annotate()
        example.update({
            "anchor_pdf": [self.pid], 
            "reference_pdf": [], 
            "conference": []
        })
        # write json file
        if write_to_json:
            # question id
            qid = generate_airqa_example_template(**example)

if __name__ == "__main__":
    for i in range(10):
        try:
            SingleAnnotator(model=DEFAULT_LLM_MODEL, temperature=DEFAULT_TEMPERATURE, explorer_cls=SingleExplorer, moderator_cls=BaseModerator).annotate(write_to_json=True)
        except Exception as e:
            logger.info(f"Failed to annotate the paper. {str(e)}")
            continue