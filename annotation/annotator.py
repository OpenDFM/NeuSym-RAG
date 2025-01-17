#coding=utf
import os, sys, logging, json, random, argparse, re
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import annotation.explorer, annotation.moderator
from annotation.explorer import BaseExplorer
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

class ParseError(Exception):
    pass
class BaseAnnotator(ABC):
    pid: str = None
    model: str = None
    temperature: float = None
    explorer_cls: Type[BaseExplorer] = None
    moderator_cls: Type[BaseModerator] = None

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
        self.pid = kwargs.get("pid", None)
        self.explorer_cls = explorer_cls
        self.moderator_cls = moderator_cls
    
    def _annotate(self, **kwargs) -> Dict[str, Any]:
        # Explore the paper
        for i in range(1 if self.pid else 10):
            try:
                if not self.pid:
                    pid = random.choice(used_uuids)
                else:
                    pid = self.pid
                explorer = self.explorer_cls(pid=pid, model=self.model, temperature=self.temperature)
                messages, tags = explorer.explore(**kwargs)
                pattern = r"```(txt)?\s*\[Question\]:\s*(.*?)\s*\[Answer\]:\s*(.*?)\s*\[Reasoning Steps\]:\s*(.*?)```"
                matched = re.findall(pattern, messages[-1]["content"], re.DOTALL)
                if len(matched) == 0:
                    raise ParseError(f"Failed to Parse the Response. {messages[-1]['content']}")
                question, answer, reasoning_steps = [s.strip() for s in matched[0][1:]]
                reasoning_steps = list(reasoning_steps.split("\n"))
                break
            except ParseError as e:
                logger.info(f"While exploring paper {pid}: {str(e)}")
            except ValueError as e:
                pass
            except FileNotFoundError as e:
                logger.info(f"While exploring paper {pid}: {str(e)}")
            except Exception as e:
                logger.info(f"Failed to explore the paper {pid}. {str(e)}")
        else:
            logger.info(f"Failed to explore the paper after 10 attempts.")
            return None
        
        self.pid = explorer.pid
        # Moderate the question
        moderator: BaseModerator = self.moderator_cls(model=self.model, temperature=self.temperature)
        messages = moderator.moderate(messages, question, answer)
        pattern = r"```(txt)?\s*\[question\]:\s*(.*?)\s*\[evaluator\]:\s*(.*?)\s*\[answer_format\]:\s*(.*?)\s*\[answer\]:\s*(.*?)\s*\[tag\]:\s*(.*?)```"
        matched = re.findall(pattern, messages[-1]["content"], re.DOTALL)
        if len(matched) == 0:
            logger.info(f"Failed to Parse the Response. {messages[-1]['content']}")
            return None
        question, evaluator, answer_format, answer, eval_tag = [s.strip() for s in matched[0][1:]]
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
            state = {"gpt-4o-2024-08-06": check_llm_baseline(explorer.get_title(), question, answer_format, evaluator)}
        except Exception as e:
            logger.info(f"Failed to check the LLM baseline. {str(e)}")
            return None
        
        if kwargs.get("log_dir", None):
            with open(os.path.join(kwargs["log_dir"], f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json"), "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False, indent=4)
        
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
    
    def annotate(self, write_to_json: bool = False, **kwargs):
        example = self._annotate(**kwargs)
        if not example:
            return
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--model", type=str, default=DEFAULT_LLM_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--explorer", type=str, default="SingleExplorer")
    parser.add_argument("--moderator", type=str, default="BaseModerator")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--explore_func", type=str, default=None)
    args = parser.parse_args()
    for i in range(args.n):
        try:
            SingleAnnotator(
                model=args.model, 
                temperature=args.temperature, 
                explorer_cls=getattr(annotation.explorer, args.explorer), 
                moderator_cls=getattr(annotation.moderator, args.moderator)
            ).annotate(write_to_json=True, log_dir=args.log_dir, explore_func=args.explore_func)
        except Exception as e:
            logger.info(f"Failed to annotate the paper. {str(e)}")