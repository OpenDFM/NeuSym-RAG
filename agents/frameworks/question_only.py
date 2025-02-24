#coding=utf8
import logging, re, os, json
from typing import List, Dict, Any, Union, Tuple, Optional
from agents.envs import AgentEnv
from agents.models import LLMClient
from agents.prompts import SYSTEM_PROMPTS, AGENT_PROMPTS
from agents.prompts.task_prompt import formulate_input
from agents.frameworks import AgentBase
from utils.functions.ai_research_metadata import get_airqa_paper_metadata


logger = logging.getLogger()

class QuestionOnlyAgent(AgentBase):

    def __init__(self, model: LLMClient, env: AgentEnv, agent_method: str = 'question_only', max_turn: int = 1) -> None:
        super(QuestionOnlyAgent, self).__init__(model, env, agent_method, max_turn)

    def interact(self,
                 dataset: str,
                 example: Dict[str, Any],
                 model: str = 'gpt-4o-mini',
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 max_tokens: int = 1500,
                 image_limit: int = 10,
                 **kwargs
    ) -> str:
        question, answer_format, pdf_context, image_message = formulate_input(dataset, example, image_limit=image_limit)
        logger.info(f'[Question]: {question}')
        logger.info(f'[Answer Format]: {answer_format}')
        prev_cost = self.model.get_cost()
        self.env.reset()

        anchor_pdf, reference_pdf = example["anchor_pdf"], example["reference_pdf"]
        dataset_dir = f"./data/dataset/{self.env.dataset}"
        anchor_title, reference_title = [get_airqa_paper_metadata(uuid, dataset_dir)['title'] for uuid in anchor_pdf], [get_airqa_paper_metadata(uuid, dataset_dir)['title'] for uuid in reference_pdf]

        # Answer the question
        prompt = AGENT_PROMPTS[self.agent_method].format(
            system_prompt = SYSTEM_PROMPTS[self.agent_method],
            question = question,
            anchor_pdf=anchor_title,
            reference_pdf=reference_title,
            answer_format = answer_format
        ) # system prompt + task prompt + cot thought hints
        logger.info('Generate Answer ...')
        messages = [{'role': 'user', 'content': prompt}]
        if image_message is not None:
            messages.append(image_message)
        response = self.model.get_response(messages, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        logger.info(f'[Response]: {response}')
        matched_list = re.findall(r"```(txt)?\s*(.*?)\s*```", response.strip(), flags=re.DOTALL)
        if not matched_list:
            answer = response.strip()
        else:
            answer = matched_list[-1][1].strip()
        logger.info(f'[Answer]: {answer}')

        cost = self.model.get_cost() - prev_cost
        logger.info(f'[Info]: LLM API call costs ${cost:.6f}.')
        return answer