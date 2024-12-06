#coding=utf8
import logging, re
from typing import List, Union, Optional
from agents.envs import TrivialEnv
from agents.models import LLMClient
from agents.prompts import SYSTEM_PROMPTS, AGENT_PROMPTS
from agents.frameworks import AgentBase


logger = logging.getLogger()

class TrivialAgent(AgentBase):

    def __init__(self, model: LLMClient, env: TrivialEnv, agent_method: str = 'trivial', max_turn: int = 1) -> None:
        super(TrivialAgent, self).__init__(model, env, agent_method, max_turn)


    def interact(self,
                 question: str,
                 answer_format: str,
                 pdf_id: Optional[Union[str, List[str]]] = None,
                 page_number: Optional[Union[str, List[str]]] = None,
                 max_length: int = 30,
                 model: str = 'gpt-4o-mini',
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 max_tokens: int = 1500,
                 **kwargs
    ) -> str:
        logger.info(f'[Question]: {question}')
        logger.info(f'[Answer Format]: {answer_format}')
        prev_cost = self.model.get_cost()
        self.env.reset()

        # 1. Retrieve the PDF content (hard coding)
        if pdf_id is not None and pdf_id != []:
            if self.env.dataset == 'airqa':
                observation = '\n'.join(f"PDF {idx}:\n{self.env.pdf_contents[idx]}" for idx in pdf_id if idx in self.env.pdf_contents)
            else:
                if page_number is not None and page_number != []:
                    page_number_list = sorted(page_number) if isinstance(page_number, list) else [page_number]
                else:
                    page_number_list = sorted(self.env.pdf_contents[pdf_id].keys())
                observation = '\n'.join(f"PDF {pdf_id} page {idx}:\n{self.env.pdf_contents[pdf_id][idx]}" for idx in page_number_list if idx in self.env.pdf_contents[pdf_id])
        else:
            observation = 'No context provided.'
        logger.info('[Stage 1]: Retrieve context ...')
        observation = self.truncate_tokens(observation, max_tokens=max_length)
        logger.info(f'[Retrieved Context]: {observation}')

        # 2. Answer the question
        prompt = AGENT_PROMPTS[self.agent_method].format(
            system_prompt = SYSTEM_PROMPTS[self.agent_method],
            question = question,
            context = observation,
            answer_format = answer_format
        ) # system prompt + task prompt + cot thought hints
        logger.info('[Stage 2]: Generate Answer ...')
        messages = [{'role': 'user', 'content': prompt}]
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