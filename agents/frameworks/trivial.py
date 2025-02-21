#coding=utf8
import logging, re, os, json
from typing import List, Dict, Any, Union, Tuple, Optional
from agents.envs import TrivialEnv
from agents.models import LLMClient
from agents.prompts import SYSTEM_PROMPTS, AGENT_PROMPTS
from agents.prompts.task_prompt import formulate_input
from agents.frameworks import AgentBase, truncate_tokens


logger = logging.getLogger()

class TrivialAgent(AgentBase):

    def __init__(self, model: LLMClient, env: TrivialEnv, agent_method: str = 'trivial', max_turn: int = 1) -> None:
        super(TrivialAgent, self).__init__(model, env, agent_method, max_turn)

    def get_pdf_content(self, idx: str) -> str:
        processed_data_dirname = os.path.join('data', 'dataset', self.env.dataset, 'processed_data')
        for filename in os.listdir(processed_data_dirname):
            if filename == f'{idx}.json':
                with open(os.path.join(processed_data_dirname, filename), 'r', encoding='utf-8') as fin:
                    processed_data = json.load(fin)
                return re.sub(r'\n+', '\n', '\n'.join(toc['title'].strip() + '\n' + toc['text'].strip() for toc in processed_data['info_from_mineru']['TOC']))
        return 'No context provided.'

    def interact(self,
                 dataset: str,
                 example: Dict[str, Any],
                 max_length: int = 16,
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

        pdf_id = example["anchor_pdf"] + example["reference_pdf"]
        
        # 1. Retrieve the PDF content (hard coding)
        if pdf_id is not None and pdf_id != []:
            observation = '\n'.join(f"PDF {idx}:\n{self.get_pdf_content(idx)}" for idx in pdf_id)
        else:
            observation = 'No context provided.'
        logger.info('[Stage 1]: Retrieve context ...')
        observation = truncate_tokens(observation, max_tokens=max_length)
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