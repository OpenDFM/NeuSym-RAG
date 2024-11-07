#coding=utf8
import logging, sys, os, re
from typing import List, Dict, Any, Union, Tuple, Optional
from agents.envs import Text2VecEnv
from agents.envs.actions import RetrieveFromVectorstore
from agents.models import LLMClient
from agents.prompts import SYSTEM_PROMPTS, AGENT_PROMPTS
from agents.frameworks import AgentBase


logger = logging.getLogger()

class ClassicRAGAgent(AgentBase):

    def __init__(self, model: LLMClient, env: Text2VecEnv, agent_method: str = 'classic_rag', max_turn: int = 1) -> None:
        super(ClassicRAGAgent, self).__init__(model, env, agent_method, max_turn)


    def interact(self,
                 question: str,
                 answer_format: str,
                 pdf_id: Optional[str] = None,
                 page_id: Optional[str] = None,
                 table_name: Union[Optional[str], List[str]] = None,
                 column_name: Union[Optional[str], List[str]] = None,
                 collection_name: str = 'text_bm25_en',
                 limit: int = 2,
                 model: str = 'gpt-4o-mini',
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 max_tokens: int = 1500
    ) -> str:
        logger.info(f'[Question]: {question}')
        logger.info(f'[Answer Format]: {answer_format}')
        prev_cost = self.model.get_cost()
        self.env.reset()

        # 1. Retrieve the result
        filter_conditions = []
        if pdf_id is not None:
            filter_conditions.append(f"pdf_id = '{pdf_id}'")
        if page_id is not None:
            filter_conditions.append(f"page_id = '{page_id}'")
        if table_name is not None:
            filter_conditions.append(f"table_name = '{table_name}'")
            if column_name is not None:
                filter_conditions.append(f" AND column_name = '{column_name}'")
        filter_str = ' AND '.join(filter_conditions)
        action = RetrieveFromVectorstore(
            query=question,
            collection_name=collection_name,
            filter=filter_str,
            limit=limit,
            output_fields=['text']
        )
        action.execute(self.env)
        
        # 2. Answer the question
        prompt = AGENT_PROMPTS[self.agent_method].format(
            system_prompt = SYSTEM_PROMPTS[self.agent_method],
            question = question,
            context = observation.obs_content,
            answer_format = answer_format
        ) # system prompt + task prompt + cot thought hints
        logger.info('[Stage]: Generate Answer ...')
        messages = [{'role': 'user', 'content': prompt}]
        response = self.model.get_response(messages, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        logger.info(f'[Response]: {response}')
        matched = re.search(r"```(txt)?\s*(.*?)\s*```", response.strip(), flags=re.DOTALL)
        answer = '' if matched is None else matched.group(2).strip()
        logger.info(f'[Answer]: {answer}')

        # 2. Answer question
        action = RetrieveFromVectorstore(query=question)
        observation = action.execute(self.env)
        prompt = AGENT_PROMPTS[self.agent_method][1].format(
            system_prompt = SYSTEM_PROMPTS['two_stage_text2sql'][1],
            question = question,
            sql = sql,
            context = observation.obs_content,
            answer_format = answer_format
        ) # system prompt (without schema) + task prompt (insert SQL, observation) + cot thought hints
        logger.info(f'[Stage]: Generate Answer ...')
        messages = [{'role': 'user', 'content': prompt}]
        response = self.model.get_response(messages, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        logger.info(f'[Response]: {response}')
        matched = re.search(r"```(txt)?\s*(.*?)\s*```", response.strip(), flags=re.DOTALL)
        answer = '' if matched is None else matched.group(2).strip()
        logger.info(f'[Answer]: {answer}')
        
        cost = self.model.get_cost() - prev_cost
        logger.info(f'[Info]: LLM API call costs ${cost:.6f}.')
        return answer