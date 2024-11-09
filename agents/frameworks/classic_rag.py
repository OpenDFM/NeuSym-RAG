#coding=utf8
import logging, sys, os, re
from typing import List, Dict, Any, Union, Tuple, Optional
from agents.envs import Text2VecEnv
from agents.envs.actions import Observation, RetrieveFromVectorstore
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
                 table_name: Union[Optional[str], List[str]] = None,
                 column_name: Union[Optional[str], List[str]] = None,
                 pdf_id: Optional[str] = None,
                 page_number: Optional[Union[str, List[str]]] = None,
                 collection_name: str = 'text_bm25_en',
                 limit: int = 2,
                 model: str = 'gpt-4o-mini',
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 max_tokens: int = 1500
    ) -> str:
        logger.info(f'[Question]: {question}')
        logger.info(f'[Answer Format]: {answer_format}')
        prev_cost = self.model.get_cost()
        self.env.reset()

        assert collection_name in self.env.get_collection_names(), f'Collection {collection_name} not found in the vectorstore environment.'

        # 1. Retrieve the result (hard coding)
        filter_conditions = []
        if pdf_id is not None:
            filter_conditions.append(f"pdf_id == '{pdf_id}'")
        if page_number is not None and page_number != []:
            page_number_filter = f'page_number in {sorted(page_number)}' if isinstance(page_number, list) else \
                f"page_number == {page_number}"
            filter_conditions.append(page_number_filter)
        if table_name is not None:
            filter_conditions.append(f"table_name == '{table_name}'")
        if column_name is not None:
            filter_conditions.append(f"column_name == '{column_name}'")
        filter_str = ' && '.join(filter_conditions) if len(filter_conditions) > 0 else ''
        action = RetrieveFromVectorstore(
            query=question,
            collection_name=collection_name,
            filter=filter_str,
            limit=limit,
            output_fields=['text']
        )
        observation: Observation = action.execute(self.env)
        logger.info(f'[Stage 1]: Retrieve top {limit} context from {collection_name} with filter {filter_str} ...')
        logger.info(f'[Retrieved Context]:\n{observation.obs_content}')

        # 2. Answer the question
        prompt = AGENT_PROMPTS[self.agent_method].format(
            system_prompt = SYSTEM_PROMPTS[self.agent_method],
            question = question,
            context = observation.obs_content,
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