#coding=utf8
import logging, sys, os, re
from typing import List, Dict, Any, Union, Tuple, Optional
from agents.envs import AgentEnv
from agents.envs.actions import GenerateSQL, GenerateAnswer, ParseActionError
from agents.models import LLMClient
from agents.prompts import SYSTEM_PROMPTS, AGENT_PROMPTS
from agents.frameworks.agent_base import AgentBase


logger = logging.getLogger()

class TwoStageText2SQLRAGAgent(AgentBase):

    def __init__(self, model: LLMClient, env: AgentEnv, agent_method: str = 'two_stage', max_turn: int = 2) -> None:
        super(TwoStageText2SQLRAGAgent, self).__init__(model, env, agent_method, max_turn)

    def close(self):
        self.env.close()
        self.model.close()


    def interact(self,
                 question: str,
                 database_prompt: str,
                 answer_format: str,
                 window_size: int = 3,
                 model: str = 'gpt-4o-mini',
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 max_tokens: int = 1500
    ) -> str:
        logger.info(f'[Question]: {question}')
        logger.info(f'[Answer Format]: {answer_format}')
        prev_cost = self.model.get_cost()
        self.env.reset()

        # 1. Generate SQL
        prompt =  AGENT_PROMPTS[self.agent_method][0].format(
            system_prompt = SYSTEM_PROMPTS['two_stage_text2sql'][0],
            question = question,
            database_schema = database_prompt
        ) # system prompt + task prompt + cot thought hints
        logger.info(f'[Stage]: {1}')
        response = self.model.get_response([{
            'role': 'user', 'content': prompt
        }], model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        logger.info(f'[Response]: {response}')
        sql = re.search(r"```(sql)?\s*(.*?)\s*```", response.strip(), flags=re.DOTALL) # parse SQL from response
        if sql is None:
            raise ParseActionError("Failed to parse the SQL query from the response.")
        sql = sql.group(2).strip()
        logger.info(f'[ParsedSQL]: {sql}')

        # 2. Answer question
        action = GenerateSQL(sql=sql)
        observation = GenerateSQL.execute(action, self.env)
        prompt = AGENT_PROMPTS[self.agent_method][1].format(
            system_prompt = SYSTEM_PROMPTS['two_stage_text2sql'][1],
            question = question,
            context = [sql, observation],
            answer_format = answer_format
        ) # system prompt (without schema) + task prompt (insert SQL, observation) + cot thought hints
        logger.info(f'[Stage]: {2}')
        response = self.model.get_response([{
            'role':'user', 'content': prompt
        }], model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        logger.info(f'[Response]: {response}')
        answer = re.search(r"```(txt)?\s*(.*?)\s*```", response.strip(), flags=re.DOTALL) # extract answer from response
        if answer is None:
            raise ParseActionError("Failed to parse the answer from the response.")
        answer = answer.group(2).strip()
        logger.info(f'[Answer]: {answer}')
        return answer