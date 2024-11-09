#coding=utf8
import logging, sys, os, re
from typing import List, Dict, Any, Union, Tuple, Optional
from agents.envs import AgentEnv
from agents.envs.actions import RetrieveFromVectorstore, Action, Observation
from agents.models import LLMClient
from agents.prompts import SYSTEM_PROMPTS, AGENT_PROMPTS
from agents.frameworks.agent_base import AgentBase


logger = logging.getLogger()

class TwoStageText2VecRAGAgent(AgentBase):

    def __init__(self, model: LLMClient, env: AgentEnv, agent_method: str = 'two_stage_text2vec', max_turn: int = 2) -> None:
        super(TwoStageText2VecRAGAgent, self).__init__(model, env, agent_method, max_turn)


    def interact(self,
                 question: str,
                 vectorstore_prompt: str,
                 answer_format: str,
                 model: str = 'gpt-4o-mini',
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 max_tokens: int = 1500
    ) -> str:
        logger.info(f'[Question]: {question}')
        logger.info(f'[Answer Format]: {answer_format}')
        prev_cost = self.model.get_cost()
        self.env.reset()

        # 1. Generate RetriveFromVectorstore action
        action_prompt = f"Your output should follow the action format below:\n{RetrieveFromVectorstore.specification(action_format=self.env.action_format)}"
        prompt = AGENT_PROMPTS[self.agent_method][0].format(
            system_prompt = SYSTEM_PROMPTS[self.agent_method][0],
            action_prompt = action_prompt,
            question = question,
            vectorstore_prompt = vectorstore_prompt
        ) # system prompt + action_prompt + task prompt
        logger.info('[Stage]: Generate RetriveFromVectorstore action ...')
        messages = [{'role': 'user', 'content': prompt}]
        response = self.model.get_response(messages, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        logger.info(f'[Response]: {response}')
        action: Action = Action.parse_action(response, action_types=[RetrieveFromVectorstore], action_format=self.env.action_format)
        logger.info(f'[Action]: {repr(action)}')

        # 2. Answer question
        observation: Observation = action.execute(self.env)
        prompt = AGENT_PROMPTS[self.agent_method][1].format(
            system_prompt = SYSTEM_PROMPTS[self.agent_method][1],
            question = question,
            context = observation.obs_content,
            answer_format = answer_format
        ) # system prompt (without schema) + task prompt (insert SQL, observation) + cot thought hints
        logger.info(f'[Stage]: Generate Answer ...')
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